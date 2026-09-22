"""
Pythia: a customizable hardware prefetching framework using online
reinforcement learning.

Based on: Rahul Bera, Konstantinos Kanellopoulos, Anant V. Nori, Taha
Shahroodi, Sreenivas Subramoney, Onur Mutlu. "Pythia: A Customizable
Hardware Prefetching Framework Using Online Reinforcement Learning".
MICRO 2021. https://doi.org/10.1145/3466752.3480114
Reference implementation: https://github.com/CMU-SAFARI/Pythia

Overview:
- Pythia casts prefetching as a reinforcement-learning problem. Every demand
  request forms a state S built from program context; the agent picks an
  action A from a fixed list of prefetch offsets; the prefetch that action
  generates is graded later and turned into a numerical reward that folds in
  both prefetch quality (timely / late / useless) and system state (memory
  bandwidth usage).
- The agent learns online with SARSA and an epsilon-greedy policy:
      Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
- Q-values live in the QVStore: one vault per state feature, each vault a
  set of tile-coded planes. A feature-action Q-value is the sum of the
  partial Q-values held by every plane of its vault, and the state-action
  Q-value is the maximum over vaults:  Q(S, A) = max_i Q(phi_i(S), A).
  Tile coding lets similar feature values share partial Q-values (faster
  training) while multiple planes keep unrelated values apart.
- Rewards are delayed, so every action taken is parked in the Evaluation
  Queue (EQ) until either a demand request hits its prefetch address
  (accurate) or the entry is evicted still unrewarded (inaccurate). The
  SARSA update happens on eviction, using the state-action pair of the
  access that evicts the entry as (S', A').

The paper's basic configuration (Table 2) is the default here:
    features      PC+Delta, sequence of last-4 deltas
    actions       {-6,-3,-1,0,1,3,4,5,10,11,12,16,22,23,30,32}
    rewards       R_AT=20, R_AL=12, R_CL=-12, R_IN^H=-14, R_IN^L=-8,
                  R_NP^H=-2, R_NP^L=-4
    hyperparams   alpha=0.0065, gamma=0.556, epsilon=0.002

Those hyperparameters were tuned for billion-instruction simulations. On
Compact-scale traces (10^4-10^5 accesses) an epsilon of 0.002 explores too
rarely for the agent to move off the first action that earns a positive
reward, so `sample_pythia.yml` raises epsilon to 0.05; raise `alpha` too if
the trace is shorter still.

Simulator adaptations (Compact is trace-driven -- addresses in, addresses
out, with no cache or DRAM timing model):
- Timeliness. Real Pythia calls a prefetch timely when it filled the cache
  before the demand arrived, and late when the demand caught it still in
  flight. Compact models no fill latency, so nothing here can be in flight;
  what it does have is a usefulness horizon, since its analyzer credits a
  prefetch only if the very next access demands it. Timeliness is therefore
  mapped onto that horizon: a prefetch demanded within `timeliness_window`
  accesses of being issued is timely (R_AT), and one demanded later, while
  still tracked by the EQ, is accurate but late (R_AL).
- Memory bandwidth level. Derived from the fraction of the last
  `bandwidth_window` accesses that issued at least one prefetch; above
  `bandwidth_threshold` the high-bandwidth reward levels apply.
- Omitted: the pipelined QVStore search, the storage/latency/area budgets,
  and the multi-core DRAM bandwidth counters, none of which have a
  counterpart in a trace-driven, address-in/address-out simulator.
"""

import logging
import random
from typing import Dict, List, Optional, Sequence, Tuple

from compact.prefetchingalgorithm.impl._shared import MRB
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess
from compact.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

logger = logging.getLogger("compact.prefetchingalgorithm.impl.pythia")

# Prefetch action list from the paper's basic configuration. Action 0 means
# "do not prefetch".
ACTIONS: Tuple[int, ...] = (-6, -3, -1, 0, 1, 3, 4, 5, 10, 11, 12, 16, 22, 23, 30, 32)

_MASK64 = (1 << 64) - 1


def _mix(value: int) -> int:
    """Deterministic 64-bit avalanche hash (splitmix64 finalizer).

    Used instead of Python's ``hash`` so tile indices stay reproducible
    across runs and interpreters.
    """
    value &= _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


class Vault:
    """Q-values of one state feature, tile-coded across several planes.

    Each plane is a ``num_tiles x num_actions`` table of partial Q-values.
    A feature value is shifted by that plane's design-time constant and
    hashed to a tile index, so the feature-action Q-value is the sum of the
    partial Q-values read out of every plane.
    """

    def __init__(self, num_actions: int, num_planes: int, num_tiles: int, rng: random.Random):
        self.num_tiles = num_tiles
        self.planes: List[List[List[float]]] = [
            [[0.0] * num_actions for _ in range(num_tiles)] for _ in range(num_planes)
        ]
        self.shifts: List[int] = [rng.getrandbits(32) for _ in range(num_planes)]

    def _tile(self, feature: int, shift: int) -> int:
        return _mix(feature + shift) % self.num_tiles

    def q(self, feature: int, action_index: int) -> float:
        """Feature-action Q-value: the sum of every plane's partial Q-value."""
        return sum(
            plane[self._tile(feature, shift)][action_index]
            for plane, shift in zip(self.planes, self.shifts)
        )

    def update(self, feature: int, action_index: int, amount: int) -> None:
        """Spread a Q-value delta evenly over the planes."""
        share = amount / len(self.planes)
        for plane, shift in zip(self.planes, self.shifts):
            plane[self._tile(feature, shift)][action_index] += share


class QVStore:
    """One vault per state feature; Q(S, A) = max over vaults."""

    def __init__(
        self, num_features: int, num_actions: int, num_planes: int, num_tiles: int, rng: random.Random
    ):
        self.num_actions = num_actions
        self.vaults = [Vault(num_actions, num_planes, num_tiles, rng) for _ in range(num_features)]

    def q(self, features: Sequence[int], action_index: int) -> float:
        return max(vault.q(f, action_index) for vault, f in zip(self.vaults, features))

    def q_values(self, features: Sequence[int]) -> List[float]:
        return [self.q(features, a) for a in range(self.num_actions)]

    def update(self, features: Sequence[int], action_index: int, amount: int) -> None:
        for vault, f in zip(self.vaults, features):
            vault.update(f, action_index, amount)


class SignatureTable:
    """Per-page access history feeding the state features.

    Holds the last offset touched in the page and the page's recent deltas,
    which is all the two default features need.
    """

    def __init__(self, size: int, delta_history: int):
        self.size = size
        self.delta_history = delta_history
        self.entries: Dict[int, Dict[str, object]] = {}

    def update(self, page: int, offset: int) -> Tuple[int, Tuple[int, ...]]:
        """Record an access and return (delta, delta history) for the page."""
        entry = self.entries.pop(page, None)
        if entry is None:
            if len(self.entries) >= self.size:
                self.entries.pop(next(iter(self.entries)))
            entry = {"last_offset": offset, "deltas": []}
            delta = 0
        else:
            delta = offset - int(entry["last_offset"])
            entry["last_offset"] = offset
            if delta:
                deltas: List[int] = entry["deltas"]  # type: ignore[assignment]
                deltas.append(delta)
                del deltas[: -self.delta_history]
        self.entries[page] = entry  # reinsert as most-recently-used
        return delta, tuple(entry["deltas"])  # type: ignore[arg-type]


class EQEntry:
    """One in-flight action awaiting its reward."""

    __slots__ = ("features", "action_index", "address", "issued_at", "reward")

    def __init__(
        self,
        features: Tuple[int, ...],
        action_index: int,
        address: Optional[int],
        issued_at: int,
        reward: Optional[int] = None,
    ):
        self.features = features
        self.action_index = action_index
        self.address = address
        self.issued_at = issued_at
        self.reward = reward


class PythiaPrefetcher(PrefetchAlgorithm):
    """Online RL prefetcher following the MICRO 2021 Pythia design."""

    def __init__(
        self,
        block_size: int = 64,
        page_size: int = 4096,
        actions: Sequence[int] = ACTIONS,
        alpha: float = 0.0065,
        gamma: float = 0.556,
        epsilon: float = 0.002,
        num_planes: int = 3,
        num_tiles: int = 128,
        st_size: int = 64,
        eq_size: int = 256,
        delta_history: int = 4,
        timeliness_window: int = 1,
        bandwidth_window: int = 64,
        bandwidth_threshold: float = 0.5,
        max_to_avg_q_thresholds: Sequence[float] = (0.5, 1.0, 2.0),
        dynamic_degrees: Sequence[int] = (1, 2, 4, 4),
        enable_dynamic_degree: bool = True,
        prefetch_degree: int = 1,
        mrb_size: int = 2,
        enable_tracker_hit_reward: bool = False,
        reward_accurate_timely: int = 20,
        reward_accurate_late: int = 12,
        reward_loss_of_coverage: int = -12,
        reward_inaccurate_low_bw: int = -8,
        reward_inaccurate_high_bw: int = -14,
        reward_no_prefetch_low_bw: int = -4,
        reward_no_prefetch_high_bw: int = -2,
        reward_tracker_hit: int = -2,
        seed: int = 200,
    ):
        self.block_size = block_size
        self.page_size = page_size
        self.blocks_per_page = page_size // block_size
        self.actions = tuple(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_planes = num_planes
        self.num_tiles = num_tiles
        self.st_size = st_size
        self.eq_size = eq_size
        self.delta_history = delta_history
        self.timeliness_window = timeliness_window
        self.bandwidth_window = bandwidth_window
        self.bandwidth_threshold = bandwidth_threshold
        self.max_to_avg_q_thresholds = tuple(max_to_avg_q_thresholds)
        self.dynamic_degrees = tuple(dynamic_degrees)
        self.enable_dynamic_degree = enable_dynamic_degree
        self.prefetch_degree = prefetch_degree
        self.mrb_size = mrb_size
        self.enable_tracker_hit_reward = enable_tracker_hit_reward
        self.reward_accurate_timely = reward_accurate_timely
        self.reward_accurate_late = reward_accurate_late
        self.reward_loss_of_coverage = reward_loss_of_coverage
        self.reward_inaccurate_low_bw = reward_inaccurate_low_bw
        self.reward_inaccurate_high_bw = reward_inaccurate_high_bw
        self.reward_no_prefetch_low_bw = reward_no_prefetch_low_bw
        self.reward_no_prefetch_high_bw = reward_no_prefetch_high_bw
        self.reward_tracker_hit = reward_tracker_hit
        self.seed = seed

        self.init()

    # ------------------------------------------------------------------
    # PrefetchAlgorithm interface
    # ------------------------------------------------------------------

    def init(self):
        """Reset the agent: empty QVStore, signature table, and EQ."""
        self.rng = random.Random(self.seed)
        # Two vaults, one per default feature: PC+Delta and last-N deltas.
        self.qvstore = QVStore(2, len(self.actions), self.num_planes, self.num_tiles, self.rng)
        self.st = SignatureTable(self.st_size, self.delta_history)
        self.eq: List[EQEntry] = []
        # Unrewarded EQ entries indexed by prefetch address, so a demand can
        # find what it resolves without scanning the whole queue.
        self.pending: Dict[int, List[EQEntry]] = {}
        self.mrb = MRB(size=self.mrb_size)
        self.clock = 0
        self.recent_issues: List[int] = []
        self.stats = {"issued": 0, "timely": 0, "late": 0, "inaccurate": 0, "no_prefetch": 0}

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Take one action for this demand access and return its prefetches.

        ``prefetch_hit`` is unused: Pythia grades its own prefetches through
        the Evaluation Queue rather than relying on the simulator's verdict
        for the previous access.
        """
        self.clock += 1
        page, offset = divmod(access.address // self.block_size, self.blocks_per_page)
        delta, delta_path = self.st.update(page, offset)

        features = self._features(access.pc, delta, delta_path)
        q_values = self.qvstore.q_values(features)
        action_index = self._select_action(q_values)

        # Reward pending entries this access resolves, then retire the ones
        # that aged out. Both learn against the freshly chosen (S', A').
        self._reward_matches(access.address)
        self._retire(features, action_index)

        prefetches = self._issue(page, offset, features, action_index, q_values)
        self._track_bandwidth(bool(prefetches))
        return prefetches

    def close(self):
        """Log what the agent did over the run."""
        logger.info(
            "Pythia: issued=%d timely=%d late=%d inaccurate=%d no_prefetch=%d",
            self.stats["issued"],
            self.stats["timely"],
            self.stats["late"],
            self.stats["inaccurate"],
            self.stats["no_prefetch"],
        )

    # ------------------------------------------------------------------
    # State, policy, and prefetch generation
    # ------------------------------------------------------------------

    def _features(self, pc: int, delta: int, delta_path: Tuple[int, ...]) -> Tuple[int, int]:
        """The paper's winning state-vector: PC+Delta and last-4 deltas."""
        pc_delta = _mix((pc << 16) ^ (delta & 0xFFFF))
        path = 0
        for d in delta_path:
            path = _mix((path << 8) ^ (d & 0xFF))
        return pc_delta, path

    def _select_action(self, q_values: Sequence[float]) -> int:
        """Epsilon-greedy over the state's action Q-values."""
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.actions))
        best = max(q_values)
        return q_values.index(best)

    def _degree(self, q_values: Sequence[float]) -> int:
        """Prefetch degree from the max-to-average Q-value ratio.

        A state whose best action stands far above the average is one the
        agent is confident about, so it is allowed to prefetch deeper.
        """
        if not self.enable_dynamic_degree:
            return self.prefetch_degree
        average = sum(q_values) / len(q_values)
        if average <= 0:
            return self.dynamic_degrees[0]
        ratio = max(q_values) / average
        for threshold, degree in zip(self.max_to_avg_q_thresholds, self.dynamic_degrees):
            if ratio <= threshold:
                return degree
        return self.dynamic_degrees[-1]

    def _issue(
        self,
        page: int,
        offset: int,
        features: Tuple[int, int],
        action_index: int,
        q_values: Sequence[float],
    ) -> List[int]:
        """Turn the chosen action into page-bounded prefetch addresses."""
        action = self.actions[action_index]
        if action == 0:
            # The agent chose not to prefetch.
            self.stats["no_prefetch"] += 1
            self._enqueue(features, action_index, None, self._reward_no_prefetch())
            return []

        prefetches: List[int] = []
        for step in range(1, self._degree(q_values) + 1):
            predicted = offset + action * step
            if not 0 <= predicted < self.blocks_per_page:
                # Pythia never prefetches across a page boundary; the action
                # loses the coverage it could have had.
                self._enqueue(features, action_index, None, self.reward_loss_of_coverage)
                break
            address = (page * self.page_size) + (predicted * self.block_size)
            if self.mrb.contains(address):
                # Already prefetched very recently; suppress the duplicate.
                if self.enable_tracker_hit_reward:
                    self._enqueue(features, action_index, None, self.reward_tracker_hit)
                continue
            self.mrb.insert(address)
            self._enqueue(features, action_index, address, None)
            prefetches.append(address)
            self.stats["issued"] += 1
            logger.debug("Pythia: action %+d -> prefetch %#x", action, address)
        return prefetches

    # ------------------------------------------------------------------
    # Evaluation queue and learning
    # ------------------------------------------------------------------

    def _enqueue(
        self,
        features: Tuple[int, int],
        action_index: int,
        address: Optional[int],
        reward: Optional[int],
    ) -> None:
        entry = EQEntry(features, action_index, address, self.clock, reward)
        self.eq.append(entry)
        if address is not None and reward is None:
            self.pending.setdefault(address, []).append(entry)

    def _reward_matches(self, address: int) -> None:
        """Grade every pending entry whose prefetch this demand just used."""
        for entry in self.pending.pop(address, ()):
            if self.clock - entry.issued_at <= self.timeliness_window:
                entry.reward = self.reward_accurate_timely
                self.stats["timely"] += 1
            else:
                # Accurate, but demanded past the horizon the simulator
                # credits: the prefetch arrived, just not when it was needed.
                entry.reward = self.reward_accurate_late
                self.stats["late"] += 1

    def _retire(self, features: Tuple[int, int], action_index: int) -> None:
        """Evict aged-out EQ entries, learning from each one as it goes."""
        while len(self.eq) > self.eq_size:
            entry = self.eq.pop(0)
            if entry.reward is None:
                # Never demanded while it was being tracked.
                waiting = self.pending.get(entry.address)
                if waiting is not None:
                    waiting.remove(entry)
                    if not waiting:
                        del self.pending[entry.address]
                entry.reward = self._reward_inaccurate()
                self.stats["inaccurate"] += 1
            self._learn(entry, features, action_index)

    def _learn(self, entry: EQEntry, next_features: Tuple[int, int], next_action: int) -> None:
        """SARSA: Q(S,A) += alpha * [R + gamma * Q(S',A') - Q(S,A)]."""
        target = entry.reward + self.gamma * self.qvstore.q(next_features, next_action)
        td_error = target - self.qvstore.q(entry.features, entry.action_index)
        self.qvstore.update(entry.features, entry.action_index, self.alpha * td_error)

    # ------------------------------------------------------------------
    # Memory bandwidth proxy
    # ------------------------------------------------------------------

    def _track_bandwidth(self, issued: bool) -> None:
        self.recent_issues.append(1 if issued else 0)
        del self.recent_issues[: -self.bandwidth_window]

    def _is_high_bandwidth(self) -> bool:
        if not self.recent_issues:
            return False
        return sum(self.recent_issues) / len(self.recent_issues) > self.bandwidth_threshold

    def _reward_inaccurate(self) -> int:
        return (
            self.reward_inaccurate_high_bw
            if self._is_high_bandwidth()
            else self.reward_inaccurate_low_bw
        )

    def _reward_no_prefetch(self) -> int:
        # Sitting out costs less when bandwidth is scarce, so the
        # high-bandwidth penalty is the milder of the two.
        return (
            self.reward_no_prefetch_high_bw
            if self._is_high_bandwidth()
            else self.reward_no_prefetch_low_bw
        )
