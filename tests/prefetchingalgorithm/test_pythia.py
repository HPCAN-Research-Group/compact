import random

from compact.prefetchingalgorithm.impl.pythia import (
    ACTIONS,
    PythiaPrefetcher,
    QVStore,
)
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess

BLOCK = 64
PAGE = 4096


def _access(index, pc=0x400, base=0x10000):
    return MemoryAccess(pc=pc, address=base + index * BLOCK)


def test_qvstore_sums_planes_and_maxes_vaults():
    store = QVStore(num_features=2, num_actions=4, num_planes=3, num_tiles=16, rng=random.Random(1))
    # A vault spreads an update over its planes, so the feature-action
    # Q-value reads back as the whole amount.
    store.vaults[0].update(feature=123, action_index=2, amount=9.0)
    assert store.vaults[0].q(123, 2) == 9.0
    # Q(S, A) is the max over vaults, and the untouched vault still reads 0.
    assert store.q((123, 456), 2) == 9.0
    store.vaults[1].update(feature=456, action_index=2, amount=20.0)
    assert store.q((123, 456), 2) == 20.0


def test_greedy_action_drives_the_prefetch_address():
    p = PythiaPrefetcher(epsilon=0.0, enable_dynamic_degree=False, prefetch_degree=1)

    # Make +5 the best action for the state the first access produces.
    access = _access(0)
    features = p._features(access.pc, 0, ())
    p.qvstore.update(features, ACTIONS.index(5), 100.0)

    assert p.progress(access, prefetch_hit=False) == [0x10000 + 5 * BLOCK]


def test_prefetch_never_crosses_the_page_boundary():
    p = PythiaPrefetcher(epsilon=0.0, enable_dynamic_degree=False, prefetch_degree=1)

    # Second-to-last line of a page; +32 would land in the next page.
    address = 0x10000 + PAGE - 2 * BLOCK
    features = p._features(0x400, 0, ())
    p.qvstore.update(features, ACTIONS.index(32), 100.0)

    assert p.progress(MemoryAccess(pc=0x400, address=address), prefetch_hit=False) == []


def test_action_zero_issues_nothing_and_is_rewarded_as_no_prefetch():
    p = PythiaPrefetcher(epsilon=0.0, enable_dynamic_degree=False)
    features = p._features(0x400, 0, ())
    p.qvstore.update(features, ACTIONS.index(0), 100.0)

    assert p.progress(_access(0), prefetch_hit=False) == []
    assert p.stats["no_prefetch"] == 1
    assert p.eq[-1].reward == p.reward_no_prefetch_low_bw


def test_dynamic_degree_grows_with_confidence():
    p = PythiaPrefetcher(epsilon=0.0)
    flat = [1.0] * len(ACTIONS)
    peaked = [1.0] * len(ACTIONS)
    peaked[4] = 50.0

    # max/avg is 1.0 when every action looks equally good, and grows as one
    # action pulls ahead.
    assert p._degree(flat) < p._degree(peaked)
    assert p._degree(peaked) == p.dynamic_degrees[-1]
    # A state whose actions all look bad falls back to the shallowest degree.
    assert p._degree([-5.0] * len(ACTIONS)) == p.dynamic_degrees[0]


def test_accurate_prefetch_is_rewarded_and_raises_its_q_value():
    # A one-entry EQ makes the reward land on the very next access.
    p = PythiaPrefetcher(epsilon=0.0, eq_size=1, enable_dynamic_degree=False, prefetch_degree=1)
    features = p._features(0x400, 0, ())
    action = ACTIONS.index(1)
    p.qvstore.update(features, action, 1.0)
    before = p.qvstore.q(features, action)

    p.progress(_access(0), prefetch_hit=False)
    p.progress(_access(1), prefetch_hit=False)  # demands the prefetched line
    p.progress(_access(2), prefetch_hit=False)  # evicts the entry, applying SARSA

    assert p.stats["timely"] >= 1
    assert p.qvstore.q(features, action) > before


def test_useless_prefetch_is_penalised():
    p = PythiaPrefetcher(epsilon=0.0, eq_size=1, enable_dynamic_degree=False, prefetch_degree=1)
    features = p._features(0x400, 0, ())
    action = ACTIONS.index(30)
    p.qvstore.update(features, action, 1.0)
    before = p.qvstore.q(features, action)

    p.progress(_access(0), prefetch_hit=False)
    p.progress(_access(1), prefetch_hit=False)
    p.progress(_access(2), prefetch_hit=False)

    assert p.stats["inaccurate"] >= 1
    assert p.qvstore.q(features, action) < before


def test_high_bandwidth_switches_the_reward_level():
    p = PythiaPrefetcher(bandwidth_window=4, bandwidth_threshold=0.5)
    p.recent_issues = [0, 0, 0, 0]
    assert p._reward_inaccurate() == p.reward_inaccurate_low_bw
    assert p._reward_no_prefetch() == p.reward_no_prefetch_low_bw

    p.recent_issues = [1, 1, 1, 0]
    assert p._reward_inaccurate() == p.reward_inaccurate_high_bw
    assert p._reward_no_prefetch() == p.reward_no_prefetch_high_bw


def test_tracker_suppresses_a_duplicate_prefetch():
    p = PythiaPrefetcher(epsilon=0.0, enable_dynamic_degree=False, prefetch_degree=1, mrb_size=8)
    features = p._features(0x400, 0, ())
    p.qvstore.update(features, ACTIONS.index(1), 100.0)

    first = p.progress(_access(0), prefetch_hit=False)
    # Same page, same PC, but a delta now feeds the state; force the same
    # action so the prefetch address repeats and the tracker filters it.
    repeat_features = p._features(0x400, 1, (1,))
    p.qvstore.update(repeat_features, ACTIONS.index(-1), 100.0)
    second = p.progress(_access(2), prefetch_hit=False)

    assert first == [0x10000 + BLOCK]
    assert second == []  # 0x10040 was just issued


def test_agent_learns_a_sequential_stream():
    p = PythiaPrefetcher(epsilon=0.05, mrb_size=2)
    hits = []
    predicted = []
    for i in range(8000):
        access = _access(i)
        hits.append(access.address in predicted)
        predicted = p.progress(access, prefetch_hit=False)

    # The agent starts cold, so judge it on the tail of the trace.
    assert sum(hits[-2000:]) / 2000 > 0.5
    q_values = p.qvstore.q_values(p._features(0x400, 1, (1, 1, 1, 1)))
    assert ACTIONS[q_values.index(max(q_values))] == 1


def test_runs_are_reproducible_for_a_fixed_seed():
    def trace():
        p = PythiaPrefetcher(seed=7)
        return [p.progress(_access(i), prefetch_hit=False) for i in range(200)]

    assert trace() == trace()
