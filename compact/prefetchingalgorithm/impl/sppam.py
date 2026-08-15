"""
SPPAM: Signature Pattern Prediction and Access-Map Prefetcher.

Based on the HPCA 2026 Data Prefetching Championship proposal.

Overview:
- Signature Pattern Prediction (SPP): per-PC delta history is folded into a
  compact signature; a (signature -> delta -> confidence) pattern table is
  trained online and chased forward to predict a run of future deltas.
- Access-Map Pattern (AMP): a per-region bitmap of already-seen block
  offsets is used to extrapolate the stride between recently accessed
  blocks within the same region, independent of PC.
- Confidence-based throttle: how many predictions a signature is allowed to
  emit scales with its learned confidence, and final issue is gated through
  the shared Scheduler (outstanding cap + MRB de-duplication).

References:
- Signature Path Prefetcher (SPP), Kim et al., ISCA 2016.
- Access Map Pattern Matching (AMPM), Ishii et al., ISCA 2009.
"""

from typing import Dict, List, Optional, Tuple

from compact.prefetchingalgorithm.impl._shared import (
    SaturatingCounter,
    Scheduler,
    get_region_index,
    get_region_offset,
)
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess
from compact.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

PatternTable = Dict[int, Dict[int, SaturatingCounter]]


class SPPAMPrefetcher(PrefetchAlgorithm):
    """Combines signature-based delta prediction with access-map extrapolation."""

    def __init__(
        self,
        signature_bits: int = 8,
        signature_shift: int = 3,
        confidence_threshold: float = 0.5,
        max_prefetch_degree: int = 4,
        region_size: int = 2048,
        block_size: int = 64,
        mrb_size: int = 64,
        max_outstanding: int = 32,
    ):
        """Initialize the SPPAM prefetcher.

        Args:
            signature_bits: Width of the folded delta-history signature.
            signature_shift: Left-shift applied when folding a new delta in.
            confidence_threshold: Minimum confidence (0-1) to act on a prediction.
            max_prefetch_degree: Upper bound on predictions issued per access.
            region_size: Access-map region size in bytes.
            block_size: Cache block size in bytes.
            mrb_size: Size of the suppression buffer shared with the scheduler.
            max_outstanding: Max outstanding prefetches tracked by the scheduler.
        """
        self.signature_mask = (1 << signature_bits) - 1
        self.signature_shift = signature_shift
        self.confidence_threshold = confidence_threshold
        self.max_prefetch_degree = max_prefetch_degree
        self.region_size = region_size
        self.block_size = block_size
        self.blocks_per_region = region_size // block_size

        # Per-PC state: last seen block address and current signature.
        self.last_block: Dict[int, int] = {}
        self.signature: Dict[int, int] = {}

        # SPP pattern table: pc -> signature -> delta -> confidence counter.
        self.pattern_table: Dict[int, PatternTable] = {}

        # AMP access-map: region index -> bitmap of accessed block offsets.
        self.region_map: Dict[int, int] = {}

        self.scheduler = Scheduler(
            max_outstanding=max_outstanding,
            mrbsz=mrb_size,
            prefetch_degree=max_prefetch_degree,
        )

    def init(self) -> None:
        """Reset all learned state."""
        self.last_block.clear()
        self.signature.clear()
        self.pattern_table.clear()
        self.region_map.clear()
        self.scheduler.clear()

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process a memory access and return addresses to prefetch."""
        pc = access.pc
        block = (access.address // self.block_size) * self.block_size

        candidates: List[int] = []
        candidates.extend(self._spp_predict(pc, block))
        candidates.extend(self._amp_predict(access.address))

        return self.scheduler.issue(candidates, degree=self.max_prefetch_degree)

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Signature Pattern Prediction (SPP)
    # ------------------------------------------------------------------

    def _fold_signature(self, signature: int, delta: int) -> int:
        return ((signature << self.signature_shift) ^ (delta & self.signature_mask)) & (
            self.signature_mask
        )

    def _best_delta(
        self, table: PatternTable, signature: int
    ) -> Optional[Tuple[int, SaturatingCounter]]:
        deltas = table.get(signature)
        if not deltas:
            return None
        delta, counter = max(deltas.items(), key=lambda item: item[1].value)
        confidence = counter.value / counter.max_value
        if confidence < self.confidence_threshold:
            return None
        return delta, confidence

    def _spp_predict(self, pc: int, block: int) -> List[int]:
        table = self.pattern_table.setdefault(pc, {})
        last_block = self.last_block.get(pc)
        self.last_block[pc] = block
        if last_block is None:
            self.signature.setdefault(pc, 0)
            return []

        delta = block - last_block
        signature = self.signature.get(pc, 0)
        counter = table.setdefault(signature, {}).setdefault(
            delta, SaturatingCounter(bits=2)
        )
        counter.increment()

        new_signature = self._fold_signature(signature, delta)
        self.signature[pc] = new_signature

        # Confidence-based throttle: chase the pattern table for a run of
        # deltas, with run length scaled by how confident the first hop is.
        best = self._best_delta(table, new_signature)
        if best is None:
            return []
        _, confidence = best
        depth = max(1, round(confidence * self.max_prefetch_degree))

        predictions: List[int] = []
        cur_signature = new_signature
        predicted_block = block
        for _ in range(depth):
            best = self._best_delta(table, cur_signature)
            if best is None:
                break
            next_delta, _ = best
            predicted_block += next_delta
            predictions.append(predicted_block)
            cur_signature = self._fold_signature(cur_signature, next_delta)
        return predictions

    # ------------------------------------------------------------------
    # Access-Map Pattern (AMP)
    # ------------------------------------------------------------------

    def _amp_predict(self, address: int) -> List[int]:
        region_index = get_region_index(address, self.region_size)
        offset = get_region_offset(address, self.region_size, self.block_size)

        bitmap = self.region_map.get(region_index, 0)
        bitmap |= 1 << offset
        self.region_map[region_index] = bitmap

        set_offsets = [i for i in range(self.blocks_per_region) if bitmap & (1 << i)]
        if len(set_offsets) < 2:
            return []

        strides = [b - a for a, b in zip(set_offsets, set_offsets[1:])]
        stride = max(set(strides), key=strides.count)

        next_offset = offset + stride
        if not (0 <= next_offset < self.blocks_per_region):
            return []
        if bitmap & (1 << next_offset):
            return []

        region_base = region_index * self.region_size
        return [region_base + next_offset * self.block_size]
