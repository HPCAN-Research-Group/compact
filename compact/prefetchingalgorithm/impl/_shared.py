"""
Shared utilities for prefetching algorithm implementations.

This module provides small, well-documented helpers that were duplicated
across several implementations in this package: a short-term suppression
buffer (MRB) and a Scheduler that issues prefetch candidates while
respecting an outstanding-cap and MRB-based suppression.

Keep the public API small and stable:
 - class MRB(size=int)
 - class Scheduler(max_outstanding=int, mrbsz=int, prefetch_degree=int)
 - class SaturatingCounter(bits=int, initial_value=int)
 - address alignment helpers (block_base, region_base, etc.)

Author and paper notes at the top of individual files were preserved and
are not changed by this centralization.
"""

import logging
from typing import List

logger = logging.getLogger("compact.prefetchingalgorithm.impl._shared")


# ------------------------------
# Address alignment helpers
# ------------------------------


def align_to_block(address: int, block_size: int = 64) -> int:
    """Align address to block boundary.

    Args:
        address: Address to align.
        block_size: Block size in bytes.

    Returns:
        Block-aligned address.
    """
    return (address // block_size) * block_size


def align_to_region(address: int, region_size: int = 2048) -> int:
    """Align address to region boundary.

    Args:
        address: Address to align.
        region_size: Region size in bytes.

    Returns:
        Region-aligned address.
    """
    return (address // region_size) * region_size


def get_block_index(address: int, block_size: int = 64) -> int:
    """Get block index from address.

    Args:
        address: Memory address.
        block_size: Block size in bytes.

    Returns:
        Block index.
    """
    return address // block_size


def get_region_index(address: int, region_size: int = 2048) -> int:
    """Get region index from address.

    Args:
        address: Memory address.
        region_size: Region size in bytes.

    Returns:
        Region index.
    """
    return address // region_size


def get_region_offset(
    address: int, region_size: int = 2048, block_size: int = 64
) -> int:
    """Get block offset within region.

    Args:
        address: Memory address.
        region_size: Region size in bytes.
        block_size: Block size in bytes.

    Returns:
        Block offset within region.
    """
    return (address % region_size) // block_size


def get_page_number(address: int, page_size: int = 4096) -> int:
    """Get page number from address.

    Args:
        address: Memory address.
        page_size: Page size in bytes.

    Returns:
        Page number.
    """
    return address // page_size


def get_page_offset(address: int, page_size: int = 4096) -> int:
    """Get offset within page.

    Args:
        address: Memory address.
        page_size: Page size in bytes.

    Returns:
        Offset within page.
    """
    return address % page_size


# ------------------------------
# Saturation counter
# ------------------------------


class SaturatingCounter:
    """Saturating counter used in prefetch algorithms.

    Counter that saturates at maximum and minimum values.

    Attributes:
        value: Current counter value.
        min_value: Minimum counter value.
        max_value: Maximum counter value.
    """

    def __init__(self, bits: int = 2, initial_value: int = 0):
        """Initialize saturating counter.

        Args:
            bits: Number of bits for the counter.
            initial_value: Initial counter value.
        """
        self.max_value = (1 << bits) - 1
        self.min_value = 0
        self.value = max(self.min_value, min(self.max_value, initial_value))

    def increment(self) -> None:
        """Increment counter, saturating at maximum."""
        self.value = min(self.value + 1, self.max_value)

    def decrement(self) -> None:
        """Decrement counter, saturating at minimum."""
        self.value = max(self.value - 1, self.min_value)

    def reset(self, value: int = 0) -> None:
        """Reset counter to specified value.

        Args:
            value: Reset value.
        """
        self.value = max(self.min_value, min(self.max_value, value))

    def is_max(self) -> bool:
        """Check if counter is at maximum."""
        return self.value >= self.max_value

    def is_min(self) -> bool:
        """Check if counter is at minimum."""
        return self.value <= self.min_value


# ------------------------------
# MRB
# ------------------------------


class MRB:
    """Miss-Resolution / short-term suppression buffer.

    Keeps a small, ordered list of recent addresses to avoid immediately
    re-issuing the same prefetches. The container preserves insertion order
    and evicts the oldest element when full.
    """

    def __init__(self, size: int = 64):
        self.size = int(size)
        self.buf: List[int] = []

    def insert(self, addr: int) -> None:
        """Insert an address into the MRB, moving it to most-recent if present."""
        if self.size <= 0:  # a zero-sized MRB suppresses nothing
            return
        if addr in self.buf:
            self.buf.remove(addr)
            self.buf.append(addr)
            return
        if len(self.buf) >= self.size:
            self.buf.pop(0)
        self.buf.append(addr)
        logger.debug("MRB: insert %x", addr)

    def contains(self, addr: int) -> bool:
        """Return True if address is present in MRB."""
        return addr in self.buf

    def remove(self, addr: int) -> None:
        """Remove address if present."""
        if addr in self.buf:
            self.buf.remove(addr)

    def clear(self) -> None:
        """Clear MRB contents."""
        self.buf.clear()


class Scheduler:
    """Simple scheduler that issues prefetches subject to caps and MRB suppression.

    Usage:
      s = Scheduler(max_outstanding=32, mrbsz=64, prefetch_degree=4)
      issued = s.issue(candidates, degree=None)  # degree defaults to prefetch_degree
      s.credit(addr)  # called when a prefetched address is used
      s.new_cycle()  # optional: reset per-cycle recent-set state
    """

    def __init__(
        self, max_outstanding: int = 32, mrbsz: int = 64, prefetch_degree: int = 4
    ):
        self.max_outstanding = int(max_outstanding)
        self.prefetch_degree = int(prefetch_degree)
        self.outstanding: List[int] = []
        self.mrb = MRB(size=mrbsz)

    def can_issue(self) -> bool:
        """Return True if we can issue more outstanding prefetches."""
        return len(self.outstanding) < self.max_outstanding

    def issue(self, candidates: List[int], degree: int = None) -> List[int]:
        """Issue addresses from candidates respecting MRB and outstanding cap.

        Args:
            candidates: candidate addresses in preferred order.
            degree: optional limit for number of prefetches to issue; if None uses
                    the scheduler's default prefetch_degree.

        Returns:
            A list of addresses that were actually issued.
        """
        if degree is None:
            degree = self.prefetch_degree
        issued: List[int] = []
        for a in candidates:
            if len(issued) >= degree:
                break
            if a in self.outstanding:
                continue
            if self.mrb.contains(a):
                continue
            if not self.can_issue():
                break
            self.outstanding.append(a)
            self.mrb.insert(a)
            issued.append(a)
            logger.info("Scheduler issued prefetch %x", a)
        return issued

    def credit(self, addr: int) -> None:
        """Called when a prefetched address is used; remove from outstanding and keep in MRB."""
        if addr in self.outstanding:
            self.outstanding.remove(addr)
            logger.debug("Scheduler credited outstanding for %x", addr)
        # keep it in MRB to avoid immediate re-issue
        self.mrb.insert(addr)

    def clear(self) -> None:
        """Reset scheduler state."""
        self.outstanding.clear()
        self.mrb.clear()

    def new_cycle(self) -> None:
        """Compatibility helper: called once per cycle to expire MRB/recency state.

        Some implementations use a "new_cycle" API to clear per-cycle recent-sets.
        Provide the method for compatibility with existing code that expects it.
        """
        self.mrb.clear()  # clear MRB to expire recent entries
