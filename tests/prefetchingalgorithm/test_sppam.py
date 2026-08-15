from compact.prefetchingalgorithm.impl.sppam import SPPAMPrefetcher
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


def test_sppam_basic_pattern():
    p = SPPAMPrefetcher(confidence_threshold=0.0)  # Always predict
    p.init()

    pc = 0x10
    # Sequential pattern
    addrs = [0x1000, 0x1040, 0x1080]

    for addr in addrs:
        p.progress(MemoryAccess(pc=pc, address=addr), prefetch_hit=False)

    # Check if it predicts 0x10C0
    preds = p.progress(MemoryAccess(pc=pc, address=0x10C0), prefetch_hit=False)
    assert 0x1100 in preds


def test_sppam_high_threshold_suppresses_unlearned_pattern():
    # Confidence starts low; a threshold near max should suppress SPP
    # predictions until the pattern table has been reinforced enough.
    # Strides span multiple regions so the AMP component (which is not
    # confidence-gated) never sees two offsets in the same region.
    p = SPPAMPrefetcher(confidence_threshold=0.99)
    p.init()

    pc = 0x20
    preds = p.progress(MemoryAccess(pc=pc, address=0x2000), prefetch_hit=False)
    assert preds == []
    preds = p.progress(MemoryAccess(pc=pc, address=0x2000 + 4096), prefetch_hit=False)
    assert preds == []


def test_sppam_init_resets_state():
    p = SPPAMPrefetcher(confidence_threshold=0.0)
    p.init()

    pc = 0x30
    for addr in (0x3000, 0x3040, 0x3080):
        p.progress(MemoryAccess(pc=pc, address=addr), prefetch_hit=False)

    p.init()

    assert p.last_block == {}
    assert p.pattern_table == {}
    assert p.region_map == {}
