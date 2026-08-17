from compact.prefetchingalgorithm.impl.forest import (
    HCHI,
    HCLI,
    LC,
    LS,
    ForestPrefetcher,
)
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess


def test_forest_leaf_migration_on_fault():
    # Default (LC) shape: 2MB tree / 64KB leaf / 4KB page -> 16 pages/leaf.
    p = ForestPrefetcher(profiling_window=10_000)
    p.init()

    preds = p.progress(MemoryAccess(pc=0x10, address=0), prefetch_hit=False)

    assert 0 not in preds
    assert sorted(preds) == [i * 4096 for i in range(1, 16)]


def test_forest_cascades_promotion_to_untouched_sibling():
    # Small tree so a 3-of-4-leaf majority is reachable in a few accesses:
    # 128KB tree / 16KB leaf / 4KB page -> 8 leaves of 4 pages each.
    p = ForestPrefetcher(
        profiling_window=10_000,
        default_tree_size=128 * 1024,
        default_leaf_size=16 * 1024,
    )
    p.init()

    # Fault the two leaves under the left half of the left subtree.
    p.progress(MemoryAccess(pc=0x10, address=0), prefetch_hit=False)
    p.progress(MemoryAccess(pc=0x10, address=16384), prefetch_hit=False)

    # Fault one of the two remaining leaves (leaf index 2); this pushes the
    # 4-leaf ancestor covering leaves 0-3 to 3/4 (>50%) migrated, so its
    # untouched 4th leaf (bytes [49152, 65536)) should be swept in too.
    preds = p.progress(MemoryAccess(pc=0x10, address=32768), prefetch_hit=False)

    leaf2_remaining = {36864, 40960, 45056}
    leaf3_swept = {49152, 53248, 57344, 61440}
    assert set(preds) == leaf2_remaining | leaf3_swept


def test_forest_does_not_reissue_already_migrated_pages():
    p = ForestPrefetcher(profiling_window=10_000)
    p.init()

    p.progress(MemoryAccess(pc=0x10, address=0), prefetch_hit=False)
    # Second access lands in the same, already fully-migrated leaf.
    preds = p.progress(MemoryAccess(pc=0x10, address=4096), prefetch_hit=False)

    assert preds == []


def test_forest_classifies_linear_stream_as_ls():
    p = ForestPrefetcher(profiling_window=32)
    p.init()

    preds = []
    for i in range(32):
        preds = p.progress(MemoryAccess(pc=0x20, address=i * 4096), prefetch_hit=False)

    obj = next(iter(p.objects.values()))
    assert obj.classified
    assert obj.pattern == LS
    assert obj.shape.tree_size == p.shapes[LS].tree_size
    assert obj.shape.leaf_size == p.shapes[LS].leaf_size


def test_forest_classifies_wide_dense_access_as_hchi():
    # 32-page object; a coprime stride visits all 32 pages once each,
    # covering the full span (high coverage) non-linearly (mod wraps).
    p = ForestPrefetcher(profiling_window=32, object_size=32 * 4096)
    p.init()

    for i in range(32):
        p.progress(MemoryAccess(pc=0x30, address=((i * 7) % 32) * 4096), prefetch_hit=False)

    obj = next(iter(p.objects.values()))
    assert obj.pattern == HCHI


def test_forest_classifies_wide_sparse_access_as_hcli():
    # Alternates between two far-apart pages: wide span, low intensity.
    p = ForestPrefetcher(profiling_window=32, object_size=32 * 4096)
    p.init()

    for i in range(32):
        page = 0 if i % 2 == 0 else 31
        p.progress(MemoryAccess(pc=0x40, address=page * 4096), prefetch_hit=False)

    obj = next(iter(p.objects.values()))
    assert obj.pattern == HCLI


def test_forest_classifies_clustered_access_as_lc():
    p = ForestPrefetcher(profiling_window=32, object_size=32 * 4096)
    p.init()

    for i in range(32):
        p.progress(MemoryAccess(pc=0x50, address=(i % 4) * 4096), prefetch_hit=False)

    obj = next(iter(p.objects.values()))
    assert obj.pattern == LC


def test_forest_init_resets_state():
    p = ForestPrefetcher(profiling_window=10_000)
    p.init()

    p.progress(MemoryAccess(pc=0x10, address=0), prefetch_hit=False)
    p.init()

    assert p.objects == {}
    assert p.trees == {}
