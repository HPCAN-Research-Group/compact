"""
Forest: Access-aware GPU UVM (Tree-based Neighboring) Prefetcher.

Based on: Mao Lin, Yuan Feng, Guilherme Cox, Hyeran Jeon.
"Forest: Access-aware GPU UVM Management". ISCA 2025.
https://doi.org/10.1145/3695053.3731047

Overview:
- NVIDIA GPUs mitigate UVM far-fault overhead with a Tree-based Neighboring
  Prefetcher (TBNp): each memory region is covered by a full binary tree of
  fixed-size leaves. A fault on any page inside a leaf migrates the whole
  leaf; once more than half of a node's descendant pages are resident, the
  remaining descendants are proactively prefetched too, and the check
  cascades up the tree.
- TBNp uses one fixed tree shape (2MB tree / 64KB leaf) for every object and
  every access pattern, which the Forest paper shows is frequently wrong:
  linear streams want large trees/leaves, dense-but-narrow accesses want
  small ones, sparse-but-wide accesses want small trees with tiny leaves.
- Forest classifies each memory object's access pattern at runtime from a
  short window of (access order, page index) samples -- Linear/Streaming
  (LS), Non-Linear High-Coverage High-Intensity (HCHI), Non-Linear
  High-Coverage Low-Intensity (HCLI), or Non-Linear Low-Coverage (LC, the
  TBNp default) -- and reconfigures that object's tree to the shape that
  paper's evaluation found best for the pattern.

This implementation keeps the two core ideas -- the TBNp cascading tree and
Forest's per-object heterogeneous reconfiguration -- and omits the
hardware/driver mechanics (access-counter registers, isolation/motion bits,
LRU-based eviction, SpecForest's compiler-assisted classification) that have
no counterpart in a trace-driven, address-in/address-out simulator.
"""

from typing import Dict, List, Optional

from compact.prefetchingalgorithm.impl._shared import MRB, align_to_block
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess
from compact.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm

_KB = 1024
_MB = 1024 * _KB

# Pattern labels.
LS, HCHI, HCLI, LC = "LS", "HCHI", "HCLI", "LC"


class TreeNode:
    """A node of a TBNp binary tree covering [start, end) bytes."""

    def __init__(self, start: int, end: int, page_size: int, children: Optional[List["TreeNode"]] = None):
        self.start = start
        self.end = end
        self.page_size = page_size
        self.children = children or []
        self.is_leaf = not self.children
        self.total_pages = (end - start) // page_size if self.is_leaf else sum(
            c.total_pages for c in self.children
        )
        # Only leaves track individually-migrated pages; a promoted node
        # (leaf or internal) has all of its pages resident.
        self.migrated_pages: set = set()
        self.promoted = False

    def migrated_count(self) -> int:
        if self.is_leaf:
            return len(self.migrated_pages)
        return sum(c.migrated_count() for c in self.children)

    def locate_leaf(self, address: int) -> List["TreeNode"]:
        """Return the root-to-leaf path of nodes covering address."""
        path = [self]
        node = self
        while not node.is_leaf:
            node = node.children[0] if address < node.children[1].start else node.children[1]
            path.append(node)
        return path

    def collect_remaining(self) -> List[int]:
        """Mark this node's whole subtree as migrated; return the newly
        migrated page addresses (i.e. those not already resident)."""
        if self.is_leaf:
            pages = range(self.start, self.end, self.page_size)
            remaining = [p for p in pages if p not in self.migrated_pages]
            self.migrated_pages = set(pages)
            self.promoted = True
            return remaining
        remaining = []
        for child in self.children:
            remaining.extend(child.collect_remaining())
        self.promoted = True
        return remaining


def build_tree(start: int, size: int, leaf_size: int, page_size: int) -> TreeNode:
    """Build a full binary TBNp tree of `size` bytes with `leaf_size` leaves."""
    if size <= leaf_size:
        return TreeNode(start, start + size, page_size)
    half = size // 2
    left = build_tree(start, half, leaf_size, page_size)
    right = build_tree(start + half, half, leaf_size, page_size)
    return TreeNode(start, start + size, page_size, children=[left, right])


class TreeShape:
    """(tree_size, leaf_size) pair a pattern is mapped to."""

    __slots__ = ("tree_size", "leaf_size")

    def __init__(self, tree_size: int, leaf_size: int):
        self.tree_size = tree_size
        self.leaf_size = leaf_size


class _ObjectProfile:
    """Per-object (fixed-size region) profiling and classification state."""

    def __init__(self, base: int, shape: TreeShape):
        self.base = base
        self.shape = shape
        self.pattern = LC
        self.classified = False
        self.samples: List[int] = []  # page index accessed at each order
        self.seen_pages: set = set()


class ForestPrefetcher(PrefetchAlgorithm):
    """Access-aware, heterogeneous tree-based neighboring prefetcher."""

    def __init__(
        self,
        page_size: int = 4 * _KB,
        object_size: int = 4 * _MB,
        default_tree_size: int = 2 * _MB,
        default_leaf_size: int = 64 * _KB,
        ls_tree_size: int = 4 * _MB,
        ls_leaf_size: int = 256 * _KB,
        hchi_tree_size: int = 512 * _KB,
        hchi_leaf_size: int = 64 * _KB,
        hcli_tree_size: int = 512 * _KB,
        hcli_leaf_size: int = 16 * _KB,
        profiling_window: int = 32,
        linearity_threshold: float = 0.8,
        coverage_threshold: float = 0.6,
        intensity_threshold: float = 0.4,
        max_prefetch_degree: int = 64,
        mrb_size: int = 256,
    ):
        """Initialize the Forest prefetcher.

        Args:
            page_size: Minimum migration/prefetch unit in bytes.
            object_size: Granularity used to group addresses into a "data
                object" for access-pattern classification.
            default_tree_size/default_leaf_size: LC / pre-classification
                (baseline TBNp) tree shape.
            ls_tree_size/ls_leaf_size: Tree shape for Linear/Streaming.
            hchi_tree_size/hchi_leaf_size: Tree shape for Non-Linear
                High-Coverage High-Intensity.
            hcli_tree_size/hcli_leaf_size: Tree shape for Non-Linear
                High-Coverage Low-Intensity.
            profiling_window: Accesses sampled per object before classifying.
            linearity_threshold: Minimum R^2 to classify as LS.
            coverage_threshold: Accessed-range-vs-object-size ratio (P) used
                to separate HCHI/HCLI from LC.
            intensity_threshold: Accessed-page-count-vs-object-size ratio
                (A) used to separate HCHI from HCLI.
            max_prefetch_degree: Cap on prefetch addresses issued per access.
            mrb_size: Suppression-buffer size to avoid re-issuing addresses.
        """
        self.page_size = page_size
        self.object_size = object_size
        self.shapes = {
            LC: TreeShape(default_tree_size, default_leaf_size),
            LS: TreeShape(ls_tree_size, ls_leaf_size),
            HCHI: TreeShape(hchi_tree_size, hchi_leaf_size),
            HCLI: TreeShape(hcli_tree_size, hcli_leaf_size),
        }
        self.profiling_window = profiling_window
        self.linearity_threshold = linearity_threshold
        self.coverage_threshold = coverage_threshold
        self.intensity_threshold = intensity_threshold
        self.max_prefetch_degree = max_prefetch_degree

        self.objects: Dict[int, _ObjectProfile] = {}
        self.trees: Dict[int, TreeNode] = {}
        self.mrb = MRB(size=mrb_size)

    def init(self) -> None:
        """Reset all learned state."""
        self.objects.clear()
        self.trees.clear()
        self.mrb.clear()

    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> List[int]:
        """Process a memory access and return addresses to prefetch."""
        address = access.address
        obj_base = align_to_block(address, self.object_size)
        obj = self.objects.get(obj_base)
        if obj is None:
            obj = _ObjectProfile(obj_base, self.shapes[LC])
            self.objects[obj_base] = obj

        if not obj.classified:
            self._profile(obj, address)

        tree = self._tree_for(obj, address)
        candidates = self._access_tree(tree, address)

        issued = []
        for addr in candidates:
            if len(issued) >= self.max_prefetch_degree:
                break
            if self.mrb.contains(addr):
                continue
            self.mrb.insert(addr)
            issued.append(addr)
        return issued

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Access Pattern Detector (APD)
    # ------------------------------------------------------------------

    def _profile(self, obj: _ObjectProfile, address: int) -> None:
        page_index = (address - obj.base) // self.page_size
        obj.samples.append(page_index)
        obj.seen_pages.add(page_index)
        if len(obj.samples) < self.profiling_window:
            return

        pattern = self._classify(obj)
        obj.pattern = pattern
        obj.shape = self.shapes[pattern]
        obj.classified = True

    def _classify(self, obj: _ObjectProfile) -> str:
        total_pages = self.object_size // self.page_size
        accessed = sorted(obj.seen_pages)
        span = accessed[-1] - accessed[0]
        intensity = len(accessed)

        if self._r_squared(obj.samples) > self.linearity_threshold:
            return LS
        if span >= total_pages * self.coverage_threshold:
            if intensity >= total_pages * self.intensity_threshold:
                return HCHI
            return HCLI
        return LC

    @staticmethod
    def _r_squared(ys: List[int]) -> float:
        """Coefficient of determination of `ys` (page index) regressed
        against its sample order (access time), i.e. how linear the stream
        of accessed pages is."""
        n = len(ys)
        xs = range(n)
        mean_x = (n - 1) / 2
        mean_y = sum(ys) / n

        var_x = sum((x - mean_x) ** 2 for x in xs)
        if var_x == 0:
            return 0.0
        cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        slope = cov_xy / var_x
        intercept = mean_y - slope * mean_x

        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        if ss_tot == 0:
            # Every sample landed on the same page: perfectly predictable.
            return 1.0
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        return 1 - ss_res / ss_tot

    # ------------------------------------------------------------------
    # Tree-based Neighboring Prefetcher (TBNp)
    # ------------------------------------------------------------------

    def _tree_for(self, obj: _ObjectProfile, address: int) -> TreeNode:
        shape = obj.shape
        tree_base = align_to_block(address, shape.tree_size)
        tree = self.trees.get(tree_base)
        if tree is None or tree.end - tree.start != shape.tree_size:
            tree = build_tree(tree_base, shape.tree_size, shape.leaf_size, self.page_size)
            self.trees[tree_base] = tree
        return tree

    def _access_tree(self, tree: TreeNode, address: int) -> List[int]:
        page = align_to_block(address, self.page_size)
        path = tree.locate_leaf(address)
        leaf = path[-1]

        if page in leaf.migrated_pages:
            return []

        # A fault anywhere in a leaf migrates the whole leaf at once, so a
        # leaf is either untouched or fully resident -- never partial. The
        # faulting page itself is a demand fetch, not a prediction, so mark
        # it resident before collecting the rest as prefetch candidates.
        leaf.migrated_pages.add(page)
        candidates: List[int] = leaf.collect_remaining()

        # Cascade the >50%-migrated check up through the ancestors.
        for node in reversed(path[:-1]):
            if node.promoted:
                continue
            if node.migrated_count() > node.total_pages * 0.5:
                candidates.extend(node.collect_remaining())

        return candidates
