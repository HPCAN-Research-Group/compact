<div align="center">
  <img src="images/banner.png" alt="COMPACT: Comprehensive Prefetching Algorithm Framework" width="100%">
</div>

---

# COMPACT: Comprehensive Prefetching Algorithm Framework

A modular Python simulation framework for developing and evaluating hardware prefetching algorithms. Use it to test new prefetching ideas on your laptop instead of in hardware simulation.

---

## Why You'd Use This

Hardware prefetching bridges the gap between processor speed and memory latency. But building and testing new prefetching strategies usually means:

- Running expensive hardware simulations  
- Using full system emulators or building prototypes
- Waiting months before you can evaluate your idea

COMPACT changes that. You run fast, software-based simulations on your laptop. You write new algorithms in Python. You iterate and compare strategies in hours, not months. No custom hardware needed.

---

## What's Inside

- 27+ algorithms spanning simple stride-based to deep learning approaches
- Modular design so you add your own algorithms
- YAML configs for parameter tuning without code changes
- Support for memory traces, synthetic workloads, and programmatic access patterns
- Analysis that measures prefetch accuracy, timeliness, and efficiency
- Pure Python with NumPy as the only external dependency
- Clean abstract base classes for extending the framework

---

## Implemented Algorithms

COMPACT includes 27+ algorithms organized by approach.

### Stride and Stream Based

- Linear Prefetcher: Detects and prefetches sequential address streams
- Strided Prefetcher: Identifies fixed-stride patterns like array rows or columns
- Best-Offset Prefetcher: Learns the best prefetch offset per program counter

### Correlation and History Based

- Markov Predictor: Models cache misses as Markov chain states
- Correlation Prefetcher: Learns sequences of correlated cache misses
- Global History Buffer (GHB): Correlates address deltas using program counter
- TCP (Tag Correlation): Correlates cache line tags instead of full addresses
- Reference Prediction Table: Predicts addresses from past miss sequences

### Spatial Patterns

- Spatial Memory Streaming (SMS): Learns spatial patterns within memory regions
- Bingo Spatial Prefetcher: Uses bit-vector footprints for spatial tracking
- DSPatch: Maintains accuracy-biased and coverage-biased footprints

### Machine Learning

- Neural Prefetcher: Combines LSTM models with per-PC linear models
- Perceptron Based Prefetcher: Uses online-trained perceptron for filtering
- LearnCluster: Learns memory access clusters unsupervised
- LearnPrefetch: General-purpose ML-based prefetch prediction

### Advanced Methods

Temporal approaches:
- Temporal Instruction Fetch Streaming: Records and replays instruction sequences
- Temporal Streaming Prefetcher: Learns temporal miss correlations
- Triage: Adaptive temporal prefetching with dynamic metadata sizing

Pattern recognition:
- Domino: Chains first and second-order miss correlations
- Triangel: Combines pattern-based and neighbor-based prediction

Specialized techniques:
- Event-Triggered Programmable: Rule-based prefetching on runtime events
- Feedback-Directed: Adapts prefetch degree based on accuracy metrics
- B-Fetch: Uses branch prediction to guide data prefetching
- Graph Prefetching: Prefetches based on graph traversal semantics
- Translation-Triggered (TEMPO): Uses address translation as prefetch trigger
- Epoch-Based Correlation (EBCP): Learns correlations between quiescence periods
- Store-Ordered Streaming: Records producer streams for consumer prefetching
- Indirect Address (IMP): Learns index-to-target relationships
- Efficient Footprint Caching (F-TDC): Prefetches based on page-level footprints
- Hot Data Stream (HDS): Models bursty access patterns with state machines
- Instruction Pointer Classification (IPCP): Classifies different PC patterns
- Metadata Based: Uses per-PC metadata with confidence tracking
- SPPAM: Combines signature-based delta prediction with access-map extrapolation
- Forest: GPU UVM tree-based neighboring prefetcher with per-object, access-aware tree reconfiguration

---

## System Requirements

- Python 3.8 or higher
- Linux, macOS, or Windows
- Any modern processor (this is software simulation, no special hardware needed)

---

## Get Started

Clone and install the project.

### Clone the Repository

```bash
git clone https://github.com/soilorian/COMPACT.git
cd COMPACT
```

### Option A: Standard Install

```bash
pip install -e .
```

### Option B: With Development Tools

If you plan to work on the code or run tests:

```bash
pip install -e ".[dev]"
```

### Option C: Using SSH

```bash
git clone git@github.com:soilorian/COMPACT.git
cd COMPACT
pip install -e .
```

### Verify the Installation

```bash
# List all algorithms
python run.py --list-algorithms

# See configuration template
python run.py --template
```

---

## What You Need

Python 3.8 or later, on Linux, macOS, or Windows.

Dependencies:

| Package | Version | Used for |
|---------|---------|----------|
| numpy | ~2.2.6 | Numerical arrays and computation |
| pytest | ~8.3.5 | Running tests |
| setuptools | ~65.5.1 | Package installation |

The setup also includes pyyaml for reading configuration files.

---

## Run Your First Simulation

Try a built-in example.

```bash
# Linear prefetcher
python run.py --config compact/config/configs/sample_linear.yml

# Spatial Memory Streaming
python run.py --config compact/config/configs/sample_sms.yml

# Global History Buffer
python run.py --config compact/config/configs/sample_ghb.yml
```

### List All Available Algorithms

```bash
python run.py --list-algorithms
```

### See Debug Output

```bash
python run.py --config compact/config/configs/sample_bingo.yml --log-level DEBUG
```

### Save Results to a File

```bash
python run.py --config compact/config/configs/sample_linear.yml --log-file results.log
```

---

## Use It in Your Code

### Run from the Command Line

Test multiple algorithms on the same trace:

```bash
for algo in linear sms ghb bingo; do
  python run.py --config compact/config/configs/sample_${algo}.yml \
    --log-file results_${algo}.log
done
```

### Run from Python

Load a prefetcher and run simulation:

```python
from compact.config.config_loader import ConfigLoader
from compact.analyzer.analyzer import Analyzer

loader = ConfigLoader("compact/config/configs/sample_linear.yml")
config = loader.load()
algorithm = loader.get_algorithm_instance()

data_loader = MyDataLoader(trace)
analyzer = Analyzer(algorithm, data_loader)
analyzer.run()
```

### Write Your Own Data Loader

Read memory access traces from a file:

```python
from compact.dataloader.dataloader import DataLoader
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess

class MyTraceLoader(DataLoader):
    def __init__(self, filename: str):
        self.filename = filename
        self.data = []
    
    def load(self):
        with open(self.filename) as f:
            for line in f:
                addr, pc = parse_line(line)
                access = MemoryAccess(address=addr, pc=pc)
                self.data.append(access)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

---

## Configure Your Simulation

COMPACT uses YAML files. Here's the basic structure:

```yaml
algorithm: linear

algorithm_config:
  STREAM_BUFFER_SIZE: 32
  PREFETCH_DEGREE: 4

log_file: "output_log.txt"
log_level: INFO
output_format: both
```

Available options:

| Field | Values | What it does |
|-------|--------|-------------|
| algorithm | linear, sms, ghb, ... | Which prefetcher to use |
| algorithm_config | Key-value pairs | Algorithm-specific settings |
| log_level | DEBUG, INFO, WARNING, ERROR, CRITICAL | How verbose the output |
| log_file | Path | Where to save logs |
| output_format | console, file, both | Output destination |

Sample configurations in `compact/config/configs/`:

- `sample_linear.yml` - Linear stride-based
- `sample_sms.yml` - Spatial Memory Streaming
- `sample_ghb.yml` - Global History Buffer
- `sample_bingo.yml` - Bingo
- `sample_bfetch.yml` - Branch-directed
- `sample_learncluster.yml` - Machine learning

View a configuration template:

```bash
python run.py --template
```

---

## Project Structure

```
COMPACT/
├── compact/
│   ├── __init__.py
│   ├── config/
│   │   ├── config_loader.py
│   │   ├── config_template.yml
│   │   ├── RptConfig.py
│   │   ├── INTERFACE.md
│   │   └── configs/
│   │       ├── sample_linear.yml
│   │       ├── sample_sms.yml
│   │       ├── sample_ghb.yml
│   │       └── ...
│   │
│   ├── prefetchingalgorithm/
│   │   ├── prefetchingalgorithm.py
│   │   ├── memoryaccess.py
│   │   ├── access/
│   │   └── impl/
│   │       ├── linear.py
│   │       ├── sms.py
│   │       ├── ghb.py
│   │       ├── bingo.py
│   │       ├── neural.py
│   │       └── ... (25+ more)
│   │
│   ├── analyzer/
│   │   └── analyzer.py
│   │
│   ├── dataloader/
│   │   ├── dataloader.py
│   │   └── impl/
│   │       └── ArrayDataLoader.py
│   │
│   ├── cache/
│   │   ├── Cache.py
│   │   └── replacementpolicy/
│   │
│   └── util/
│       └── size.py
│
├── tests/
│   ├── prefetchingalgorithm/
│   │   ├── test_linear.py
│   │   ├── test_sms.py
│   │   ├── test_bingo.py
│   │   └── ... (25+ test files)
│   ├── analyzer/
│   ├── dataloader/
│   └── ...
│
├── run.py
├── setup.py
├── requirements.txt
├── README.md
└── paper.md
```

---

## How to Add a New Prefetching Algorithm

### Step 1: Create the Prefetcher Class

Create a new file in `compact/prefetchingalgorithm/impl/` (for example, `myprefetcher.py`):

```python
from compact.prefetchingalgorithm.prefetchingalgorithm import PrefetchAlgorithm
from compact.prefetchingalgorithm.memoryaccess import MemoryAccess
import logging

logger = logging.getLogger(__name__)

class MyPrefetcher(PrefetchAlgorithm):
    """Description of your prefetcher algorithm."""
    
    def __init__(self, prefetch_degree: int = 4, **kwargs):
        """
        Initialize your prefetcher with parameters.
        
        Args:
            prefetch_degree: Number of cache lines to prefetch ahead
        """
        self.prefetch_degree = prefetch_degree
        self.history = []
    
    def init(self):
        """Initialize state before simulation begins."""
        self.history = []
        logger.info("MyPrefetcher initialized")
    
    def progress(self, access: MemoryAccess, prefetch_hit: bool) -> list:
        """
        Process a single memory access and return predicted addresses.
        
        Args:
            access: Current memory access (contains address, PC, etc.)
            prefetch_hit: Whether this access was a prefetch hit
        
        Returns:
            List of addresses to prefetch
        """
        addr = access.address
        self.history.append(addr)
        
        predictions = []
        if len(self.history) >= 2:
            stride = self.history[-1] - self.history[-2]
            for i in range(1, self.prefetch_degree + 1):
                predictions.append(addr + stride * i)
        
        return predictions
    
    def close(self):
        """Clean up after simulation ends."""
        logger.info(f"MyPrefetcher: Processed {len(self.history)} accesses")
```

### Step 2: Register Your Prefetcher

Edit `compact/config/config_loader.py` and add your algorithm to `AVAILABLE_ALGORITHMS`:

```python
AVAILABLE_ALGORITHMS = {
    # ... existing algorithms ...
    'myprefetcher': 'compact.prefetchingalgorithm.impl.myprefetcher',
}
```

### Step 3: Create a Configuration File

Create `compact/config/configs/sample_myprefetcher.yml`:

```yaml
algorithm: myprefetcher

algorithm_config:
  PREFETCH_DEGREE: 4

log_file: "compact/config/logs/myprefetcher_run.log"
log_level: INFO
output_format: both
```

### Step 4: Test Your Implementation

```bash
# Run your prefetcher
python run.py --config compact/config/configs/sample_myprefetcher.yml

# Write a unit test in tests/prefetchingalgorithm/test_myprefetcher.py
pytest tests/prefetchingalgorithm/test_myprefetcher.py -v
```

---

## What You Get

Simulations produce logs with:

- Correct Predictions: How many prefetch requests hit on future accesses
- Incorrect Predictions: How many prefetches missed
- Prefetch Accuracy: Correct divided by total predictions
- Algorithm Statistics: Algorithm-specific metrics like hit/miss counts

Example log output:

```
INFO: Starting analysis on 1000 data...
DEBUG: Correct: 0x1008
DEBUG: Correct: 0x1010
DEBUG: Incorrect: 0x1018
...
INFO: Correct predictions: 782
INFO: Incorrect predictions: 218
```

Enable debug logging to trace every prediction:

```bash
python run.py --config your_config.yml --log-level DEBUG
```

---

## Test the Code

Run the test suite:

```bash
pytest
```

Test one component:

```bash
pytest tests/prefetchingalgorithm/ -v
```

Test one file:

```bash
pytest tests/prefetchingalgorithm/test_linear.py -v
```

With coverage report:

```bash
pytest --cov=compact tests/
```

---

## Contributing

You can help. Here's how.

### Report an Issue

1. Check existing issues first
2. Describe what happened:
   - Steps to reproduce
   - What you expected to see
   - What you actually saw
   - Your Python version and OS
   - Log output if relevant

### Add a New Algorithm

1. Follow the "How to Add a New Prefetching Algorithm" section above
2. Write a docstring that describes your algorithm
3. Add unit tests in `tests/prefetchingalgorithm/test_<algorithm>.py`
4. Update this README with your algorithm
5. Submit a pull request with a clear title and description

### Code Style

Follow PEP 8. Format your code with:

```bash
black compact/ tests/
isort compact/ tests/
```

Check for issues:

```bash
flake8 compact/ tests/
```

### Open a Pull Request

1. Fork the repository
2. Create a feature branch (example: `git checkout -b feature/my-algorithm`)
3. Commit with clear messages
4. Push to your fork
5. Open a pull request
6. Make sure tests pass
7. Address any feedback

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright 2025 Mahdi Alinejad and contributors

---

## Citation

If you use COMPACT in your research, cite it as follows.

BibTeX:

```bibtex
@software{COMPACT2025,
  title = {COMPACT: Comprehensive Prefetching Algorithm Framework},
  author = {Alinejad, Mahdi and Zoufan, Sahand and Gheibi-Fetrat, Atiyeh and Hessabi, Shaahin and Sarbazi-Azad, Hamid},
  year = {2025},
  month = {11},
  day = {3},
  url = {https://github.com/soilorian/COMPACT},
  institution = {Sharif University of Technology and IPM Institute For Research In Fundamental Sciences}
}
```

Plain text:

```
COMPACT: Comprehensive Prefetching Algorithm Framework. Alinejad, M., Zoufan, S., 
Gheibi-Fetrat, A., Hessabi, S., and Sarbazi-Azad, H. (2025). 
https://github.com/soilorian/COMPACT
```

---

## Getting Help

Documentation is in [INTERFACE.md](compact/config/INTERFACE.md).

Sample configurations live in `compact/config/configs/` and show how to configure each algorithm.

Search [GitHub Issues](https://github.com/soilorian/COMPACT/issues) for questions others might have already answered.

Discussions coming soon for the community.

---

## Acknowledgments

Decades of research went into prefetching. We thank all researchers who published these algorithms.

---

Last updated February 2026. Active development. Pull requests welcome.
