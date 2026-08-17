"""
Configuration loader for Compact - handles YAML config files
and dynamically loads prefetching algorithms with specified parameters.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Type
import importlib


class ConfigLoader:
    """Loads and validates configuration files for prefetching algorithms."""

    # All available algorithms in Compact
    AVAILABLE_ALGORITHMS = {
        'linear': 'compact.prefetchingalgorithm.impl.linear',
        'strided': 'compact.prefetchingalgorithm.impl.strided',
        'bestoffset': 'compact.prefetchingalgorithm.impl.bestoffset',
        'ghb': 'compact.prefetchingalgorithm.impl.ghb',
        'correlation': 'compact.prefetchingalgorithm.impl.correlation',
        'markovpredictor': 'compact.prefetchingalgorithm.impl.markovpredictor',
        'referencepredictiontable': 'compact.prefetchingalgorithm.impl.referencepredictiontable',
        'sms': 'compact.prefetchingalgorithm.impl.sms',
        'bingo': 'compact.prefetchingalgorithm.impl.bingo',
        'triage': 'compact.prefetchingalgorithm.impl.triage',
        'neural': 'compact.prefetchingalgorithm.impl.neural',
        'learncluster': 'compact.prefetchingalgorithm.impl.learncluster',
        'learnprefetch': 'compact.prefetchingalgorithm.impl.learnprefetch',
        'perceptron': 'compact.prefetchingalgorithm.impl.perceptron',
        'bfetch': 'compact.prefetchingalgorithm.impl.bfetch',
        'domino': 'compact.prefetchingalgorithm.impl.domino',
        'ebcp': 'compact.prefetchingalgorithm.impl.ebcp',
        'eventtriggered': 'compact.prefetchingalgorithm.impl.eventtriggered',
        'feedbackdirected': 'compact.prefetchingalgorithm.impl.feedbackdirected',
        'f_tdc': 'compact.prefetchingalgorithm.impl.f_tdc_prefetcher',
        'graph': 'compact.prefetchingalgorithm.impl.graph',
        'hds': 'compact.prefetchingalgorithm.impl.hds',
        'ipcp': 'compact.prefetchingalgorithm.impl.ipcp',
        'metadata': 'compact.prefetchingalgorithm.impl.metadata',
        'tempo': 'compact.prefetchingalgorithm.impl.tempo',
        'temporalmemorystreaming': 'compact.prefetchingalgorithm.impl.temporalmemorystreaming',
        'tcp': 'compact.prefetchingalgorithm.impl.tcp',
        'triangel': 'compact.prefetchingalgorithm.impl.triangel',
        'indirectmemory': 'compact.prefetchingalgorithm.impl.indirectmemory',
        'storeorderedstreamer': 'compact.prefetchingalgorithm.impl.storeorderedstreamer',
        'dspatch': 'compact.prefetchingalgorithm.impl.dspatch',
        'sppam': 'compact.prefetchingalgorithm.impl.sppam',
        'forest': 'compact.prefetchingalgorithm.impl.forest',
    }

    def __init__(self, config_path: str):
        """Initialize loader with config file path."""
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def load(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.logger.info(f"Loading configuration from: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f) or {}

        self._validate_config()
        self.logger.info("Configuration loaded and validated successfully")
        return self.config

    def _validate_config(self) -> None:
        """Validate required configuration fields."""
        if 'algorithm' not in self.config:
            raise ValueError("Config must specify 'algorithm' field")

        algo_name = self.config['algorithm'].lower()
        if algo_name not in self.AVAILABLE_ALGORITHMS:
            available = ', '.join(sorted(self.AVAILABLE_ALGORITHMS.keys()))
            raise ValueError(
                f"Unknown algorithm: {algo_name}\n"
                f"Available algorithms: {available}"
            )

    def get_algorithm_instance(self):
        """Create and return an instance of the specified algorithm."""
        algo_name = self.config['algorithm'].lower()
        module_path = self.AVAILABLE_ALGORITHMS[algo_name]

        self.logger.info(f"Loading algorithm: {algo_name}")

        try:
            # Import the module dynamically
            module = importlib.import_module(module_path)

            # Get the algorithm class (usually named like "LinearPrefetcher")
            # Try common naming patterns
            class_names = [
                f'{algo_name.capitalize()}Prefetcher',
                f'{algo_name.upper()}',
                f'{algo_name}Prefetcher',
            ]

            algorithm_class = None
            for class_name in class_names:
                if hasattr(module, class_name):
                    algorithm_class = getattr(module, class_name)
                    break

            # If not found, get the first class that's not builtin
            if algorithm_class is None:
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and 'Prefetcher' in name:
                        algorithm_class = obj
                        break

            if algorithm_class is None:
                raise ImportError(f"Could not find Prefetcher class in {module_path}")

            # Extract algorithm-specific config (everything under 'algorithm_config')
            algo_config = self.config.get('algorithm_config', {})

            self.logger.debug(f"Algorithm config: {algo_config}")

            # Instantiate with config as kwargs if provided
            if algo_config:
                try:
                    # Try with **kwargs first (some algorithms accept config params)
                    instance = algorithm_class(**algo_config)
                except TypeError:
                    # If that fails, try without parameters
                    self.logger.debug(f"Algorithm {algo_name} does not accept parameters, using defaults")
                    instance = algorithm_class()
            else:
                instance = algorithm_class()

            self.logger.info(f"Successfully instantiated {algo_name} prefetcher")
            return instance

        except ImportError as e:
            raise ImportError(f"Failed to load algorithm {algo_name}: {e}")
        except Exception as e:
            raise Exception(f"Failed to instantiate algorithm {algo_name}: {e}")

    def get_log_file(self) -> Optional[str]:
        """Get log file path from config."""
        return self.config.get('log_file')

    def get_log_level(self) -> str:
        """Get log level from config."""
        return self.config.get('log_level', 'INFO').upper()

    def get_input_trace(self) -> Optional[str]:
        """Get input trace file path from config."""
        return self.config.get('input_trace')

    def get_output_format(self) -> str:
        """Get output format (console, file, or both)."""
        return self.config.get('output_format', 'console').lower()

    def list_algorithms(self) -> None:
        """Print all available algorithms."""
        print("\nAvailable Prefetching Algorithms in Compact:\n")
        for i, algo_name in enumerate(sorted(self.AVAILABLE_ALGORITHMS.keys()), 1):
            print(f"  {i:2d}. {algo_name}")
        print(f"\nTotal: {len(self.AVAILABLE_ALGORITHMS)} algorithms\n")


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging based on configuration."""
    log_level = config.get('log_level', 'INFO').upper()
    log_file = config.get('log_file')
    output_format = config.get('output_format', 'console').lower()

    # Get or create root logger for all compact modules
    logger = logging.getLogger('compact')
    logger.setLevel(getattr(logging, log_level))

    # Also setup the incorrectly-named analyzer logger
    analyzer_logger = logging.getLogger('compact.analysis')
    analyzer_logger.setLevel(getattr(logging, log_level))

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Clear existing handlers
    logger.handlers = []
    analyzer_logger.handlers = []

    # Add console handler if needed
    if output_format in ['console', 'both']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        analyzer_logger.addHandler(console_handler)

    # Add file handler if needed
    if output_format in ['file', 'both'] or log_file:
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            analyzer_logger.addHandler(file_handler)
