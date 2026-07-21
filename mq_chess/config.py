"""Configuration constants for MQ-Chess."""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LIB_PATH = PROJECT_ROOT / "lib" / "libanalyzer.so"
DATA_PATH = PROJECT_ROOT / "data" / "calibration_data.json"

# Stockfish
STOCKFISH_PATH = "stockfish"          # or full path

# Analysis settings
NUM_RECENT_GAMES = 30
MAX_CONCURRENT_GAMES = 2
ANALYSIS_NODES = 250_000

# Player (set in scripts)
DEFAULT_NICKNAME = "IvanKulkin"