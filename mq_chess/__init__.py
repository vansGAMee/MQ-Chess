"""MQ-Chess: Loss Index – absolute chess quality metric."""
from .config import STOCKFISH_PATH, NUM_RECENT_GAMES, MAX_CONCURRENT_GAMES, ANALYSIS_NODES
from .common import MoveInput, calculate_loss_index, analyze_game
from .api import fetch_games, parse_pgn_for_analysis