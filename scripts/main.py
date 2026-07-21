#!/usr/bin/env python3
"""
Single-player Loss Index calculation.
Usage: python scripts/main.py [nickname]
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm.asyncio import tqdm

from mq_chess.config import (
    STOCKFISH_PATH, NUM_RECENT_GAMES, MAX_CONCURRENT_GAMES,
    ANALYSIS_NODES, DATA_PATH, DEFAULT_NICKNAME
)
from mq_chess.api import fetch_games, parse_pgn_for_analysis
from mq_chess.common import analyze_game, calculate_loss_index, MoveInput
from mq_chess.analyzer import plot_calibration


async def main():
    nickname = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NICKNAME
    print(f"Loading last {NUM_RECENT_GAMES} games for {nickname}...")

    try:
        pgns = await fetch_games(nickname, NUM_RECENT_GAMES)
    except Exception as e:
        print(f"Error fetching games: {e}")
        return

    if not pgns:
        print("No games found.")
        return

    print(f"Loaded {len(pgns)} PGNs. Filtering Rapid (>=10 min)...")
    games_to_analyze = []
    for pgn_text in pgns:
        res = parse_pgn_for_analysis(pgn_text, nickname)
        if res:
            games_to_analyze.append(res)

    if not games_to_analyze:
        print("❌ No rapid/classical games found in the last 3 months.")
        return

    print(f"Found {len(games_to_analyze)} games. Starting analysis...")
    print("(This may take a few minutes, depending on your CPU)")

    sem = asyncio.Semaphore(MAX_CONCURRENT_GAMES)
    tasks = [
        analyze_game(game, color, STOCKFISH_PATH, sem, ANALYSIS_NODES)
        for game, color in games_to_analyze
    ]

    all_moves = []
    with tqdm(total=len(tasks), desc="Analyzing games", unit="game") as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if isinstance(res, Exception):
                print(f"\nError in analysis: {res}")
                continue
            all_moves.extend(res)
            sep = MoveInput()
            sep.legalMoves = -1
            all_moves.append(sep)
            pbar.update(1)

    real_moves = [m for m in all_moves if m.legalMoves != -1]
    print(f"\nPlayer moves for calculation: {len(real_moves)}")

    if not real_moves:
        print("No valid moves.")
        return

    loss_index = calculate_loss_index(all_moves)
    if loss_index is None:
        print("❌ Not enough data to calculate Loss Index.")
        return

    print(f"✅ Your Loss Index: {loss_index:.2f}% (the lower, the better)")

    # Build graph (uses calibration data if exists, otherwise fallback)
    plot_calibration(nickname, loss_index)


if __name__ == "__main__":
    asyncio.run(main())