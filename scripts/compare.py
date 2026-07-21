#!/usr/bin/env python3
"""
Compare Loss Index for multiple players.
Saves results to data/calibration_data.json for later use.
"""

import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from tqdm.asyncio import tqdm

from mq_chess.config import (
    STOCKFISH_PATH, NUM_RECENT_GAMES, MAX_CONCURRENT_GAMES,
    ANALYSIS_NODES
)
from mq_chess.api import fetch_games, parse_pgn_for_analysis
from mq_chess.common import analyze_game, calculate_loss_index, MoveInput

# ---------- CONFIG ----------
PLAYERS = [
    ("BigSmoke_Tr", 200),
    ("IvanKulkin", 750),
    ("testg123", 830),
    ("mihaley2013", 1500),
    ("ChessDestroyer102", 2452),
]
# ----------------------------

async def get_loss_index_for_player(nickname: str):
    """Fetch and analyse games for one player."""
    print(f"\n=== Analysing {nickname} ===")
    try:
        pgns = await fetch_games(nickname, NUM_RECENT_GAMES)
    except Exception as e:
        print(f"Error: {e}")
        return None

    if not pgns:
        print("No games found.")
        return None

    print(f"Loaded {len(pgns)} PGNs. Filtering Rapid...")
    games = []
    for pgn in pgns:
        res = parse_pgn_for_analysis(pgn, nickname)
        if res:
            games.append(res)

    if not games:
        print("❌ No rapid/classical games found.")
        return None

    print(f"Found {len(games)} games. Analysing...")
    sem = asyncio.Semaphore(MAX_CONCURRENT_GAMES)
    tasks = [analyze_game(g, c, STOCKFISH_PATH, sem, ANALYSIS_NODES) for g, c in games]

    all_moves = []
    with tqdm(total=len(tasks), desc=f"Analysing {nickname}", unit="game") as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            if isinstance(res, Exception):
                continue
            all_moves.extend(res)
            sep = MoveInput()
            sep.legalMoves = -1
            all_moves.append(sep)
            pbar.update(1)

    real = [m for m in all_moves if m.legalMoves != -1]
    print(f"Player moves: {len(real)}")
    if not real:
        return None

    return calculate_loss_index(all_moves)


async def main():
    print("=== MQ-Chess Player Comparison ===\n")
    results = []
    for nick, elo in PLAYERS:
        li = await get_loss_index_for_player(nick)
        if li is not None:
            print(f"✅ {nick} (ELO {elo}): Loss Index = {li:.2f}%")
            results.append((nick, elo, li))
        else:
            print(f"⚠️ {nick}: skipped.")

    if len(results) < 2:
        print("\nNot enough players for graph.")
        return

    # ---------- Save calibration data ----------
    calib_path = Path(__file__).parent.parent / "data" / "calibration_data.json"
    calib_data = {nick: {"elo": elo, "loss_index": li} for nick, elo, li in results}
    with open(calib_path, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"\n✅ Calibration data saved to {calib_path}")

    # ---------- Plot comparison ----------
    nicks, elos, lis = zip(*results)

    plt.figure(figsize=(12, 7))
    plt.scatter(lis, elos, c='#e67e22', s=130, edgecolors='#2c3e50', linewidth=1.5, zorder=5)
    for i, nick in enumerate(nicks):
        plt.annotate(nick, (lis[i], elos[i]),
                     xytext=(0, 15), textcoords='offset points',
                     ha='center', fontsize=10, fontweight='bold')

    if len(lis) >= 2:
        z = np.polyfit(lis, elos, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(lis)*0.9, max(lis)*1.1, 10)
        plt.plot(x_line, p(x_line), '--', color='gray', alpha=0.6, label='Trend')

    plt.xlabel('Loss Index (%) – lower is better')
    plt.ylabel('Chess.com Rapid Rating')
    plt.title('Absolute Accuracy Comparison')
    plt.grid(True, alpha=0.25)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig('players_comparison.png', dpi=150)
    plt.close()
    print("✅ Comparison graph saved as players_comparison.png")


if __name__ == "__main__":
    asyncio.run(main())