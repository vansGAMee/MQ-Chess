"""
Plotting utilities for Loss Index visualization.
Supports both calibration file and standalone mode.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import DATA_PATH


def plot_single_point(nickname: str, loss_index: float):
    """Standalone graph: user's point on colored skill scale."""
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.linspace(0, 25, 100)
    ax.fill_between(x, 0, 1, where=(x < 2), color='#2ecc71', alpha=0.4, label='GM')
    ax.fill_between(x, 0, 1, where=(x >= 2) & (x < 4), color='#27ae60', alpha=0.4, label='IM')
    ax.fill_between(x, 0, 1, where=(x >= 4) & (x < 7), color='#f1c40f', alpha=0.4, label='CM')
    ax.fill_between(x, 0, 1, where=(x >= 7) & (x < 12), color='#e67e22', alpha=0.4, label='Club')
    ax.fill_between(x, 0, 1, where=(x >= 12) & (x < 20), color='#e74c3c', alpha=0.4, label='Beginner')
    ax.fill_between(x, 0, 1, where=(x >= 20), color='#c0392b', alpha=0.4, label='Needs work')

    ax.scatter([loss_index], [0.5], c='#2c3e50', s=300, edgecolors='white', linewidth=2, zorder=5)
    ax.annotate(f'{loss_index:.1f}%', (loss_index, 0.5),
                xytext=(0, 30), textcoords='offset points',
                ha='center', fontsize=14, fontweight='bold')

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Loss Index (%) – lower is better')
    ax.set_title(f'Your Accuracy: {nickname}')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('mq_real_position.png', dpi=150)
    plt.close()
    print("✅ Single‑point graph saved as mq_real_position.png")


def plot_calibration(user_nickname: str, user_loss_index: float) -> None:
    """
    Plot user position against calibration data (if exists).
    Falls back to single‑point graph if no data.
    """
    if not DATA_PATH.exists():
        plot_single_point(user_nickname, user_loss_index)
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    points = []
    for nick, info in data.items():
        if 'loss_index' in info:
            points.append((nick, info.get('elo', 0), info['loss_index']))
        elif 'pairs' in info:
            pairs = info['pairs']
            if pairs:
                deltaW = [p[0] for p in pairs]
                weights = [p[1] for p in pairs]
                li = np.average(deltaW, weights=weights) * 100.0
                points.append((nick, info.get('elo', 0), li))

    if not points:
        plot_single_point(user_nickname, user_loss_index)
        return

    # Add user if not present
    if not any(p[0] == user_nickname for p in points):
        user_elo = 1200   # fallback ELO for display
        points.append((user_nickname, user_elo, user_loss_index))

    points.sort(key=lambda x: x[1])   # sort by ELO

    nicks = [p[0] for p in points]
    elos = [p[1] for p in points]
    lis = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#3498db' if nick != user_nickname else '#e67e22' for nick in nicks]
    sizes = [100 if nick != user_nickname else 220 for nick in nicks]
    ax.scatter(lis, elos, c=colors, s=sizes, edgecolors='#2c3e50', linewidth=1.5, zorder=5, alpha=0.85)

    for i, nick in enumerate(nicks):
        offset = 25 if nick == user_nickname else 10
        ax.annotate(nick, (lis[i], elos[i]),
                    xytext=(0, offset), textcoords='offset points',
                    ha='center', fontsize=10,
                    fontweight='bold' if nick == user_nickname else 'normal',
                    color=colors[i])

    if len(lis) >= 2:
        z = np.polyfit(lis, elos, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(lis) * 0.9, max(lis) * 1.1, 10)
        ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.6, label='Trend')

    # Skill level markers
    levels = [(2, 'GM'), (4, 'IM'), (7, 'CM'), (12, 'Club'), (20, 'Beginner')]
    for x, label in levels:
        ax.axvline(x=x, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
        ax.text(x, ax.get_ylim()[1] * 0.95, label, ha='center', fontsize=8, color='gray')

    ax.set_xlabel('Loss Index (%) – lower is better')
    ax.set_ylabel('Chess.com Rapid Rating')
    ax.set_title('Your Position on the Absolute Accuracy Scale')
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig('mq_real_position.png', dpi=150)
    plt.close()
    print("✅ Calibration graph saved as mq_real_position.png")