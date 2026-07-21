"""Shared utilities: C++ binding, move structure, loss calculation."""

import ctypes
import re
from pathlib import Path
from typing import List, Optional, Tuple, Any

import chess
import chess.engine
import numpy as np

from .config import LIB_PATH


# ---------- C++ binding ----------
class MoveInput(ctypes.Structure):
    _fields_ = [
        ("bestEval", ctypes.c_int),
        ("playedEval", ctypes.c_int),
        ("legalMoves", ctypes.c_int),
        ("timeSpent", ctypes.c_double),
        ("isMate", ctypes.c_bool),
    ]


if not LIB_PATH.exists():
    raise FileNotFoundError(
        f"Library not found at {LIB_PATH}. Please compile it:\n"
        "g++ -shared -fPIC -o lib/libanalyzer.so libanalyzer.cpp -O3"
    )

_lib = ctypes.CDLL(str(LIB_PATH))
_lib.calculate_advanced_mq.argtypes = [ctypes.POINTER(MoveInput), ctypes.c_int]
_lib.calculate_advanced_mq.restype = ctypes.c_double


def calculate_loss_index(moves: List[MoveInput]) -> Optional[float]:
    """
    Call the C++ library to compute the Loss Index.
    Returns percentage (0-100) or None on failure.
    """
    if not moves:
        return None
    arr = (MoveInput * len(moves))(*moves)
    result = _lib.calculate_advanced_mq(arr, len(moves))
    if result < 0:
        return None
    return result


# ---------- Helper functions ----------
def extract_time_spent_from_comments(node: chess.pgn.BaseGame, prev_clock: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """Extract spent time from node.comment [%clk ...]."""
    comment = node.comment
    if not comment:
        return None, prev_clock
    m = re.search(r'\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]', comment)
    if not m:
        return None, prev_clock
    h, min, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
    cur_clock = h * 3600 + min * 60 + sec
    if prev_clock is not None:
        spent = prev_clock - cur_clock
        if spent < 0:
            spent = 0.0
    else:
        spent = None
    return spent, cur_clock


def safe_score_to_cp(score, player_color: chess.Color) -> Tuple[int, bool]:
    """
    Convert engine score to centipawns from player's perspective.
    Returns (cp, is_mate).
    """
    if hasattr(score, 'white'):
        w_score = score.white()
    else:
        w_score = score

    if w_score.is_mate():
        mate_moves = w_score.mate()
        if player_color == chess.WHITE:
            cp = 10000 if mate_moves > 0 else -10000
        else:
            cp = -10000 if mate_moves > 0 else 10000
        return cp, True
    else:
        cp = w_score.score()
        if player_color == chess.BLACK:
            cp = -cp
        return cp, False


async def analyze_game(
    game: chess.pgn.Game,
    player_color: chess.Color,
    engine_path: str,
    sem: Any,
    nodes: int = 250_000
) -> List[MoveInput]:
    """
    Analyse one game and return a list of MoveInput structures for the player's moves.
    """
    async with sem:
        board = game.board()
        moves = list(game.mainline_moves())
        if not moves:
            return []

        move_inputs = []
        prev_clock = None
        mainline_nodes = list(game.mainline())[1:]   # nodes after moves

        transport, engine = await chess.engine.popen_uci(engine_path)
        try:
            for idx, move in enumerate(moves):
                if board.is_game_over():
                    break
                if board.turn != player_color:
                    board.push(move)
                    continue

                legal_moves = board.legal_moves.count()
                if legal_moves == 0:
                    break

                # Extract clock time if available
                time_spent = None
                if idx < len(mainline_nodes):
                    node = mainline_nodes[idx]
                    time_spent, prev_clock = extract_time_spent_from_comments(node, prev_clock)

                # Best move evaluation
                info_before = await engine.analyse(
                    board, chess.engine.Limit(nodes=nodes), multipv=1
                )
                if not info_before:
                    break
                best_score = info_before[0]["score"]
                best_eval, _ = safe_score_to_cp(best_score, player_color)

                # Played move evaluation
                board_copy = board.copy()
                board_copy.push(move)
                try:
                    info_after = await engine.analyse(
                        board_copy, chess.engine.Limit(nodes=nodes)
                    )
                    if isinstance(info_after, list):
                        played_score = info_after[0]["score"]
                    else:
                        played_score = info_after["score"]
                except Exception:
                    # If analysis fails, skip this move
                    board.push(move)
                    continue

                played_eval, is_mate = safe_score_to_cp(played_score, player_color)

                mi = MoveInput()
                mi.bestEval = int(best_eval)
                mi.playedEval = int(played_eval)
                mi.legalMoves = legal_moves
                mi.timeSpent = time_spent if time_spent is not None else -1.0
                mi.isMate = is_mate
                move_inputs.append(mi)

                board.push(move)

        finally:
            await engine.quit()

        return move_inputs