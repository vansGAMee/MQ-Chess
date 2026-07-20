#!/usr/bin/env python3
"""
Сравнение Loss Index для нескольких игроков.
Выбери 5 представителей разных уровней, скрипт построит график.
"""

import asyncio, ctypes, re, io as std_io, json, sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import aiohttp, chess, chess.engine, chess.pgn
import matplotlib.pyplot as plt
import numpy as np
from tqdm.asyncio import tqdm

# --------------------- НАСТРОЙКИ ---------------------
STOCKFISH_PATH = "stockfish"
ANALYSIS_NODES = 250_000
MAX_CONCURRENT_GAMES = 2          # для каждого игрока
NUM_GAMES_PER_PLAYER = 30        # сколько последних партий скачивать
MIN_MOVE_TIME = 0.4

# ---------- СПИСОК ИГРОКОВ (ЗАМЕНИ НА СВОИХ) ----------
PLAYERS = [
    ("IvanKulkin", 800),
    ("ChessMaster1000", 1000),
    ("ChessMaster1400", 1400),
    ("ChessDestroyer102", 2452),
    ("Hikaru", 2700),          # если есть рапид-партии
]
# ------------------------------------------------------

LIB_PATH = Path(__file__).parent / "libanalyzer.so"
if not LIB_PATH.exists():
    raise FileNotFoundError("libanalyzer.so не найден. Скомпилируйте: g++ -shared -fPIC -o libanalyzer.so libanalyzer.cpp -O3")
lib = ctypes.CDLL(str(LIB_PATH))

class MoveInput(ctypes.Structure):
    _fields_ = [
        ("bestEval", ctypes.c_int),
        ("playedEval", ctypes.c_int),
        ("legalMoves", ctypes.c_int),
        ("secondBestEval", ctypes.c_int),
        ("timeSpent", ctypes.c_double),
        ("isMate", ctypes.c_bool),
    ]

lib.calculate_advanced_mq.argtypes = [ctypes.POINTER(MoveInput), ctypes.c_int]
lib.calculate_advanced_mq.restype = ctypes.c_double


def extract_time_spent_from_comments(node, prev_clock):
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


async def fetch_games(nickname: str, num: int) -> List[str]:
    url = f"https://api.chess.com/pub/player/{nickname}/games/archives"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
        archives = data.get("archives", [])
        if not archives:
            return []
        recent = archives[-3:]  # последние 3 месяца
        pgns = []
        for arch_url in recent:
            async with session.get(arch_url) as resp:
                resp.raise_for_status()
                month_data = await resp.json()
            for g in month_data.get("games", []):
                if g.get("pgn"):
                    pgns.append(g["pgn"])
        pgns.reverse()
        return pgns[:num]


def parse_pgn_for_analysis(pgn_text: str, nickname: str) -> Optional[Tuple[chess.pgn.Game, chess.Color]]:
    game = chess.pgn.read_game(std_io.StringIO(pgn_text))
    if game is None:
        return None
    headers = game.headers

    time_control = headers.get("TimeControl", "")
    if time_control:
        parts = time_control.split("+")
        if len(parts) >= 1:
            try:
                if int(parts[0]) < 600:   # только рапид и классика
                    return None
            except ValueError:
                pass
    else:
        event = headers.get("Event", "")
        if "Rapid" not in event and "Classical" not in event:
            return None

    white = headers.get("White", "")
    black = headers.get("Black", "")
    if nickname.lower() == white.lower():
        return game, chess.WHITE
    elif nickname.lower() == black.lower():
        return game, chess.BLACK
    return None


def safe_score_to_cp(score, player_color: chess.Color) -> Tuple[int, bool]:
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


async def analyze_game(game: chess.pgn.Game, player_color: chess.Color,
                       engine_path: str, sem: asyncio.Semaphore) -> List[MoveInput]:
    async with sem:
        board = game.board()
        moves = list(game.mainline_moves())
        if not moves:
            return []

        move_inputs = []
        prev_clock = None
        mainline_nodes = list(game.mainline())[1:]

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

                time_spent = None
                if idx < len(mainline_nodes):
                    node = mainline_nodes[idx]
                    time_spent, prev_clock = extract_time_spent_from_comments(node, prev_clock)

                info_before = await engine.analyse(
                    board, chess.engine.Limit(nodes=ANALYSIS_NODES), multipv=1)
                if not info_before:
                    break
                best_score = info_before[0]["score"]
                bestEval, _ = safe_score_to_cp(best_score, player_color)

                board_copy = board.copy()
                board_copy.push(move)
                try:
                    info_after = await engine.analyse(
                        board_copy, chess.engine.Limit(nodes=ANALYSIS_NODES))
                    if isinstance(info_after, list):
                        played_score = info_after[0]["score"]
                    else:
                        played_score = info_after["score"]
                except Exception:
                    board.push(move)
                    continue

                playedEval, isMate = safe_score_to_cp(played_score, player_color)

                mi = MoveInput()
                mi.bestEval = int(bestEval)
                mi.playedEval = int(playedEval)
                mi.legalMoves = legal_moves
                mi.secondBestEval = int(bestEval)
                mi.timeSpent = time_spent if time_spent is not None else -1.0
                mi.isMate = isMate
                move_inputs.append(mi)

                board.push(move)

        finally:
            await engine.quit()

    return move_inputs


async def get_loss_index_for_player(nickname: str) -> Optional[float]:
    """Возвращает Loss Index для игрока или None, если не хватило данных."""
    print(f"\n=== Анализ {nickname} ===")
    try:
        pgns = await fetch_games(nickname, NUM_GAMES_PER_PLAYER)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None

    if not pgns:
        print("Партии не найдены.")
        return None

    print(f"Загружено {len(pgns)} PGN. Ищем рапид (>=10 мин)...")
    games_to_analyze = []
    for pgn_text in pgns:
        res = parse_pgn_for_analysis(pgn_text, nickname)
        if res:
            games_to_analyze.append(res)

    if not games_to_analyze:
        print("❌ Нет подходящих партий (рапид/классика).")
        return None

    print(f"Найдено {len(games_to_analyze)} партий, анализируем...")
    sem = asyncio.Semaphore(MAX_CONCURRENT_GAMES)
    tasks = [analyze_game(game, color, STOCKFISH_PATH, sem) for game, color in games_to_analyze]

    results = []
    with tqdm(total=len(tasks), desc=f"Анализ {nickname}", unit="game") as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            pbar.update(1)

    all_moves = []
    for res in results:
        if isinstance(res, Exception):
            continue
        all_moves.extend(res)
        term = MoveInput()
        term.legalMoves = -1
        all_moves.append(term)

    real_moves = [m for m in all_moves if m.legalMoves != -1]
    print(f"Ходов игрока для расчёта: {len(real_moves)}")
    if not real_moves:
        return None

    arr = (MoveInput * len(all_moves))(*all_moves)
    loss_index = lib.calculate_advanced_mq(arr, len(all_moves))
    if loss_index < 0:
        return None
    return loss_index


async def main():
    print("=== Сравнение Loss Index для выбранных игроков ===\n")
    results = []
    for nick, elo in PLAYERS:
        li = await get_loss_index_for_player(nick)
        if li is not None:
            print(f"✅ {nick} (рейтинг {elo}): Loss Index = {li:.2f}%")
            results.append((nick, elo, li))
        else:
            print(f"⚠️ {nick}: недостаточно данных, пропускаем.")

    if len(results) < 2:
        print("\nСлишком мало игроков для графика.")
        return

    # Построение графика
    nicks = [r[0] for r in results]
    elos = [r[1] for r in results]
    lis = [r[2] for r in results]

    plt.figure(figsize=(12, 7))
    plt.scatter(lis, elos, c='#ff6600', s=120, edgecolors='#2c3e50', zorder=5)

    # Подписи
    for i, nick in enumerate(nicks):
        plt.annotate(nick, (lis[i], elos[i]),
                     textcoords="offset points", xytext=(0, 15),
                     ha='center', fontsize=10, fontweight='bold')

    # Линия тренда
    if len(lis) >= 2:
        z = np.polyfit(lis, elos, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(lis)*0.9, max(lis)*1.1, 10)
        plt.plot(x_line, p(x_line), '--', color='gray', alpha=0.7, label='Тренд')

    plt.xlabel('Loss Index (%) – чем меньше, тем лучше')
    plt.ylabel('Рейтинг Chess.com (рапид)')
    plt.title('Сравнение абсолютной точности игры')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # слева лучшие
    plt.legend()
    plt.tight_layout()
    plt.savefig('players_comparison.png', dpi=150)
    plt.close()
    print("\nГрафик сохранён в players_comparison.png")


if __name__ == "__main__":
    asyncio.run(main())