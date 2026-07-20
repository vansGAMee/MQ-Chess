#!/usr/bin/env python3
"""
MQ-Chess: Loss Index (абсолютная метрика качества игры).
С научными графиками на основе реальных данных калибровки.
"""

import asyncio, ctypes, re, io as std_io, json
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp, chess, chess.engine, chess.pgn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tqdm.asyncio import tqdm

# --------------------- КОНФИГУРАЦИЯ ---------------------
NICKNAME = "testg123"              # твой ник
STOCKFISH_PATH = "stockfish"
NUM_RECENT_GAMES = 30
MAX_CONCURRENT_GAMES = 2
ANALYSIS_NODES = 250_000
MIN_MOVE_TIME = 0.4
# ----------------------------------------------------------

LIB_PATH = Path(__file__).parent / "libanalyzer.so"
if not LIB_PATH.exists():
    raise FileNotFoundError(
        "libanalyzer.so не найден.\n"
        "Скомпилируйте: g++ -shared -fPIC -o libanalyzer.so libanalyzer.cpp -O3"
    )
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
        recent = archives[-3:]
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
                if int(parts[0]) < 600:
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


def compute_loss_index_from_pairs(pairs):
    """Вычисляет Loss Index (%) по списку (deltaW, weight)."""
    if not pairs:
        return None
    deltaW = np.array([p[0] for p in pairs])
    weight = np.array([p[1] for p in pairs])
    avg = np.average(deltaW, weights=weight)
    return avg * 100.0


def build_real_distribution_plot(user_loss_index, user_nickname):
    """Строит график на основе реальных данных из calibration_data.json."""
    json_path = Path(__file__).parent / "calibration_data.json"
    if not json_path.exists():
        print("calibration_data.json не найден – реальный график невозможен.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    points = []  # (nickname, elo, loss_index)
    for nick, info in data.items():
        pairs = info.get("pairs", [])
        li = compute_loss_index_from_pairs(pairs)
        if li is not None:
            points.append((nick, info["elo"], li))

    if not points:
        print("Нет данных для построения графика.")
        return

    # Сортируем по Эло для наглядности
    points.sort(key=lambda x: x[1])

    nicks = [p[0] for p in points]
    elos = [p[1] for p in points]
    lis = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Рисуем точки реальных игроков
    colors = ['#3498db' if nick != user_nickname else '#ff6600' for nick in nicks]
    sizes = [120 if nick != user_nickname else 220 for nick in nicks]
    ax.scatter(lis, elos, c=colors, s=sizes, edgecolors='#2c3e50', zorder=5)

    # Подписываем каждого
    for i, nick in enumerate(nicks):
        offset = 25 if nick == user_nickname else 10
        ax.annotate(nick, (lis[i], elos[i]),
                    textcoords="offset points", xytext=(0, offset),
                    ha='center', fontsize=10, fontweight='bold' if nick == user_nickname else 'normal',
                    color=colors[i])

    # Соединяем линией для тренда
    ax.plot(lis, elos, '--', color='gray', alpha=0.5)

    ax.set_xlabel('Loss Index (%) – чем меньше, тем лучше')
    ax.set_ylabel('Рейтинг Chess.com (рапид)')
    ax.set_title('Ваше место на шкале абсолютной точности')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # слева – лучшие (меньше потеря)
    plt.tight_layout()
    plt.savefig('mq_real_position.png', dpi=150)
    plt.close()
    print("График реального положения сохранён в mq_real_position.png")


async def main():
    print(f"Загрузка последних {NUM_RECENT_GAMES} партий игрока {NICKNAME}...")
    try:
        pgns = await fetch_games(NICKNAME, NUM_RECENT_GAMES)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return

    if not pgns:
        print("Партии не найдены.")
        return

    print(f"Загружено {len(pgns)} PGN. Ищем рапид (>=10 мин)...")
    games_to_analyze = []
    for pgn_text in pgns:
        res = parse_pgn_for_analysis(pgn_text, NICKNAME)
        if res:
            games_to_analyze.append(res)

    if not games_to_analyze:
        print("❌ Нет рапид-партий с контролем >= 10 минут за последние 3 месяца.")
        return

    print(f"Найдено {len(games_to_analyze)} партий. Старт анализа...")

    sem = asyncio.Semaphore(MAX_CONCURRENT_GAMES)
    tasks = [analyze_game(game, color, STOCKFISH_PATH, sem) for game, color in games_to_analyze]

    results = []
    with tqdm(total=len(tasks), desc="Анализ партий", unit="game") as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            pbar.update(1)

    all_moves = []
    for res in results:
        if isinstance(res, Exception):
            print(f"Ошибка анализа: {res}")
            continue
        all_moves.extend(res)
        term = MoveInput()
        term.legalMoves = -1
        all_moves.append(term)

    real_moves = [m for m in all_moves if m.legalMoves != -1]
    print(f"Ходов игрока для расчёта: {len(real_moves)}")

    if not real_moves:
        print("Нет ходов.")
        return

    arr = (MoveInput * len(all_moves))(*all_moves)
    loss_index = lib.calculate_advanced_mq(arr, len(all_moves))

    if loss_index < 0:
        print("❌ Недостаточно данных для расчёта Loss Index.")
        return

    print(f"✅ Ваш Loss Index: {loss_index:.2f}% (чем меньше, тем лучше)")

    # Строим графики: старый (распределение) и новый (реальные точки)
    # Старый график тоже можно оставить, но я заменяю его на реальный.
    build_real_distribution_plot(loss_index, NICKNAME)


if __name__ == "__main__":
    asyncio.run(main())