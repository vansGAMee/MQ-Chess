"""Chess.com API wrapper and PGN filtering."""

import io as std_io
from typing import List, Optional, Tuple

import aiohttp
import chess
import chess.pgn

from .config import NUM_RECENT_GAMES


async def fetch_games(nickname: str, num: int = NUM_RECENT_GAMES) -> List[str]:
    """Fetch last num PGNs from chess.com archives (recent 3 months)."""
    url = f"https://api.chess.com/pub/player/{nickname}/games/archives"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.json()
        archives = data.get("archives", [])
        if not archives:
            return []

        # Take the last 3 months
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
    """
    Parse a PGN, filter for time control >= 10 minutes,
    and return (game, color) if the player is found.
    """
    game = chess.pgn.read_game(std_io.StringIO(pgn_text))
    if game is None:
        return None

    headers = game.headers
    # Time filter: Rapid/Classical only (>= 600 seconds)
    time_control = headers.get("TimeControl", "")
    if time_control:
        parts = time_control.split("+")
        if len(parts) >= 1:
            try:
                if int(parts[0]) < 600:   # less than 10 min
                    return None
            except ValueError:
                pass
    else:
        # fallback: check event name
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