"""Plays Theseus (scripts/agents/groba) against the course's baseline agents on 13x13.

    pip install numpy gymnasium
    python docs/figures/run_matches.py      # ~1 h on 9 cores; writes docs/figures/matches.json

Each pairing is played twice (Theseus first and second). The second player sees the
board transposed and negated, exactly as in scripts/tournament_utils.competition_match.
Move times are recorded but not enforced, so every game reaches a result; the README
reports how many moves exceeded the 30 s limit of the assignment.
"""
import json
import sys
import time
import types
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "hex_udesa")]
sys.modules.setdefault("pygame", types.ModuleType("pygame"))  # only needed for rendering

import numpy as np  # noqa: E402

from agents.groba.grobaAgent import GrobaAgent  # noqa: E402
from agents.test_agents.adjacent import AdjacentAgent  # noqa: E402
from agents.test_agents.anti import AntiAgent  # noqa: E402
from agents.test_agents.center import CenterAgent  # noqa: E402
from agents.test_agents.cross import CrossAgent  # noqa: E402
from agents.test_agents.first import FirstAgent  # noqa: E402
from agents.test_agents.gandalf import GandalfAgent  # noqa: E402
from agents.test_agents.half import HalfAgent  # noqa: E402
from agents.test_agents.randy import RandomAgent  # noqa: E402
from agents.test_agents.smart1 import SmartAgent1  # noqa: E402
from hex_udesa.envs.hex import HexEnv  # noqa: E402

SIZE = 13
OPPONENTS = {"Random": RandomAgent, "First": FirstAgent, "Adjacent": AdjacentAgent,
             "Center": CenterAgent, "Anti": AntiAgent, "Half": HalfAgent, "Cross": CrossAgent,
             "Gandalf": GandalfAgent, "SmartAgent1": SmartAgent1}


def play(job):
    opp_name, theseus_first, seed = job
    np.random.seed(seed)
    theseus, opp = GrobaAgent(), OPPONENTS[opp_name]()
    agents = (theseus, opp) if theseus_first else (opp, theseus)
    for a in agents:
        a.reset()
    env = HexEnv(board_size=SIZE)
    board, _ = env.reset()
    moves, times, n, reward = [], [], 0, 0
    while not reward and (board == 0).any():
        agent = agents[n % 2]
        view = board if n % 2 == 0 else board.T * -1
        t0 = time.perf_counter()
        a = int(agent.action(view))
        dt = time.perf_counter() - t0
        if n % 2 == 1:  # undo the transpose
            r, c = divmod(a, SIZE)
            a = c * SIZE + r
        if agent is theseus:
            times.append(dt)
        board, reward, *_ = env.step(a)
        moves.append(a)
        n += 1
    winner_is_first = n % 2 == 1
    return {"opponent": opp_name, "theseus_first": theseus_first, "seed": seed,
            "theseus_won": bool(winner_is_first == theseus_first), "moves": moves,
            "theseus_times": times}


if __name__ == "__main__":
    jobs = [(o, first, 1000 + i) for i, o in enumerate(OPPONENTS) for first in (True, False)]
    with Pool(9) as pool:
        games = []
        for g in pool.imap_unordered(play, jobs):
            games.append(g)
            print(f"{g['opponent']:12s} first={g['theseus_first']!s:5s} won={g['theseus_won']!s:5s} "
                  f"moves={len(g['moves'])} max_t={max(g['theseus_times']):.1f}s", flush=True)
    games.sort(key=lambda g: (list(OPPONENTS).index(g["opponent"]), not g["theseus_first"]))
    (Path(__file__).with_name("matches.json")).write_text(json.dumps(games, indent=1))
