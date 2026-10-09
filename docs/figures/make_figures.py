"""Regenerates the figures and the numbers shown in the README.

    pip install numpy matplotlib
    python docs/figures/run_matches.py   # optional, ~1 h: replays the games into matches.json
    python docs/figures/make_figures.py

Figure 4 re-plots the provisional tournament table published in the course README
(15 April 2025, 17:00), copied below; Figures 2 and 3 use matches.json.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon

HERE = Path(__file__).resolve().parent
for f in Path("/usr/share/fonts/lm").glob("lm*10-*.otf"):  # Latin Modern, if installed
    font_manager.fontManager.addfont(str(f))
plt.style.use(HERE / "paper.mplstyle")
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]
INK, GRAY = "#1a1a1a", "#8c8c8c"
SIZE = 13
games = json.loads((HERE / "matches.json").read_text())


def save(fig, name):
    fig.savefig(HERE / name, metadata={"Date": None})
    plt.close(fig)


# ---- Figure 2: final position of Theseus (first, left-right) vs SmartAgent1
game = next(g for g in games if g["opponent"] == "SmartAgent1" and g["theseus_first"])
board, order = np.zeros((SIZE, SIZE), int), {}
for k, a in enumerate(game["moves"]):
    board[divmod(a, SIZE)] = 1 if k % 2 == 0 else -1
    order[divmod(a, SIZE)] = k + 1


def winning_chain(b):  # stones of player +1 connected to both the left and the right edge
    def flood(starts):
        seen, stack = set(starts), list(starts)
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)):
                q = (r + dr, c + dc)
                if 0 <= q[0] < SIZE and 0 <= q[1] < SIZE and b[q] == 1 and q not in seen:
                    seen.add(q)
                    stack.append(q)
        return seen
    return flood([(r, 0) for r in range(SIZE) if b[r, 0] == 1]) & \
        flood([(r, SIZE - 1) for r in range(SIZE) if b[r, SIZE - 1] == 1])


chain = winning_chain(board)
R = 1 / np.sqrt(3)
hexagon = np.array([(R * np.sin(a), R * np.cos(a)) for a in np.arange(6) * np.pi / 3])


def centre(r, c):
    return c + 0.5 * r, -r * np.sqrt(3) / 2


fig, ax = plt.subplots(figsize=(7.2, 3.6))
for r in range(-1, SIZE + 1):
    for c in range(-1, SIZE + 1):
        inside = 0 <= r < SIZE and 0 <= c < SIZE
        if not inside and (r in (-1, SIZE)) == (c in (-1, SIZE)):
            continue  # skip corners
        x, y = centre(r, c)
        if not inside:
            fc = "#f3d3cd" if r in (-1, SIZE) else "#cfdcec"
            ax.add_patch(Polygon(hexagon * 0.92 + (x, y), fc=fc, ec="none"))
            continue
        v = board[r, c]
        fc = {0: "#eeeeee", 1: C[0], -1: C[1]}[v]
        ax.add_patch(Polygon(hexagon * 0.96 + (x, y), fc=fc, ec="white", lw=0.6))
        if (r, c) in chain:
            ax.add_patch(Polygon(hexagon * 0.96 + (x, y), fc="none", ec=INK, lw=1.3))
        if v:
            ax.text(x, y, order[(r, c)], ha="center", va="center", fontsize=5.5, color="white")
ax.text(*np.add(centre(SIZE // 2, -1), (-0.75, 0)), "Theseus\n(blue)", ha="right", va="center", fontsize=8,
        color=C[0])
ax.text(*np.add(centre(-1, SIZE // 2), (0, 0.6)), "SmartAgent1 (red)", ha="center", va="bottom",
        fontsize=8, color=C[1])
ax.set_aspect("equal")
ax.axis("off")
ax.autoscale_view()
save(fig, "fig2-board.svg")

# ---- Figure 1: decision procedure of GrobaAgent.action()
fig, ax = plt.subplots(figsize=(7.2, 2.25))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.15)
ax.axis("off")
BOX = [  # (x, y, w, h, text)
    (0.0, 1.85, 1.85, 0.95, "board $b$\n(own stones $+1$,\nleft–right)"),
    (2.35, 1.85, 2.0, 0.95, "winning move?\none stone completes\nour chain"),
    (4.85, 1.85, 2.0, 0.95, "forced block?\none stone completes\nthe opponent's chain"),
    (7.35, 1.85, 2.65, 0.95, "iterative deepening\n$d = 1, \\dots, 7$, alpha–beta,\ntransposition table"),
    (7.35, 0.1, 2.65, 1.2, "evaluation at the leaves\n$e(b) = L_{\\mathrm{opp}}(b) - L_{\\mathrm{own}}(b)$\n"
                           "$L$: A* shortest path, cost\n0 own / 1 empty / $\\infty$ opponent"),
    (3.6, 0.1, 3.25, 1.2, "move ordering\nadjacent to own chain first,\ncentral cells, bonus if a stone\n"
                          "shortens our path by $>1$"),
    (0.0, 0.1, 3.1, 1.2, "time control (25 s budget)\nsearch stops at ~20 s; best move\nof the deepest "
                         "finished depth,\nelse most connected cell"),
]
for x, y, w, h, t in BOX:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=INK, lw=0.7))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.6, linespacing=1.15)


def arrow(p, q, text=None, color=INK):
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, mutation_scale=8))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.06, text, ha="center", va="bottom", fontsize=7,
                color=GRAY)


arrow((1.85, 2.32), (2.35, 2.32))
arrow((4.35, 2.32), (4.85, 2.32), "no")
arrow((6.85, 2.32), (7.35, 2.32), "no")
for x in (8.675, 5.225, 1.55):  # components used inside the search
    arrow((8.675, 1.85), (x, 1.3), color=GRAY if x != 8.675 else INK)
for x in (3.35, 5.85):
    ax.text(x, 2.86, "if yes: play it", ha="center", va="bottom", fontsize=7, color=GRAY)
save(fig, "fig1-agent.svg")

# ---- Figure 4: provisional tournament (course README, 15/4 17:00): mu, sigma, baseline flag
TABLE = [(44.20, 2.07, 0), (38.88, 1.72, 0), (37.27, 1.59, 0), (36.09, 1.72, 0), (34.96, 1.67, 0),
         (35.00, 1.71, 0), (34.02, 1.67, 0), (32.66, 1.70, 0), (30.20, 1.38, 0), (30.66, 1.62, 0),
         (30.33, 1.63, 0), (29.52, 1.64, 0), (28.53, 1.68, "SmartAgent1"), (27.76, 1.62, 0),
         (25.53, 1.67, 0), (25.47, 1.65, "Gandalf"), (24.54, 1.65, "Half"), (24.27, 1.64, 0),
         (23.56, 1.68, "First"), (21.26, 1.64, 0), (21.11, 1.67, "Anti"), (20.54, 1.64, "Adjacent"),
         (19.49, 1.61, 0), (19.06, 1.62, "Random"), (18.70, 1.61, 0), (18.48, 1.59, 0),
         (18.26, 1.60, 0), (15.25, 1.79, "Cross"), (15.25, 1.88, "Center")]
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for i, (mu, sd, base) in enumerate(TABLE):
    c = GRAY if base else C[0]
    ax.errorbar(i, mu, yerr=sd, fmt="o", ms=3.5, color=c, elinewidth=0.8, capsize=0)
    ax.plot(i, mu - 3 * sd, "_", color=c, ms=5, mew=0.8)
    if base:
        ax.text(i, mu + sd + 0.8, base, rotation=90, ha="center", va="bottom", fontsize=7, color="#4d4d4d")
ax.axhline(28.53, color=C[1], ls="--", lw=0.8)
ax.text(28.6, 28.9, "pass threshold:\nbest baseline (28.53)", ha="right", va="bottom", fontsize=7.5, color=C[1])
ax.errorbar([], [], yerr=[], fmt="o", ms=3.5, color=C[0], label="student agents")
ax.errorbar([], [], yerr=[], fmt="o", ms=3.5, color=GRAY, label="course baselines")
ax.plot([], [], "_", color=INK, ms=5, label="$\\mu - 3\\sigma$ (ranking key)")
ax.set_xticks(range(0, len(TABLE), 4), [str(i + 1) for i in range(0, len(TABLE), 4)])
ax.set_xlabel("rank in the provisional table")
ax.set_ylabel("TrueSkill rating $\\mu \\pm \\sigma$")
ax.set_ylim(8, 52)
ax.legend(loc="upper right")
ax.tick_params(top=False)
save(fig, "fig4-tournament.svg")

# ---- Figure 3: replayed games, move times and game length
times = np.concatenate([g["theseus_times"] for g in games])
opps = list(dict.fromkeys(g["opponent"] for g in games))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
ax1.hist(times, bins=np.arange(0, 31, 1), color=C[0], edgecolor="white", lw=0.4)
ax1.set_ylim(0, ax1.get_ylim()[1] * 1.3)
for lim, lab, ha in ((20, "20 s (parallel script) ", "right"), (30, "30 s (assignment) ", "right")):
    ax1.axvline(lim, color=C[1], ls="--", lw=0.8)
    ax1.text(lim - 0.3, ax1.get_ylim()[1] * (0.97 if lim == 30 else 0.88), lab, ha=ha, va="top",
             fontsize=7.5, color=C[1])
ax1.set_xlabel("time per move [s]")
ax1.set_ylabel("moves")
ax1.set_title(f"(a) Theseus move times ({len(times)} moves)")
for k, first in enumerate((True, False)):
    n = [sum(1 for i in range(len(g["moves"])) if (i % 2 == 0) == first)
         for o in opps for g in games if g["opponent"] == o and g["theseus_first"] == first]
    ax2.plot(n, np.arange(len(opps)) + (k - 0.5) * 0.25, "o" if first else "s", ms=4, color=C[k],
             label="Theseus first" if first else "Theseus second")
ax2.axvline(SIZE, color=GRAY, ls=":", lw=0.8)
ax2.text(15.4, 4.5, "dotted line: 13 stones,\nthe shortest possible chain", fontsize=7.5, color="#4d4d4d",
         va="center")
ax2.set_yticks(range(len(opps)), opps)
ax2.invert_yaxis()
ax2.set_xlabel("stones placed by Theseus until the win")
ax2.set_title("(b) Game length against each baseline")
ax2.legend(loc="upper right")
ax2.tick_params(right=False)
fig.tight_layout()
save(fig, "fig3-games.svg")

# ---- numbers for the README
print(f"games: {len(games)}, won: {sum(g['theseus_won'] for g in games)}")
print(f"move time: median {np.median(times):.1f} s, max {times.max():.2f} s, "
      f">20 s: {(times > 20).sum()}, >30 s: {(times > 30).sum()}, total moves {len(times)}")
for o in opps:
    row = [g for g in games if g["opponent"] == o]
    print(o, [(g["theseus_first"], g["theseus_won"], len(g["moves"]), len(g["theseus_times"])) for g in row])
