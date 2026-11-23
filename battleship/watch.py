"""Watch an agent play, with its belief drawn on the board.

    uv run python -m battleship.watch                          # POMDP agent on 4 x 4
    uv run python -m battleship.watch --agent optimal          # exact optimal policy
    uv run python -m battleship.watch --agent density --no-reveal
    uv run python -m battleship.watch --board classic --seed 7
    uv run python -m battleship.watch --agent optimal --show q   # Q* on every square
    uv run python -m battleship.watch --agent infogain --show i  # information gain (bits)

Each untried square is shaded by the agent's P(ship at square), with the
percentage written on it (relative scores for probability density). The ring
marks where the agent will fire next.

With --show q (or the Q key), the POMDP agents on 4 x 4 show Q*(b, a) instead:
the expected number of shots still needed if the agent fires at that square now
and plays optimally afterwards. The optimal agent fires at the smallest. The
panel splits the chosen square's Q* into ship cells still unhit + chance of a
miss now + misses expected later, and names greedy's pick when it differs.

With --show i (or the I key) they show each square's expected information gain
instead: how many bits of uncertainty about the layout the shot removes on
average, log2(layouts now) - E[log2(layouts left)]. The infogain agent fires at
the largest, among squares that could still hold a ship. The panel also gives
that square's Q*, so the two rules can be compared.

It plays by itself (one shot every --delay ms), or with --step one shot at a
time: each Space, Right or click on the board fires the next shot, and new
games stay in step mode. --show q and --show i start in step mode (A plays on
by itself). Keys: Space or Right = fire one shot, A = pause / play, H =
show/hide the hidden ships, Q = Q* view, I = information-gain view (press again
for P(ship)), N = new game, Esc = quit.
"""

import argparse
import random
from math import log2

import pygame

from .agents import BeliefAgent, HuntTargetAgent, ProbabilityDensityAgent, RandomAgent
from .belief import Belief
from .config import CLASSIC, MINI, GameConfig, label
from .env import BattleshipEnv, Result
from .planning import planner
from .ui import ACCENT, BG, MUTED, PANEL, TEXT, BoardApp

AGENTS = {
    "belief": lambda: BeliefAgent(),
    "rollout": lambda: BeliefAgent(planner="rollout"),
    "optimal": lambda: BeliefAgent(planner="optimal"),
    "infogain": lambda: BeliefAgent(planner="infogain"),
    "density": ProbabilityDensityAgent,
    "hunt": lambda: HuntTargetAgent(parity=True),
    "random": RandomAgent,
}


def _mix(low: str, high: str, t: float) -> pygame.Color:
    return pygame.Color(low).lerp(pygame.Color(high), max(0.0, min(1.0, t)))


class WatchApp(BoardApp):
    WIDTH = 900

    def __init__(self, config: GameConfig, agent_name: str, seed: int, auto: bool = True,
                 show: str = "p", step_mode: bool = False):
        super().__init__(config)
        pygame.display.set_caption(f"Battleship - watching {agent_name} - "
                                   f"{config.size} x {config.size}")
        self.agent_name = agent_name
        self.agent = AGENTS[agent_name]()
        self.seed = seed
        self.step_mode = step_mode            # one shot per key press or click
        self.auto = auto and not step_mode
        self.show_ships = False
        self.next_auto = 0
        self.target = self.heat = self.kind = None
        self.show = show
        self.q = None                     # Q*(b, a) per square, in the Q* and info views
        self.info = None                  # information gain per square, in the info view
        self.cell_font = pygame.font.SysFont("Segoe UI", 15 if config.size <= 6 else 11)
        self.new_game(seed)

    # --- game flow -------------------------------------------------------------

    def new_game(self, seed: int) -> None:
        self.seed = seed
        self.observation = self.game.reset(seed=seed)
        self.agent.new_game(self.config, random.Random(f"agent:{seed}"))
        self.last = ("Space or click the board: next shot" if self.step_mode
                     else "Space: one shot   A: pause / play")
        if getattr(self.agent, "planner", "greedy") in ("optimal", "rollout") or self.show != "p":
            self.solving_notice()
        self.plan_next()

    def solving_notice(self) -> None:
        # The first optimal decision solves the belief-state MDP, which takes a
        # while; say so instead of showing a frozen window.
        self.screen.fill(BG)
        self.label("Solving the belief-state MDP...", (40, 300), font=self.title_font)
        self.label("about 10-30 seconds, only once", (40, 350), MUTED)
        pygame.display.flip()
        pygame.event.pump()

    def plan_next(self) -> None:
        """Ask the agent for its next shot and read its belief for display."""
        if self.game.done:
            self.target, self.heat, self.kind, self.q, self.info = None, None, None, None, None
            return
        self.target = self.agent.select_shot(self.observation)
        self.heat, self.kind = self.read_belief()
        self.q = self.read_q() if self.show in ("q", "i") else None
        self.info = self.read_info() if self.show == "i" else None

    def can_show_q(self) -> bool:
        """Q* needs the exact belief, which the POMDP agents keep on 4 x 4."""
        return isinstance(self.agent, BeliefAgent) and isinstance(self.agent.belief, Belief)

    def read_q(self):
        """Q*(b, a) for every untried square, from the planner the agent uses.
        It only reads the planner's values; the agent chooses as before."""
        if not self.can_show_q():
            return None
        plan = planner(self.config)
        layouts, fired = plan.state(self.agent.belief)
        q = {cell: plan._q(layouts, fired, cell, plan.optimal_value)
             for cell in self.observation.legal_actions}
        unhit = plan.total - (plan._ship_cells(layouts) & fired).bit_count()
        return {"q": q, "unhit": unhit, "best": min(q.values()), "worst": max(q.values())}

    def read_info(self):
        """Expected information gain per square, from the planner's own
        ``expected_entropy`` (what the infogain agent maximizes). Squares no
        remaining layout covers are left out, as the infogain planner does."""
        if not self.can_show_q():
            return None
        plan = planner(self.config)
        layouts, fired = plan.state(self.agent.belief)
        n = layouts.bit_count()
        candidates = plan.hit_counts(layouts, fired)
        bits = {cell: max(0.0, log2(n) - plan.expected_entropy(layouts, fired, cell))
                for cell in candidates}
        picks = plan.info_gain_shots(layouts, fired)
        return {"bits": bits, "n": n, "picks": picks,
                "best": max(bits.values()), "worst": min(bits.values())}

    def toggle_view(self, kind: str = "q") -> None:
        """Switch to the Q* view (kind "q") or the information-gain view ("i");
        pressing the same key again goes back to P(ship)."""
        if self.show == kind:
            self.show, self.q, self.info = "p", None, None
        elif self.can_show_q():
            self.show = kind
            if not self.game.done:
                self.solving_notice()
                self.q = self.read_q()
                self.info = self.read_info() if kind == "i" else None

    def read_belief(self):
        if isinstance(self.agent, BeliefAgent):
            return self.agent.belief.occupancy(), "P(ship)"
        if isinstance(self.agent, ProbabilityDensityAgent):
            scores = self.agent.density(self.observation)
            top = max(scores) or 1
            return [s / top for s in scores], "score / best"
        return None, None

    def step(self) -> None:
        if self.game.done:
            return
        before = self.observation
        result = self.game.step(self.target)
        self.agent.observe(before, self.target, result)
        self.observation = result.observation
        shot = result.shot
        words = {Result.MISS: "miss", Result.HIT: "hit",
                 Result.SUNK: f"sunk a {shot.sunk_length}-ship"}
        self.last = f"Shot {self.observation.num_shots}: {label(shot.cell, self.config)} {words[shot.result]}"
        if result.done:
            self.last = f"All ships sunk in {self.observation.num_shots} shots.  N: new game"
            self.auto = False
        self.plan_next()

    # --- events -----------------------------------------------------------------

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                self.step()
            elif event.key == pygame.K_a:
                self.auto = not self.auto and not self.game.done
            elif event.key == pygame.K_h:
                self.show_ships = not self.show_ships
            elif event.key == pygame.K_q:
                self.toggle_view("q")
            elif event.key == pygame.K_i:
                self.toggle_view("i")
            elif event.key in (pygame.K_n, pygame.K_r):
                self.new_game(self.seed + 1)
                self.auto = not self.step_mode
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.new_game_button.collidepoint(event.pos):
                self.new_game(self.seed + 1)
                self.auto = self.auto and not self.step_mode
            elif self.step_mode and any(rect.collidepoint(event.pos) for rect in self.cells):
                self.step()                          # a click on the board: next shot

    # --- drawing ----------------------------------------------------------------

    def render(self):
        size, obs = self.config.size, self.observation
        self.screen.fill(BG)
        self.label("WATCHING  " + self.agent_name.upper(), (40, 25), font=self.title_font)
        if self.step_mode:
            self.label("one shot at a time", (520, 40), ACCENT, self.font)
        reveal = "sunk ships revealed" if self.config.reveal_sunk_cells else "sunk ships hidden"
        self.label(f"{size} x {size} / ships {', '.join(map(str, self.config.fleet))} / "
                   f"{reveal} / board seed {self.seed}", (40, 77), MUTED, self.small_font)
        self.label(f"Shots: {obs.num_shots}", (40, 122))
        self.label(f"Ships left: {len(obs.remaining_ship_lengths)} / {len(self.config.fleet)}",
                   (265, 122))
        offset = self.cell_size / 2
        for col in range(size):
            self.label(chr(65 + col), (self.GRID_LEFT + col * self.pitch + offset, 172),
                       MUTED, center=True)
        for row in range(size):
            self.label(str(row + 1), (55, self.GRID_TOP + row * self.pitch + offset),
                       MUTED, center=True)
        colors = {"?": PANEL, "o": "#30465b", "x": "#b35542", "#": "#237d70"}
        radius = max(3, round(9 * self.marker_scale))
        hidden = {c for ship in self.game._layout for c in ship.cells} if self.show_ships else set()
        for cell, rect in enumerate(self.cells):
            state = obs.board[cell]
            color = colors[state.value]
            p = qv = bits = None
            if state.value == "?" and self.info is not None:
                bits = self.info["bits"].get(cell)
                if bits is not None:
                    spread = self.info["best"] - self.info["worst"]
                    t = 1.0 if spread == 0 else (bits - self.info["worst"]) / spread
                    color = _mix(PANEL, "#5fb0e0", t)      # brightest = most bits
            elif state.value == "?" and self.q is not None:
                qv = self.q["q"][cell]
                spread = self.q["worst"] - self.q["best"]
                t = 1.0 if spread == 0 else (self.q["worst"] - qv) / spread
                color = _mix(PANEL, "#e0a43a", t)          # brightest = lowest Q*
            elif state.value == "?" and self.heat is not None:
                p = self.heat[cell]
                color = _mix(PANEL, "#e0a43a", p)
            pygame.draw.rect(self.screen, color, rect, border_radius=radius)
            if cell in hidden and state.value == "?":
                pygame.draw.rect(self.screen, TEXT, rect, width=2, border_radius=radius)
            if bits is not None:
                self.label(f"{bits:.3f}", rect.center, BG if t > 0.55 else TEXT,
                           self.cell_font, center=True)
            elif qv is not None:
                self.label(f"{qv:.3f}", rect.center, BG if t > 0.55 else TEXT,
                           self.cell_font, center=True)
            elif p is not None and p > 0:
                self.label(f"{100 * p:.0f}", rect.center, BG if p > 0.55 else TEXT,
                           self.cell_font, center=True)
            self.draw_marker(state, rect.center, self.marker_scale)
            if cell == self.target:
                pygame.draw.rect(self.screen, ACCENT, rect.inflate(6, 6), width=3,
                                 border_radius=radius + 2)
        self.render_panel()
        self.label(self.last, (260, 596), ACCENT if self.game.done else TEXT, self.font,
                   center=True)
        pygame.draw.rect(self.screen, ACCENT, self.new_game_button, border_radius=9)
        self.label("New game  (N)", self.new_game_button.center, BG, self.button_font,
                   center=True)
        pygame.display.flip()

    def render_panel(self):
        x, y = 520, 192
        if self.info is not None:
            self.render_info_panel(x, y)
        elif self.q is not None:
            self.render_q_panel(x, y)
        else:
            self.render_p_panel(x, y)
        first = ("Space / Right / click board   next shot" if self.step_mode
                 else "Space / Right   fire one shot")
        keys = [first, f"A   pause / play ({'playing' if self.auto else 'paused'})",
                f"H   hidden ships ({'shown' if self.show_ships else 'hidden'})",
                "N   new game", "Esc   quit"]
        if self.can_show_q() or self.show != "p":
            view = {"p": "P(ship)", "q": "Q*", "i": "info gain"}[self.show]
            keys.insert(3, f"Q / I   Q* / info-gain view (now {view})")
        for i, text in enumerate(keys):
            self.label(text, (x, 482 + 22 * i), MUTED, self.small_font)

    def render_q_panel(self, x, y):
        q, unhit = self.q["q"], self.q["unhit"]
        self.label("Next shot   (Q* = expected shots left)", (x, y), MUTED, self.small_font)
        if self.target is not None:
            p_miss = 1 - self.heat[self.target]
            later = q[self.target] - unhit - p_miss
            self.label(f"{label(self.target, self.config)}   Q* {q[self.target]:.3f}", (x, y + 22), ACCENT)
            self.label(f"= {unhit} unhit + {p_miss:.3f} miss now + {later:.3f} later",
                       (x, y + 50), TEXT, self.small_font)
            best_p = max(self.heat[c] for c in q)
            greedy = [c for c in q if self.heat[c] >= best_p - 1e-12]
            optimal = [c for c in q if q[c] <= self.q["best"] + 1e-9]
            if self.target not in optimal:            # e.g. watching the greedy agent
                self.label(f"Optimal would fire {self.names(optimal)}: Q* {self.q['best']:.3f}",
                           (x, y + 74), TEXT, self.small_font)
            elif self.target not in greedy:
                g = sum(q[c] for c in greedy) / len(greedy)
                self.label(f"Greedy would fire {self.names(greedy)}: Q* {g:.3f}", (x, y + 74),
                           TEXT, self.small_font)
        ranked = sorted(q, key=lambda c: q[c])[:6]
        self.label("Lowest Q*", (x, y + 108), MUTED, self.small_font)
        for i, c in enumerate(ranked):
            self.label(f"{label(c, self.config):>4}   {q[c]:6.3f}   P(hit) {100 * self.heat[c]:3.0f}%",
                       (x, y + 130 + 22 * i), TEXT, self.small_font)

    def render_info_panel(self, x, y):
        bits, q, n = self.info["bits"], self.q["q"], self.info["n"]
        self.label("Next shot   (info gain = bits removed)", (x, y), MUTED, self.small_font)
        if self.target is not None:
            b = bits.get(self.target, 0.0)
            self.label(f"{label(self.target, self.config)}   {b:.3f} bits   Q* {q[self.target]:.3f}",
                       (x, y + 22), ACCENT)
            self.label(f"{n} layouts = {log2(n):.2f} bits of uncertainty now", (x, y + 50), TEXT,
                       self.small_font)
            optimal = [c for c in q if q[c] <= self.q["best"] + 1e-9]
            most = self.info["picks"]
            if self.target not in most:                # e.g. watching the optimal agent
                self.label(f"Info gain would fire {self.names(most)}: {self.info['best']:.3f} bits",
                           (x, y + 74), TEXT, self.small_font)
            elif self.target not in optimal:
                self.label(f"Optimal would fire {self.names(optimal)}: Q* {self.q['best']:.3f}",
                           (x, y + 74), TEXT, self.small_font)
            else:
                best_p = max(self.heat[c] for c in q)
                greedy = [c for c in q if self.heat[c] >= best_p - 1e-12]
                if self.target not in greedy:
                    self.label(f"Greedy would fire {self.names(greedy)}: P(hit) {100 * best_p:.0f}%",
                               (x, y + 74), TEXT, self.small_font)
        ranked = sorted(bits, key=lambda c: (-bits[c], q[c]))[:6]
        self.label("Most information", (x, y + 108), MUTED, self.small_font)
        for i, c in enumerate(ranked):
            self.label(f"{label(c, self.config):>4}  {bits[c]:.3f} bits  P(hit) {100 * self.heat[c]:3.0f}%"
                       f"  Q* {q[c]:.3f}", (x, y + 130 + 22 * i), TEXT, self.small_font)

    def names(self, cells, most: int = 4) -> str:
        shown = ", ".join(label(c, self.config) for c in sorted(cells)[:most])
        return shown + (", ..." if len(cells) > most else "")

    def render_p_panel(self, x, y):
        self.label("Next shot", (x, y), MUTED, self.small_font)
        if self.target is not None:
            text = label(self.target, self.config)
            if self.heat is not None:
                text += f"   ({self.kind} {100 * self.heat[self.target]:.0f}%)"
            self.label(text, (x, y + 22), ACCENT)
        if self.heat is not None:
            ranked = sorted((c for c in self.observation.legal_actions if self.heat[c] > 0),
                            key=lambda c: -self.heat[c])[:6]
            self.label(f"Highest {self.kind}", (x, y + 70), MUTED, self.small_font)
            for i, c in enumerate(ranked):
                self.label(f"{label(c, self.config):>4}   {100 * self.heat[c]:5.1f}%",
                           (x, y + 94 + 24 * i), TEXT, self.small_font)

    def run(self, delay_ms: int = 600):
        self.running = True
        try:
            while self.running:
                for event in pygame.event.get():
                    self.handle_event(event)
                now = pygame.time.get_ticks()
                if self.auto and now >= self.next_auto:
                    self.step()
                    self.next_auto = now + delay_ms
                if self.running:
                    self.render()
                    self.clock.tick(60)
        finally:
            pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Watch an agent play Battleship.")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="belief")
    parser.add_argument("--board", choices=("mini", "classic"), default="mini")
    parser.add_argument("--no-reveal", action="store_true",
                        help="sinking a ship does not reveal its cells")
    parser.add_argument("--seed", type=int, default=0, help="which hidden board to start on")
    parser.add_argument("--delay", type=int, default=700, help="auto-play ms per shot")
    parser.add_argument("--step", "--manual", dest="step", action="store_true",
                        help="one shot at a time: Space, Right or a click on the board fires "
                             "the next shot (the default with --show q or i)")
    parser.add_argument("--auto", action="store_true",
                        help="play by itself even with --show q or i")
    parser.add_argument("--show", choices=("p", "q", "i"), default="p",
                        help="p: P(ship) on each square; q: Q*, the expected shots left; "
                             "i: information gain in bits (POMDP agents on 4 x 4)")
    args = parser.parse_args(argv)
    base = MINI if args.board == "mini" else CLASSIC
    config = GameConfig(base.size, base.fleet, reveal_sunk_cells=not args.no_reveal)
    if args.agent in ("optimal", "rollout", "infogain") and args.board == "classic":
        parser.error(f"--agent {args.agent} needs the exact belief; use --board mini")
    pomdp = ("belief", "rollout", "optimal", "infogain")
    if args.show != "p" and (args.board == "classic" or args.agent not in pomdp):
        parser.error(f"--show {args.show} needs a POMDP agent ({', '.join(pomdp)}) on --board mini")
    if args.agent in ("optimal", "rollout") or args.show != "p":
        print("Solving the belief-state MDP for this board (about 10-30 s, once)...")
    step_mode = args.step or (args.show != "p" and not args.auto)
    WatchApp(config, args.agent, args.seed, show=args.show, step_mode=step_mode).run(args.delay)


if __name__ == "__main__":
    main()
