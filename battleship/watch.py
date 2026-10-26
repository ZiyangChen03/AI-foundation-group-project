"""Watch an agent play, with its belief drawn on the board.

    uv run python -m battleship.watch                          # POMDP agent on 4 x 4
    uv run python -m battleship.watch --agent optimal          # exact optimal policy
    uv run python -m battleship.watch --agent density --no-reveal
    uv run python -m battleship.watch --board classic --seed 7

Each untried square is shaded by the agent's P(ship at square), with the
percentage written on it (relative scores for probability density). The ring
marks where the agent will fire next.

It plays by itself (one shot every --delay ms; --manual starts paused).
Keys: Space or Right = fire one shot, A = pause / play, H = show/hide the
hidden ships, N = new game, Esc = quit.
"""

import argparse
import random

import pygame

from .agents import BeliefAgent, HuntTargetAgent, ProbabilityDensityAgent, RandomAgent
from .config import CLASSIC, MINI, GameConfig, label
from .env import BattleshipEnv, Result
from .ui import ACCENT, BG, MUTED, PANEL, TEXT, BoardApp

AGENTS = {
    "belief": lambda: BeliefAgent(),
    "rollout": lambda: BeliefAgent(planner="rollout"),
    "optimal": lambda: BeliefAgent(planner="optimal"),
    "density": ProbabilityDensityAgent,
    "hunt": lambda: HuntTargetAgent(parity=True),
    "random": RandomAgent,
}


def _mix(low: str, high: str, t: float) -> pygame.Color:
    return pygame.Color(low).lerp(pygame.Color(high), max(0.0, min(1.0, t)))


class WatchApp(BoardApp):
    WIDTH = 900

    def __init__(self, config: GameConfig, agent_name: str, seed: int, auto: bool = True):
        super().__init__(config)
        pygame.display.set_caption(f"Battleship - watching {agent_name} - "
                                   f"{config.size} x {config.size}")
        self.agent_name = agent_name
        self.agent = AGENTS[agent_name]()
        self.seed = seed
        self.auto = auto
        self.show_ships = False
        self.next_auto = 0
        self.target = self.heat = self.kind = None
        self.cell_font = pygame.font.SysFont("Segoe UI", 15 if config.size <= 6 else 11)
        self.new_game(seed)

    # --- game flow -------------------------------------------------------------

    def new_game(self, seed: int) -> None:
        self.seed = seed
        self.observation = self.game.reset(seed=seed)
        self.agent.new_game(self.config, random.Random(f"agent:{seed}"))
        self.last = "Space: one shot   A: pause / play"
        if getattr(self.agent, "planner", "greedy") != "greedy":
            # The first decision solves the belief-state MDP, which takes a
            # while; say so instead of showing a frozen window.
            self.screen.fill(BG)
            self.label("Solving the belief-state MDP...", (40, 300), font=self.title_font)
            self.label("about 10-30 seconds, only once", (40, 350), MUTED)
            pygame.display.flip()
            pygame.event.pump()
        self.plan_next()

    def plan_next(self) -> None:
        """Ask the agent for its next shot and read its belief for display."""
        if self.game.done:
            self.target, self.heat, self.kind = None, None, None
            return
        self.target = self.agent.select_shot(self.observation)
        self.heat, self.kind = self.read_belief()

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
            elif event.key in (pygame.K_n, pygame.K_r):
                self.new_game(self.seed + 1)
                self.auto = True
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.new_game_button.collidepoint(event.pos):
                self.new_game(self.seed + 1)

    # --- drawing ----------------------------------------------------------------

    def render(self):
        size, obs = self.config.size, self.observation
        self.screen.fill(BG)
        self.label("WATCHING  " + self.agent_name.upper(), (40, 25), font=self.title_font)
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
            p = None
            if state.value == "?" and self.heat is not None:
                p = self.heat[cell]
                color = _mix(PANEL, "#e0a43a", p)
            pygame.draw.rect(self.screen, color, rect, border_radius=radius)
            if cell in hidden and state.value == "?":
                pygame.draw.rect(self.screen, TEXT, rect, width=2, border_radius=radius)
            if p is not None and p > 0:
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
        keys = ["Space / Right   fire one shot", f"A   pause / play ({'playing' if self.auto else 'paused'})",
                f"H   hidden ships ({'shown' if self.show_ships else 'hidden'})",
                "N   new game", "Esc   quit"]
        for i, text in enumerate(keys):
            self.label(text, (x, 470 + 24 * i), MUTED, self.small_font)

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
    parser.add_argument("--manual", action="store_true",
                        help="start paused; press Space for each shot")
    args = parser.parse_args(argv)
    base = MINI if args.board == "mini" else CLASSIC
    config = GameConfig(base.size, base.fleet, reveal_sunk_cells=not args.no_reveal)
    if args.agent in ("optimal", "rollout") and args.board == "classic":
        parser.error(f"--agent {args.agent} needs the exact belief; use --board mini")
    if args.agent in ("optimal", "rollout"):
        print("Solving the belief-state MDP for this board (about 10-30 s, once)...")
    WatchApp(config, args.agent, args.seed, auto=not args.manual).run(args.delay)


if __name__ == "__main__":
    main()
