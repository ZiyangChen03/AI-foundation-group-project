"""Pygame front end for playing Battleship by hand.

    uv run battleship_4x4.py                     # 4 x 4
    uv run python -m battleship.ui --board classic   # 10 x 10
    uv run battleship_4x4.py --agent belief      # watch the POMDP agent play instead
"""

import argparse

import pygame

from .config import CLASSIC, MINI, label
from .env import BattleshipEnv, CellState, Result

BG = '#101d2b'
PANEL = '#192c3e'
TEXT = '#eaf2f8'
MUTED = '#a5bacd'
ACCENT = '#60d5bb'


class BoardApp:
    WIDTH, HEIGHT = 520, 740
    GRID_LEFT, GRID_TOP, GRID_SIZE = 78, 192, 384

    def __init__(self, config=MINI):
        pygame.display.init()
        pygame.font.init()
        self.config = config
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption(f'Battleship - {config.size} x {config.size}')
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont('Segoe UI', 34, bold=True)
        self.font = pygame.font.SysFont('Segoe UI', 19)
        self.small_font = pygame.font.SysFont('Segoe UI', 16)
        self.button_font = pygame.font.SysFont('Segoe UI', 20, bold=True)
        self.pitch = self.GRID_SIZE / config.size
        self.cell_size = round(self.pitch) - max(2, round(self.pitch * 0.08))
        self.marker_scale = self.cell_size / 88
        self.cells = [pygame.Rect(round(self.GRID_LEFT + col * self.pitch),
                                  round(self.GRID_TOP + row * self.pitch),
                                  self.cell_size, self.cell_size)
                      for row in range(config.size) for col in range(config.size)]
        self.new_game_button = pygame.Rect(40, 660, 440, 48)
        self.game = BattleshipEnv(config)
        self.observation = self.game.observation()
        self.message = 'Click a square to fire.'
        self.running = False

    def restart(self):
        self.observation = self.game.reset()
        self.message = 'Click a square to fire.'

    def fire(self, action):
        if action not in self.game.legal_actions():
            return
        self.observation, _, done, shot = self.game.step(action)
        feedback = {Result.MISS: 'Miss', Result.HIT: 'Hit!',
                    Result.SUNK: f'Ship sunk! Length: {shot.sunk_length}.'}
        self.message = f'{label(action, self.config)}: {feedback[shot.result]}'
        if done:
            self.message = f'All ships sunk! You took {self.observation.num_shots} shots.'

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_r:
                self.restart()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.new_game_button.collidepoint(event.pos):
                self.restart()
            else:
                for action, rect in enumerate(self.cells):
                    if rect.collidepoint(event.pos):
                        self.fire(action)
                        break

    def label(self, text, position, color=TEXT, font=None, center=False):
        surface = (font or self.font).render(text, True, color)
        rect = surface.get_rect(center=position) if center else surface.get_rect(topleft=position)
        self.screen.blit(surface, rect)

    def draw_marker(self, value, center, scale=1.0):
        x, y = center
        if value is CellState.MISS:
            pygame.draw.circle(self.screen, TEXT, center, max(2, round(5 * scale)))
        elif value is CellState.HIT:
            arm, width = round(12 * scale), max(2, round(4 * scale))
            pygame.draw.line(self.screen, TEXT, (x - arm, y - arm), (x + arm, y + arm), width)
            pygame.draw.line(self.screen, TEXT, (x + arm, y - arm), (x - arm, y + arm), width)
        elif value is CellState.SUNK:
            half = round(11 * scale)
            pygame.draw.rect(self.screen, TEXT, (x - half, y - half, 2 * half, 2 * half),
                             border_radius=3)

    def render(self):
        size, fleet = self.config.size, self.config.fleet
        observation = self.observation
        self.screen.fill(BG)
        self.label('BATTLESHIP', (40, 25), font=self.title_font)
        lengths = ', '.join(map(str, fleet))
        self.label(f'{size} x {size} / {len(fleet)} hidden ships / Lengths: {lengths}',
                   (40, 77), MUTED, self.small_font)
        remaining = len(observation.remaining_ship_lengths)
        self.label(f'Shots: {observation.num_shots}', (40, 122))
        self.label(f'Ships remaining: {remaining} / {len(fleet)}', (265, 122))
        offset = self.cell_size / 2
        for col in range(size):
            self.label(chr(65 + col), (self.GRID_LEFT + col * self.pitch + offset, 172),
                       MUTED, center=True)
        for row in range(size):
            self.label(str(row + 1), (55, self.GRID_TOP + row * self.pitch + offset),
                       MUTED, center=True)
        mouse = pygame.mouse.get_pos()
        legal = set(self.game.legal_actions())
        colors = {CellState.UNKNOWN: PANEL, CellState.MISS: '#30465b',
                  CellState.HIT: '#b35542', CellState.SUNK: '#237d70'}
        for action, rect in enumerate(self.cells):
            value = observation.board[action]
            color = '#31526e' if action in legal and rect.collidepoint(mouse) else colors[value]
            pygame.draw.rect(self.screen, color, rect, border_radius=max(3, round(9 * self.marker_scale)))
            self.draw_marker(value, rect.center, self.marker_scale)
        for value, text, x in ((CellState.MISS, 'Miss', 85), (CellState.HIT, 'Hit', 220),
                               (CellState.SUNK, 'Sunk', 345)):
            self.draw_marker(value, (x, 630))
            self.label(text, (x + 19, 618), MUTED, self.small_font)
        done = not observation.remaining_ship_lengths
        self.label(self.message, (260, 596), ACCENT if done else TEXT, self.font, center=True)
        hover = self.new_game_button.collidepoint(mouse)
        pygame.draw.rect(self.screen, '#8ee5d1' if hover else ACCENT,
                         self.new_game_button, border_radius=9)
        self.label('New Game  (R)', self.new_game_button.center, BG,
                   self.button_font, center=True)
        pygame.display.flip()

    def run(self):
        self.running = True
        try:
            while self.running:
                for event in pygame.event.get():
                    self.handle_event(event)
                if self.running:
                    self.render()
                    self.clock.tick(60)
        finally:
            pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Play Battleship by hand, or watch an agent '
                                                 'play with --agent (see battleship/watch.py).')
    parser.add_argument('--board', choices=('mini', 'classic'), default='mini')
    parser.add_argument('--agent', help='watch this agent instead, e.g. belief or optimal')
    args, rest = parser.parse_known_args(argv)
    if args.agent:
        from . import watch
        watch.main(['--agent', args.agent, '--board', args.board, *rest])
        return
    if rest:
        parser.error(f'unrecognized arguments: {" ".join(rest)}')
    BoardApp(MINI if args.board == 'mini' else CLASSIC).run()


if __name__ == '__main__':
    main()
