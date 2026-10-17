import random
import pygame

class Battleship:
    SIZE = 4
    FLEET = (3, 2)
    UNKNOWN = "?"
    MISS = "o"
    HIT = "x"
    SUNK = "#"

    def __init__(self, seed=None):
        self.reset(seed)

    def reset(self, seed=None):
        self._rng = random.Random(seed)
        self._ships = []
        occupied = set()
        for length in self.FLEET:
            placements = []
            for row in range(self.SIZE):
                for col in range(self.SIZE):
                    for dr, dc in ((0, 1), (1, 0)):
                        cells = {(row + dr * i) * self.SIZE + col + dc * i
                                 for i in range(length)}
                        if (row + dr * (length - 1) < self.SIZE
                                and col + dc * (length - 1) < self.SIZE
                                and not cells & occupied):
                            placements.append(cells)
            ship = self._rng.choice(placements)
            self._ships.append(ship)
            occupied.update(ship)
        self._shots = set()
        self._board = [self.UNKNOWN] * (self.SIZE ** 2)
        self.done = False
        return self.observation()

    def observation(self):
        remaining = []
        for ship in self._ships:
            is_sunk = ship <= self._shots
            if not is_sunk:
                ship_length = len(ship)
                remaining.append(ship_length)
        remaining.sort(reverse=True)
        board = tuple(self._board)
        remaining_ship_lengths = tuple(remaining)
        return {
            "board": board,
            "remaining_ship_lengths": remaining_ship_lengths,
        }

    def legal_actions(self):
        if self.done:
            return []
        actions = []
        for i, value in enumerate(self._board):
            if value == self.UNKNOWN:
                actions.append(i)
        return actions
    
    def shots(self):
        return len(self._shots)

    def step(self, action):
        if self.done:
            raise ValueError("Game already finished")
        if type(action) is not int or action not in range(self.SIZE ** 2):
            raise ValueError("Action must be an integer from 0 to 15")
        if action in self._shots:
            raise ValueError("This square has already been attacked")
        self._shots.add(action)
        ship = None
        for candidate in self._ships:
            if action in candidate:
                ship = candidate
                break
        result = "miss"
        sunk_length = None
        self._board[action] = self.MISS
        if ship is not None:
            result = "hit"
            self._board[action] = self.HIT
            if ship <= self._shots:
                result = "sunk"
                sunk_length = len(ship)
                for cell in ship:
                    self._board[cell] = self.SUNK
        self.done = all(ship <= self._shots for ship in self._ships)
        return self.observation(), -1, self.done, {
            "result": result, "sunk_length": sunk_length, "shots": self.shots(),
        }

BG = '#101d2b'
PANEL = '#192c3e'
TEXT = '#eaf2f8'
MUTED = '#a5bacd'
ACCENT = '#60d5bb'


class BoardApp:
    WIDTH, HEIGHT = 520, 740

    def __init__(self):
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Battleship - 4 x 4')
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.SysFont('Segoe UI', 34, bold=True)
        self.font = pygame.font.SysFont('Segoe UI', 19)
        self.small_font = pygame.font.SysFont('Segoe UI', 16)
        self.button_font = pygame.font.SysFont('Segoe UI', 20, bold=True)
        self.cells = [pygame.Rect(78 + col * 96, 192 + row * 96, 88, 88)
                      for row in range(4) for col in range(4)]
        self.new_game_button = pygame.Rect(40, 660, 440, 48)
        self.game = Battleship()
        self.message = 'Click a square to fire.'
        self.running = False

    def restart(self):
        self.game.reset()
        self.message = 'Click a square to fire.'

    def fire(self, action):
        if action not in self.game.legal_actions():
            return
        _, _, done, info = self.game.step(action)
        coordinate = f'{chr(65 + action % 4)}{action // 4 + 1}'
        feedback = {'miss': 'Miss', 'hit': 'Hit!',
                    'sunk': f"Ship sunk! Length: {info['sunk_length']}."}
        self.message = f"{coordinate}: {feedback[info['result']]}"
        if done:
            self.message = f'All ships sunk! You took {self.game.shots()} shots.'

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

    def draw_marker(self, value, center):
        x, y = center
        if value == self.game.MISS:
            pygame.draw.circle(self.screen, TEXT, center, 5)
        elif value == self.game.HIT:
            pygame.draw.line(self.screen, TEXT, (x - 12, y - 12), (x + 12, y + 12), 4)
            pygame.draw.line(self.screen, TEXT, (x + 12, y - 12), (x - 12, y + 12), 4)
        elif value == self.game.SUNK:
            pygame.draw.rect(self.screen, TEXT, (x - 11, y - 11, 22, 22), border_radius=3)

    def render(self):
        self.screen.fill(BG)
        self.label('BATTLESHIP', (40, 25), font=self.title_font)
        self.label('4 x 4 / Two hidden ships / Lengths: 3 and 2', (40, 77), MUTED,
                   self.small_font)
        observation = self.game.observation()
        remaining = len(observation['remaining_ship_lengths'])
        self.label(f'Shots: {self.game.shots()}', (40, 122))
        self.label(f'Ships remaining: {remaining} / 2', (265, 122))
        for col in range(4):
            self.label(chr(65 + col), (122 + col * 96, 172), MUTED, center=True)
        for row in range(4):
            self.label(str(row + 1), (55, 236 + row * 96), MUTED, center=True)
        mouse = pygame.mouse.get_pos()
        legal = set(self.game.legal_actions())
        colors = {'?': PANEL, 'o': '#30465b', 'x': '#b35542', '#': '#237d70'}
        for action, rect in enumerate(self.cells):
            value = observation['board'][action]
            color = '#31526e' if action in legal and rect.collidepoint(mouse) else colors[value]
            pygame.draw.rect(self.screen, color, rect, border_radius=9)
            self.draw_marker(value, rect.center)
        for value, text, x in (('o', 'Miss', 85), ('x', 'Hit', 220), ('#', 'Sunk', 345)):
            self.draw_marker(value, (x, 630))
            self.label(text, (x + 19, 618), MUTED, self.small_font)
        self.label(self.message, (260, 596), ACCENT if self.game.done else TEXT,
                   self.font, center=True)
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


if __name__ == '__main__':
    BoardApp().run()
