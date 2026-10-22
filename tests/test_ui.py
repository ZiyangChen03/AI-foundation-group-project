import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # no window needed

import pygame
import pytest

from battleship import MINI, layout_from_cells
from battleship.ui import BoardApp

LAYOUT = layout_from_cells(MINI, [[0, 1, 2], [8, 12]])


@pytest.fixture
def app():
    app = BoardApp(MINI)
    app.observation = app.game.reset(layout=LAYOUT)
    yield app
    pygame.quit()


def click(app, cell):
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                        pos=app.cells[cell].center))
    app.render()


def test_click_fires_once(app):
    click(app, 0)
    click(app, 0)
    assert app.observation.num_shots == 1
    assert app.message == "A1: Hit!"


def test_click_after_win_is_ignored(app):
    for cell in (0, 1, 2, 8, 12):
        click(app, cell)
    assert app.game.done
    message = app.message
    click(app, 15)   # an unknown cell, but the game is over
    assert app.observation.num_shots == 5
    assert app.message == message


def test_restart_after_win(app):
    for cell in (0, 1, 2, 8, 12):
        click(app, cell)
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    assert not app.game.done and app.observation.num_shots == 0
    click(app, 15)
    assert app.observation.num_shots == 1
