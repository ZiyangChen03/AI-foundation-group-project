import pytest

from battleship import (CLASSIC, MINI, all_layouts, evaluate, play_match, play_solo,
                        round_robin, simulate_records, train)
from battleship.agents import Agent, HuntTargetAgent, RandomAgent


class RecordingAgent(RandomAgent):
    name = "recording"

    def __init__(self):
        super().__init__()
        self.calls = []

    def new_game(self, config, rng):
        super().new_game(config, rng)
        self.calls.append(("new_game", self.training))

    def observe(self, obs, action, result):
        assert action not in result.observation.legal_actions
        assert action in obs.legal_actions
        self.calls.append(("observe", action))

    def end_game(self, record):
        self.calls.append(("end_game", record.num_shots))


@pytest.mark.parametrize("config", [MINI, CLASSIC])
@pytest.mark.parametrize("agent", [RandomAgent(), HuntTargetAgent(False), HuntTargetAgent(True)])
def test_agents_finish_games(agent, config):
    for seed in range(20):
        record = play_solo(agent, config, seed=seed)
        assert sum(config.fleet) <= record.num_shots <= config.num_cells
        assert len(record.move_times) == record.num_shots


def test_hooks_are_called_in_order():
    agent = RecordingAgent()
    record = play_solo(agent, MINI, seed=1)
    assert agent.calls[0] == ("new_game", False)
    assert [name for name, _ in agent.calls[1:-1]] == ["observe"] * record.num_shots
    assert agent.calls[-1] == ("end_game", record.num_shots)


def test_train_switches_training_on_and_back():
    agent = RecordingAgent()
    records = train(agent, MINI, episodes=3)
    assert len(records) == 3
    assert [call for call in agent.calls if call[0] == "new_game"] == [("new_game", True)] * 3
    assert agent.training is False


def test_evaluate_is_reproducible_and_paired():
    first = evaluate(RandomAgent(), MINI, seeds=range(50))
    second = evaluate(RandomAgent(), MINI, seeds=range(50))
    assert [r.shots for r in first.records] == [r.shots for r in second.records]
    hunter = evaluate(HuntTargetAgent(), MINI, seeds=range(50))
    assert [r.layout for r in first.records] == [r.layout for r in hunter.records]
    assert sum(first.histogram.values()) == first.games == 50
    assert first.best <= first.median <= first.worst


def test_evaluate_on_every_layout():
    layouts = list(all_layouts(MINI))
    report = evaluate(HuntTargetAgent(), MINI, layouts=layouts)
    assert report.games == len(layouts)
    assert [r.layout for r in report.records] == layouts


def test_hunt_target_beats_random_on_classic():
    random_mean = evaluate(RandomAgent(), CLASSIC, seeds=range(200)).mean
    hunt_mean = evaluate(HuntTargetAgent(), CLASSIC, seeds=range(200)).mean
    assert hunt_mean < random_mean - 15


def test_simulate_records():
    records = simulate_records(MINI, 5)
    assert len(records) == 5 and all(r.num_shots >= 5 for r in records)


def test_illegal_agent_shot_names_the_agent():
    class Repeater(Agent):
        name = "repeater"

        def select_shot(self, obs):
            return 0

    with pytest.raises(ValueError, match="repeater"):
        play_solo(Repeater(), MINI, seed=0)


def test_play_match():
    a, b = RandomAgent(), HuntTargetAgent()
    match = play_match(a, b, CLASSIC, first=1, seed=5)
    winner, loser = match.records[match.winner], match.records[1 - match.winner]
    assert sum(1 for shot in winner.shots if shot.result.name == "SUNK") == len(CLASSIC.fleet)
    # players alternate, so the shot counts differ by at most one
    assert winner.num_shots - loser.num_shots in ((0, 1) if match.winner == match.first else (1,))
    replay = play_match(a, b, CLASSIC, first=1, seed=5)
    assert [(r.layout, r.shots) for r in replay.records] == \
        [(r.layout, r.shots) for r in match.records]
    with pytest.raises(ValueError):
        play_match(a, a)


def test_round_robin():
    agents = [RandomAgent(), HuntTargetAgent(False), HuntTargetAgent(True)]
    report = round_robin(agents, MINI, games_per_pair=40, seed=1)
    rates = report.win_rate
    for i in range(3):
        assert rates[i][i] is None
        for j in range(3):
            if i != j:
                assert rates[i][j] + rates[j][i] == pytest.approx(1.0)
    assert report.total_games == 120
    assert "hunt-target" in str(report)
