"""Learn a frozen prior, train Q, and compare policies on the true habit."""
import argparse
import csv
import json
import random
from pathlib import Path

from battleship import MINI, play_solo
from battleship.agents import OpponentBeliefAgent, RandomAgent, BeliefQAgent
from battleship.agents.belief_q import _actions
from battleship.opponent_belief_q.agent import OpponentBeliefQAgent
from battleship.opponents import FeatureHabit, border_cells, expected_shots
from battleship.planning import Planner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=1_000_000)
    p.add_argument('--history-games', type=int, default=1000)
    p.add_argument('--every', type=int, default=100000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--strength', type=float, default=1.0, help='edge preference; 0 is uniform')
    p.add_argument('--output', default='results/opponent_belief_q')
    p.add_argument('--uniform-models', default='results/belief_q')
    args = p.parse_args()
    if args.episodes < 1 or args.history_games < 0 or args.every < 1:
        p.error('episodes/every must be positive; history-games nonnegative')
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        p.error('use a new or empty output directory')
    for seed in args.seeds:
        if not (Path(args.uniform_models) / f'seed_{seed}.json.gz').exists():
            p.error(f'missing uniform checkpoint for seed {seed}')
    output.mkdir(parents=True, exist_ok=True)
    habit = FeatureHabit('edges', border_cells, args.strength)
    truth = habit.distribution(MINI)
    results = []
    with (output / 'curve.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['seed', 'episode', 'shots_on_true_habit', 'states'])
        for seed in args.seeds:
            learner = OpponentBeliefAgent()
            learner.training = True
            rng = random.Random(f'history:{seed}')
            for i in range(args.history_games):
                play_solo(learner, MINI, layout=habit.sample(MINI, rng), agent_seed=i)
            learner.training = False
            weights = learner.prior_weights(MINI)
            agent = OpponentBeliefQAgent(weights)
            def checkpoint(episode):
                value = agent.exact_value(truth)
                writer.writerow([seed, episode, value, agent.num_states()])
                stream.flush()
                print(f'seed {seed}, episode {episode}: {value:.4f} shots', flush=True)
            checkpoint(0)
            agent.fast_train(args.episodes, random.Random(seed), callback=checkpoint, every=args.every)
            if args.episodes % args.every:
                checkpoint(args.episodes)
            agent.save(output / f'seed_{seed}.json.gz')
            uniform = BeliefQAgent.load(Path(args.uniform_models) / f'seed_{seed}.json.gz')
            plan = Planner(MINI, truth)
            uniform_value = plan.policy_value(lambda layouts, fired: uniform.best_actions(
                plan._key(layouts, fired), _actions(plan, layouts, fired)))
            results.append(dict(seed=seed, habit_q=agent.exact_value(truth),
                                uniform_q=uniform_value,
                                habit_greedy=expected_shots(learner, MINI, habit),
                                learned_prior=list(weights)))
    (output / 'report.json').write_text(json.dumps(
        dict(config=vars(args), results=results,
             note='Q values are exact expectations on true habit; greedy averages ties over 3 seeds. '
                  'All 264 layouts can occur in training and evaluation; not held-out-layout generalization.'),
        indent=2), encoding='utf-8')
    print(json.dumps([{k: v for k, v in r.items() if k != 'learned_prior'} for r in results], indent=2))


if __name__ == '__main__':
    main()
