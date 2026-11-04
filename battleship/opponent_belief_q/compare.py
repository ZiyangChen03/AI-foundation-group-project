"""Four opponents x three Q variants. Run from the project root; see docs."""
import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

from battleship import MINI, play_solo
from battleship.agents import BeliefQAgent, OpponentBeliefAgent, RandomAgent
from battleship.opponent_belief_q.raw_q_checkpoint import RawQCheckpointAgent
from battleship.opponent_belief_q.agent import OpponentBeliefQAgent
from battleship.opponents import standard_habits, expected_shots
from battleship.planning import Planner


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def opponents():
    habits = standard_habits(MINI)
    return dict(zip(('random', 'edges', 'horizontal', 'favorites'),
                    (habits[0], habits[1], habits[3], habits[5])))


def write_report(output, rows, config):
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ['# Q-learning against placement habits', '',
            f"Completed {len(rows)} of {len(config['seeds']) * len(config['opponents'])} requested seed/opponent runs. "
            'Partial rows average only completed seeds.', '',
            '| Opponent | Raw-board Q | Belief Q | Opponent Belief Q | Optimal |',
            '|---|---:|---:|---:|---:|']
    for name in opponents():
        group = [r for r in rows if r['opponent'] == name]
        if group:
            means = [statistics.mean(r[k] for r in group) for k in ('raw_q', 'belief_q', 'opponent_q', 'optimal')]
            text.append('| ' + name + ' | ' + ' | '.join(f'{x:.3f}' for x in means) + ' |')
    text += ['', f"Training/history seeds: {config['seeds']}. Habit history: {config['history_games']} games; "
             f"opponent Q: {config['episodes']} episodes per seed.", '',
             'All three policies are evaluated on every layout with nonzero true opponent probability, '
             f"weighted by that probability, with {config['tie_seeds']} common policy RNG seeds per layout. "
             'Layout integration is exact; random tie-breaking is sampled, not exact. Optimal is exact.', '',
             'Raw Q and Belief Q reuse existing checkpoints; their training budgets, checkpoint selection '
             'and data splits differ. This is a comparison of saved systems, not a controlled ablation. '
             'The original raw Q test split is not preserved here: results cover the full opponent distribution. '
             'All layouts can appear during training. These are not unseen-layout generalization scores.', '',
             'The opponent prior is learned from finished games and frozen before Q training. '
             'Favorites uses layout counts with smoothing 1; other opponents use placement counts with smoothing 3. '
             'The optimum knows the true distribution; learners do not. '
             'Each row averages model/history seeds. No significance claim is implied.', '',
             'See config.json, per-seed comparison.csv and checkpoint hashes for provenance.']
    (output / 'README.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-models', required=True, type=Path)
    p.add_argument('--belief-models', default='results/belief_q', type=Path)
    p.add_argument('--reuse-edges', type=Path, help='completed train_opponent_belief_q run directory')
    p.add_argument('--output', default='results/q_habits', type=Path)
    p.add_argument('--episodes', type=int, default=1_000_000)
    p.add_argument('--history-games', type=int, default=1000)
    p.add_argument('--every', type=int, default=100000)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--tie-seeds', type=int, default=10)
    p.add_argument('--opponents', choices=list(opponents()), nargs='+', default=list(opponents()))
    args = p.parse_args()
    if min(args.episodes, args.every, args.tie_seeds) < 1 or args.history_games < 0:
        p.error('invalid training/evaluation counts')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.opponents)) != len(args.opponents):
        p.error('duplicate seeds or opponents')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('output must be new or empty')
    sources = {}
    for kind, directory in [('raw', args.raw_models), ('belief', args.belief_models)]:
        sources[kind] = {}
        for seed in args.seeds:
            path = directory / f'seed_{seed}.json.gz'
            if not path.exists():
                p.error(f'missing {path}')
            sources[kind][seed] = dict(path=str(path.resolve()), sha256=digest(path))
    reused = None
    if args.reuse_edges and 'edges' in args.opponents:
        reused = json.loads((args.reuse_edges / 'report.json').read_text())
        c = reused['config']
        if c['episodes'] != args.episodes or c['history_games'] != args.history_games or c['strength'] != 1.0:
            p.error('reused edge run must match episodes/history and strength=1')
        for seed in args.seeds:
            path = args.reuse_edges / f'seed_{seed}.json.gz'
            if seed not in c['seeds'] or not path.exists():
                p.error(f'missing reused edge seed {seed}')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config['sources'] = sources
    for kind, directory in [('raw', args.raw_models), ('belief', args.belief_models)]:
        source_config = directory / 'config.json'
        if source_config.exists():
            config[f'{kind}_training_config'] = json.loads(source_config.read_text())
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    rows = []
    for name in args.opponents:
        habit = opponents()[name]
        truth = habit.distribution(MINI)
        print(f'{name}: calculating oracle optimum', flush=True)
        plan = Planner(MINI, truth)
        optimum = plan.optimal_value(plan.everything, 0)
        for seed in args.seeds:
            folder = args.output / name / f'seed_{seed}'
            folder.mkdir(parents=True)
            if name == 'edges' and reused:
                path = args.reuse_edges / f'seed_{seed}.json.gz'
                agent = OpponentBeliefQAgent.load(path)
                source = dict(path=str(path.resolve()), sha256=digest(path))
            else:
                learner = OpponentBeliefAgent(model='layouts' if name == 'favorites' else 'placements',
                                             smoothing=1 if name == 'favorites' else 3)
                learner.training = True
                rng = random.Random(f'history:{seed}')
                for i in range(args.history_games):
                    play_solo(learner, MINI, layout=habit.sample(MINI, rng), agent_seed=i)
                agent = OpponentBeliefQAgent(learner.prior_weights(MINI))
                with (folder / 'curve.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['episode', 'true_habit_shots', 'states'])
                    def checkpoint(episode):
                        value = expected_shots(agent, MINI, habit, range(args.tie_seeds))
                        writer.writerow([episode, value, agent.num_states()])
                        stream.flush()
                        print(f'{name} seed {seed} episode {episode}: {value:.4f}', flush=True)
                    checkpoint(0)
                    agent.fast_train(args.episodes, random.Random(seed), callback=checkpoint, every=args.every)
                    if args.episodes % args.every:
                        checkpoint(args.episodes)
                path = folder / 'model.json.gz'
                agent.save(path)
                source = dict(path=str(path.resolve()), sha256=digest(path))
            raw = RawQCheckpointAgent.load(args.raw_models / f'seed_{seed}.json.gz')
            raw_value = expected_shots(raw, MINI, habit, range(args.tie_seeds))
            del raw
            belief = BeliefQAgent.load(args.belief_models / f'seed_{seed}.json.gz')
            row = dict(opponent=name, seed=seed, raw_q=raw_value,
                       belief_q=expected_shots(belief, MINI, habit, range(args.tie_seeds)),
                       opponent_q=expected_shots(agent, MINI, habit, range(args.tie_seeds)), optimal=optimum)
            (folder / 'metadata.json').write_text(json.dumps(dict(source=source,
                prior=agent.prior_weights, scores=row), indent=2), encoding='utf-8')
            rows.append(row)
            write_report(args.output, rows, config)
            print(row, flush=True)


if __name__ == '__main__':
    main()
