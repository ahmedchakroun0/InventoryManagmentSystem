"""
Combined grid search over economic parameters and PPO hyperparameters.
Writes results to CSV. Supports optional early stopping and saving models.

Usage (smoke test):
  python3 grid_search.py --timesteps 2000 --max-combinations 2 --eval-episodes 2 --out-csv grid_search_smoke.csv

Defaults include the economics grid you requested.
"""
import argparse
import itertools
import csv
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from InventoryEnv import InventoryEnv
from utils import evaluate_policy


def make_env_factory(hc, of, sp, reward_scale=1000.0):
    def make_env():
        def _init():
            env = InventoryEnv(
                num_products=3,
                order_up_to=True,
                discrete_actions=False,
                holding_cost_per_unit=hc,
                order_cost_fixed=of,
                order_cost_per_unit=1.0,
                stockout_penalty=sp,
                reward_scale=reward_scale
            )
            return Monitor(env)
        return _init
    return make_env


def list_product(dict_of_lists):
    keys = list(dict_of_lists.keys())
    vals = [dict_of_lists[k] for k in keys]
    for comb in itertools.product(*vals):
        yield dict(zip(keys, comb))


def run_cartesian_grid(args):
    """Run a full Cartesian grid search over all provided parameter lists (econ + hp).

    This function expects per-parameter lists passed in on the args object (e.g.
    args.holding_costs, args.order_fixed, args.stockout_penalties,
    args.learning_rate_list, args.n_steps_list, ...). 
    """
    # Ensure per-parameter hp lists are lists
    if isinstance(args.learning_rate_list, (float, int)):
        args.learning_rate_list = [float(args.learning_rate_list)]
    if isinstance(args.n_steps_list, (float, int)):
        args.n_steps_list = [int(args.n_steps_list)]
    if isinstance(args.n_epochs_list, (float, int)):
        args.n_epochs_list = [int(args.n_epochs_list)]
    if isinstance(args.vf_coef_list, (float, int)):
        args.vf_coef_list = [float(args.vf_coef_list)]
    if isinstance(args.ent_coef_list, (float, int)):
        args.ent_coef_list = [float(args.ent_coef_list)]

    holding_costs = args.holding_costs
    order_fixed = args.order_fixed
    stockout_penalties = args.stockout_penalties

    learning_rate_list = args.learning_rate_list
    n_steps_list = args.n_steps_list
    n_epochs_list = args.n_epochs_list
    vf_coef_list = args.vf_coef_list
    ent_coef_list = args.ent_coef_list
    batch_size_list = [args.batch_size]

    dict_of_lists = {
        'holding_cost': holding_costs,
        'order_fixed': order_fixed,
        'stockout_penalty': stockout_penalties,
        'learning_rate': learning_rate_list,
        'n_steps': n_steps_list,
        'n_epochs': n_epochs_list,
        'vf_coef': vf_coef_list,
        'ent_coef': ent_coef_list,
        'batch_size': batch_size_list
    }

    out_csv = args.out_csv.replace('.csv', '_cartesian.csv')
    param_fields = list(dict_of_lists.keys())
    fieldnames = param_fields + ['rl_avg_profit', 'rl_avg_stockouts', 'rl_avg_inventory', 'train_time_s']

    best_row = None
    best_score = None
    total = 0

    max_comb = getattr(args, 'max_combinations', None)
    if max_comb == 0:
        max_comb = None

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for comb in list_product(dict_of_lists):
            total += 1
            if max_comb is not None and total > max_comb:
                break

            hc = float(comb['holding_cost'])
            of = float(comb['order_fixed'])
            sp = float(comb['stockout_penalty'])

            lr = float(comb['learning_rate'])
            n_steps = int(comb['n_steps'])
            n_epochs = int(comb['n_epochs'])
            vf_coef = float(comb['vf_coef'])
            ent_coef = float(comb['ent_coef'])
            batch_size = int(comb.get('batch_size', args.batch_size))

            print(f"Testing combo {total}: hc={hc}, of={of}, sp={sp}, lr={lr}, n_steps={n_steps}, batch_size={batch_size}")
            start = time.time()

            make_env = make_env_factory(hc, of, sp, reward_scale=args.reward_scale)
            vec_env = DummyVecEnv([make_env() for _ in range(args.n_envs)])

            model = PPO(
                policy='MlpPolicy',
                env=vec_env,
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=args.gamma,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                verbose=0,
                device='cpu'
            )

            model.learn(total_timesteps=args.timesteps)
            train_time = time.time() - start

            eval_env = Monitor(InventoryEnv(
                num_products=3,
                order_up_to=True,
                discrete_actions=False,
                holding_cost_per_unit=hc,
                order_cost_fixed=of,
                order_cost_per_unit=1.0,
                stockout_penalty=sp,
                reward_scale=args.reward_scale
            ))

            rl_results = evaluate_policy(model, eval_env, num_episodes=args.eval_episodes, use_trained=True)

            row = {k: comb[k] for k in param_fields}
            row.update({
                'rl_avg_profit': float(np.mean(rl_results['rewards'])),
                'rl_avg_stockouts': float(np.mean(rl_results['stockouts'])),
                'rl_avg_inventory': float(np.mean(rl_results['inventory_levels'])),
                'train_time_s': float(train_time)
            })

            writer.writerow(row)
            f.flush()
            print(' Result:', row)

            score = row['rl_avg_profit']
            if best_score is None or (score, -row['rl_avg_stockouts']) > (best_score, -best_row.get('rl_avg_stockouts', 0)):
                best_score = score
                best_row = row.copy()

    import json
    best_cfg = {k: best_row[k] for k in param_fields}
    best_json = args.out_csv.replace('.csv', '_cartesian_best_config.json')
    with open(best_json, 'w') as f:
        json.dump(best_cfg, f, indent=2)

    print('\nCartesian grid complete. Results saved to', out_csv)
    print('Best combination saved to', best_json)
    return out_csv, best_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combined grid search: economics x PPO hyperparameters')
    parser.add_argument('--timesteps', type=int, default=20000)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--max-combinations', type=int, default=50)
    parser.add_argument('--out-csv', default='grid_search_results.csv')

    # Economics grid defaults (from your request)
    parser.add_argument('--holding-costs', type=lambda s: [float(x) for x in s.split(',')], default=[0.5,1.5,3.0])
    parser.add_argument('--order-fixed', type=lambda s: [float(x) for x in s.split(',')], default=[5.0,20.0,50.0])
    parser.add_argument('--stockout-penalties', type=lambda s: [float(x) for x in s.split(',')], default=[10.0,50.0,200.0])

    # HP grid defaults
    parser.add_argument('--learning-rate-list', type=lambda s: [float(x) for x in s.split(',')], default=[3e-5,1e-4,1e-5])
    parser.add_argument('--n-steps-list', type=lambda s: [int(x) for x in s.split(',')], default=[512,2048,4096])
    parser.add_argument('--n-epochs-list', type=lambda s: [int(x) for x in s.split(',')], default=[5,10])
    parser.add_argument('--vf-coef-list', type=lambda s: [float(x) for x in s.split(',')], default=[0.25,0.5])
    parser.add_argument('--ent-coef-list', type=lambda s: [float(x) for x in s.split(',')], default=[0.0,0.01])

    # PPO options
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.995)

    # Early stopping options
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--eval-freq', type=int, default=10000)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min-delta', type=float, default=1e-3)

    # Misc
    parser.add_argument('--reward-scale', type=float, default=1000.0)
    parser.add_argument('--save-models', dest='save_models', action='store_true', help='Save best checkpoints during early-stop')

    args = parser.parse_args()

    # Ensure lists when argparse default lambda isn't applied
    if isinstance(args.holding_costs, float) or isinstance(args.holding_costs, int):
        args.holding_costs = [args.holding_costs]
    if isinstance(args.order_fixed, float) or isinstance(args.order_fixed, int):
        args.order_fixed = [args.order_fixed]
    if isinstance(args.stockout_penalties, float) or isinstance(args.stockout_penalties, int):
        args.stockout_penalties = [args.stockout_penalties]

    # Ensure hp lists
    if isinstance(args.learning_rate_list, float) or isinstance(args.learning_rate_list, int):
        args.learning_rate_list = [args.learning_rate_list]
    if isinstance(args.n_steps_list, int):
        args.n_steps_list = [args.n_steps_list]

    # Build args aliases expected later
    args.holding_costs = args.holding_costs
    args.order_fixed = args.order_fixed
    args.stockout_penalties = args.stockout_penalties

    # Run full Cartesian grid by default
    run_cartesian_grid(args)
