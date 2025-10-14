"""End-to-end pipeline: grid search, retrain best config, evaluate and visualize.

This script uses the local `trainer.Trainer`, `tester.Tester`, and
`visulizations.visulisor` to run a compact combined pipeline. It is intended
to be a convenience wrapper for quick experiments; for large sweeps prefer
running the dedicated `grid_search.py` or other experiment managers.

Usage (smoke):
    python3 pipeline.py --timesteps_grid 2000 --timesteps_final 2000 --max-combinations 2 --n_eval 2 --out_csv pipeline_results.csv

"""
from typing import List, Tuple, Dict, Any
import argparse
import itertools
import csv
import time
import numpy as np

from trainer import Trainer
from tester import Tester
from visulizations import visulisor
import grid_search


def retrain_best_and_evaluate(best: Dict[str, Any], timesteps_final: int, n_envs: int, n_eval: int, out_model_path: str, out_eval_json: str, out_image: str = 'pipeline_comparison.png'):
    # Parse best config
    hc = float(best['holding_cost'])
    of = float(best['order_fixed'])
    sp = float(best['stockout_penalty'])
    # hp in 'best' may contain individual hyperparameter keys (learning_rate, n_steps, ...)
    hp_src = best.get('hp', {}) if isinstance(best.get('hp', {}), dict) else {}

    print(f"\nRetraining best config: holding={hc}, order_fixed={of}, stockout={sp}")

    # Resolve hyperparameters: prefer values from best (best JSON), else fall back to top-level config lists
    import json as _json
    try:
        # load pipeline config to access defaults (if any)
        with open('pipeline_config.json', 'r') as _f:
            _cfg = _json.load(_f)
    except Exception:
        _cfg = {}

    def _first_or(value, cfg_key, default):
        if value is not None:
            return value
        lst = _cfg.get(cfg_key, None)
        if isinstance(lst, list) and len(lst) > 0:
            return lst[0]
        return default

    lr = float(_first_or(hp_src.get('learning_rate') or best.get('learning_rate'), 'learning_rate_list', 3e-5))
    n_steps = int(_first_or(hp_src.get('n_steps') or best.get('n_steps'), 'n_steps_list', 2048))
    n_epochs = int(_first_or(hp_src.get('n_epochs') or best.get('n_epochs'), 'n_epochs_list', 10))
    vf_coef = float(_first_or(hp_src.get('vf_coef') or best.get('vf_coef'), 'vf_coef_list', 0.25))
    ent_coef = float(_first_or(hp_src.get('ent_coef') or best.get('ent_coef'), 'ent_coef_list', 0.0))
    batch_size = int(_first_or(hp_src.get('batch_size') or best.get('batch_size'), 'batch_size', 64))
    gamma = float(_cfg.get('gamma', 0.995))

    print(f"Using hyperparameters: lr={lr}, n_steps={n_steps}, n_epochs={n_epochs}, vf_coef={vf_coef}, ent_coef={ent_coef}, batch_size={batch_size}, gamma={gamma}")

    trainer = Trainer(
        holding_cost_per_unit=hc,
        order_cost_fixed=of,
        stockout_penalty=sp,
        n_envs=n_envs,
        learning_rate=lr,
        n_steps=n_steps,
        n_epochs=n_epochs,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        batch_size=batch_size,
        gamma=gamma,
        save_path=out_model_path
    )

    train_time = trainer.train(timesteps_final)
    trainer.save(out_model_path)

    tester = Tester(holding_cost_per_unit=hc, order_cost_fixed=of, stockout_penalty=sp)
    rl_results = tester.test_trained(model=trainer.model, episodes=n_eval)
    random_results = tester.test_random(episodes=n_eval)
    fixed_results = tester.test_fixed_order(target=30.0, episodes=n_eval)

    # Save evaluation JSON
    all_results = {
        'Trained RL Policy': rl_results['per_episode'],
        'Random Policy': random_results['per_episode'],
        'Fixed Order (30 units)': fixed_results['per_episode']
    }

    import json
    with open(out_eval_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved evaluation comparisons to {out_eval_json}")

    # Visualization
    vis = visulisor()
    vis.set_policy_comparison(all_results)
    vis.compare_and_plot(out_path=out_image, show=False)
    print(f"Saved pipeline comparison figure to {out_image}")

    return {
        'train_time_s': train_time,
        'evaluation_path': out_eval_json,
        'model_path': out_model_path
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps_grid', type=int, default=20000)
    parser.add_argument('--timesteps_final', type=int, default=200000)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--n_eval', type=int, default=5)
    parser.add_argument('--max-combinations', type=int, default=0)
    parser.add_argument('--out_csv', type=str, default='pipeline_grid_results.csv')
    parser.add_argument('--out_model', type=str, default='pipeline_best_model')
    parser.add_argument('--out_eval', type=str, default='pipeline_eval.json')
    parser.add_argument('--smoke', action='store_true', help='Run a tiny smoke pipeline (very short timesteps)')
    parser.add_argument('--config', type=str, default='pipeline_config.json', help='Path to pipeline JSON config')

    args = parser.parse_args()

    # Load config file
    import json
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    if args.smoke:
        timesteps_grid = 2000
        timesteps_final = 2000
        max_combinations = 2
        n_eval = 2
        out_csv = 'pipeline_grid_results_smoke.csv'
        out_model = 'pipeline_best_model_smoke'
        out_eval = 'pipeline_eval_smoke.json'
        out_image = 'pipeline_comparison_smoke.png'
    else:
        timesteps_grid = cfg.get('timesteps_grid', args.timesteps_grid)
        timesteps_final = cfg.get('timesteps_final', args.timesteps_final)
        max_combinations = cfg.get('max_combinations', args.max_combinations) if cfg.get('max_combinations', args.max_combinations) > 0 else None
        n_eval = cfg.get('n_eval', args.n_eval)
        out_csv = cfg.get('out_csv', args.out_csv)
        out_model = cfg.get('out_model', args.out_model)
        out_eval = cfg.get('out_eval', args.out_eval)
        out_image = cfg.get('out_image', 'pipeline_comparison.png')

    print('Starting pipeline using grid_search independent scans...')
    start_all = time.time()

    # Prepare args Namespace for grid_search.run_independent_grid from our config
    import argparse as _argparse

    # Build hp lists from cfg['hp_grid'] if provided, else use sensible defaults
    hp_grid = cfg.get('hp_grid', [])
    if hp_grid and isinstance(hp_grid, list):
        learning_rate_list = [hp.get('learning_rate', 3e-5) for hp in hp_grid]
        n_steps_list = [hp.get('n_steps', 512) for hp in hp_grid]
        n_epochs_list = [hp.get('n_epochs', 5) for hp in hp_grid]
        vf_coef_list = [hp.get('vf_coef', 0.25) for hp in hp_grid]
        ent_coef_list = [hp.get('ent_coef', 0.0) for hp in hp_grid]
        batch_size = hp_grid[0].get('batch_size', 64)
    else:
        learning_rate_list = [3e-5, 1e-4]
        n_steps_list = [512, 2048]
        n_epochs_list = [5, 10]
        vf_coef_list = [0.25, 0.5]
        ent_coef_list = [0.0, 0.01]
        batch_size = 64

    econ = cfg.get('econ_grid', {})
    holding_costs = econ.get('holding_costs', [0.5, 1.5, 3.0])
    order_fixed = econ.get('order_fixed', [5.0, 20.0, 50.0])
    stockout_penalties = econ.get('stockout_penalties', [10.0, 50.0, 200.0])

    gs_args = _argparse.Namespace(
        holding_costs=holding_costs,
        order_fixed=order_fixed,
        stockout_penalties=stockout_penalties,
        learning_rate_list=learning_rate_list,
        n_steps_list=n_steps_list,
        n_epochs_list=n_epochs_list,
        vf_coef_list=vf_coef_list,
        ent_coef_list=ent_coef_list,
        batch_size=batch_size,
        gamma=cfg.get('gamma', 0.995),
        reward_scale=cfg.get('reward_scale', 1000.0),
        n_envs=args.n_envs,
        timesteps=timesteps_grid,
        eval_episodes=n_eval,
        out_csv=out_csv,
        save_models=False
    )

    # grid_search will write <out_csv>_cartesian.csv and <out_csv>_cartesian_best_config.json
    gs_out_csv, gs_best_json = grid_search.run_cartesian_grid(gs_args)

    # Load the best-config JSON produced by grid_search
    import json
    with open(gs_best_json, 'r') as f:
        best_params = json.load(f)

    # Use the best combination returned by the cartesian grid directly
    best = best_params

    print('\nBest row from independent scans (approx):', best)

    result = retrain_best_and_evaluate(best, timesteps_final, args.n_envs, n_eval, args.out_model, args.out_eval, out_image=out_image)

    total_time = time.time() - start_all
    print(f"\nPipeline complete in {total_time:.1f}s. Artifacts: model={result['model_path']}, eval={result['evaluation_path']}")


if __name__ == '__main__':
    main()
