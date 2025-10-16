"""End-to-end pipeline: grid search, retrain best config, evaluate and visualize.

This script uses the local `trainer.Trainer`, `tester.Tester`, and
`visulizations.visulisor` to run a compact combined pipeline. It is intended
to be a convenience wrapper for quick experiments; for large sweeps prefer
running the dedicated `grid_search.py` or other experiment managers.

Usage:
    python3 pipeline.py --config pipeline_config.json
    python3 pipeline.py  # uses pipeline_config.json by default

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
    
    # Extract individual hyperparameters from best config
    learning_rate = float(best.get('learning_rate', 3e-5))
    n_steps = int(best.get('n_steps', 2048))
    n_epochs = int(best.get('n_epochs', 10))
    vf_coef = float(best.get('vf_coef', 0.25))
    ent_coef = float(best.get('ent_coef', 0.0))
    batch_size = int(best.get('batch_size', 64))
    gamma = float(best.get('gamma', 0.995))

    print(f"\nRetraining best config: holding={hc}, order_fixed={of}, stockout={sp}")
    print(f"  HP: lr={learning_rate}, n_steps={n_steps}, n_epochs={n_epochs}, vf={vf_coef}, ent={ent_coef}, batch={batch_size}, gamma={gamma}")

    trainer = Trainer(
        holding_cost_per_unit=hc,
        order_cost_fixed=of,
        stockout_penalty=sp,
        n_envs=n_envs,
        learning_rate=learning_rate,
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
    
    # Extract training rewards from the model's episode info buffer
    training_rewards = []
    if hasattr(trainer.model, 'ep_info_buffer') and len(trainer.model.ep_info_buffer) > 0:
        training_rewards = [ep_info['r'] for ep_info in trainer.model.ep_info_buffer]
    
    print(f"Collected {len(training_rewards)} episode rewards from training")

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

    # Visualization with training rewards
    vis = visulisor()
    if len(training_rewards) > 0:
        vis.set_training_rewards(training_rewards)
    vis.set_policy_comparison(all_results)
    vis.compare_and_plot(out_path=out_image, show=False)
    print(f"Saved pipeline comparison figure to {out_image}")

    return {
        'train_time_s': train_time,
        'evaluation_path': out_eval_json,
        'model_path': out_model_path
    }


def main():
    parser = argparse.ArgumentParser(description='Run end-to-end pipeline from config file')
    parser.add_argument('--config', type=str, default='pipeline_config.json', 
                        help='Path to pipeline JSON config file')
    args = parser.parse_args()

    # Load config file
    import json
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    # Read all parameters from config
    timesteps_grid = cfg.get('timesteps_grid', 20000)
    timesteps_final = cfg.get('timesteps_final', 200000)
    n_envs = cfg.get('n_envs', 4)
    n_eval = cfg.get('n_eval', 5)
    max_combinations = cfg.get('max_combinations', 2) if cfg.get('max_combinations', 0) > 0 else None
    out_csv = cfg.get('out_csv', 'pipeline_grid_results.csv')
    out_model = cfg.get('out_model', 'pipeline_best_model')
    out_eval = cfg.get('out_eval', 'pipeline_eval.json')
    out_image = cfg.get('out_image', 'pipeline_comparison.png')
    best_config_json = cfg.get('best_config_json', 'best_config.json')

    print('Starting pipeline using full Cartesian grid search...')
    start_all = time.time()

    # Prepare args Namespace for grid_search.run_independent_grid from our config
    import argparse as _argparse

    # Build hp lists from cfg['hp_grid'] if provided, else use sensible defaults
    
    learning_rate_list = cfg.get('learning_rate_list', [3e-5, 1e-4])
    n_steps_list = cfg.get('n_steps_list', [512, 2048]) 
    n_epochs_list = cfg.get('n_epochs_list', [5, 10])
    vf_coef_list = cfg.get('vf_coef_list', [0.25, 0.5])
    ent_coef_list = cfg.get('ent_coef_list', [0.0, 0.01])
    batch_size = cfg.get('batch_size', 64)
    
    holding_costs = cfg.get('holding_costs', [0.5, 1.5, 3.0])
    order_fixed = cfg.get('order_fixed', [5.0, 20.0, 50.0])
    stockout_penalties = cfg.get('stockout_penalties', [10.0, 50.0, 200.0])

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
        n_envs=n_envs,
        max_combinations=max_combinations,
        timesteps=timesteps_grid,
        eval_episodes=n_eval,
        out_csv=out_csv,
        save_models=False
    )

    # grid_search will write <out_csv>_cartesian.csv and <out_csv>_cartesian_best_config.json
    #gs_out_csv, gs_best_json = grid_search.run_cartesian_grid(gs_args)

    # Load the best-config JSON produced by grid_search
    import json
    with open(best_config_json, 'r') as f:
        best_params = json.load(f)

    # Construct best dict with all required parameters
    best = {
        'holding_cost': best_params.get('holding_cost', holding_costs),
        'order_fixed': best_params.get('order_fixed', order_fixed),
        'stockout_penalty': best_params.get('stockout_penalty', stockout_penalties),
        'learning_rate': best_params.get('learning_rate', learning_rate_list),
        'n_steps': best_params.get('n_steps', n_steps_list),
        'n_epochs': best_params.get('n_epochs', n_epochs_list),
        'vf_coef': best_params.get('vf_coef', vf_coef_list),
        'ent_coef': best_params.get('ent_coef', ent_coef_list),
        'batch_size': best_params.get('batch_size', batch_size),
        'gamma': best_params.get('gamma', 0.995)
    }

    print('\nBest configuration from Cartesian grid:', best)

    result = retrain_best_and_evaluate(best, timesteps_final, n_envs, n_eval, out_model, out_eval, out_image=out_image)

    total_time = time.time() - start_all
    print(f"\nPipeline complete in {total_time:.1f}s. Artifacts: model={result['model_path']}, eval={result['evaluation_path']}")


if __name__ == '__main__':
    main()
