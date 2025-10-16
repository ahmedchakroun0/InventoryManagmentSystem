"""Train a model directly from best_config.json without grid search.

Usage:
    python3 train_from_best.py
    python3 train_from_best.py --timesteps 5000000
    python3 train_from_best.py --config best_config.json --timesteps 10000000
"""

import argparse
import json
import time
import numpy as np
from trainer import Trainer
from tester import Tester
from visulizations import visulisor



def main():
    parser = argparse.ArgumentParser(
        description='Train RL agent from best configuration'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='best_config.json',
        help='Path to best config JSON file'
    )
    
    
    args = parser.parse_args()
    
    # Load best config
    print(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Extract parameters
    hc = float(config['holding_cost'])
    of = float(config['order_fixed'])
    sp = float(config['stockout_penalty'])
    lr = float(config['learning_rate'])
    n_steps = int(config['n_steps'])
    n_epochs = int(config['n_epochs'])
    vf_coef = float(config['vf_coef'])
    ent_coef = float(config['ent_coef'])
    batch_size = int(config['batch_size'])
    gamma = float(config.get('gamma', 0.995))
    
    print(f"\nTraining Configuration:")
    print(f"  Economic: holding={hc}, order_fixed={of}, stockout={sp}")
    print(f"  Hyperparameters: lr={lr}, n_steps={n_steps}, n_epochs={n_epochs}")
    print(f"  Training: {args.timesteps:,} timesteps with {args.n_envs} parallel envs")
    print(f"  Total steps per env: {args.timesteps // args.n_envs:,}\n")
    
    # Create trainer
    trainer = Trainer(
        holding_cost_per_unit=hc,
        order_cost_fixed=of,
        stockout_penalty=sp,
        n_envs=args.n_envs,
        learning_rate=lr,
        n_steps=n_steps,
        n_epochs=n_epochs,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        batch_size=batch_size,
        gamma=gamma,
        save_path=args.output
    )
    
    # Train
    start_time = time.time()
    print("Starting training...")
    
    train_time = trainer.train(args.timesteps)
    trainer.save(args.output)
    total_time = time.time() - start_time
    
    # Extract training rewards from the model's episode info buffer
    training_rewards = []
    if hasattr(trainer.model, 'ep_info_buffer') and len(trainer.model.ep_info_buffer) > 0:
        training_rewards = [ep_info['r'] for ep_info in trainer.model.ep_info_buffer]
    
    print(f"Collected {len(training_rewards)} episode rewards from training")
    
    print(f"\n✓ Training complete in {total_time:.1f}s")
    print(f"✓ Model saved to: {args.output}.zip")
    
    # Evaluate
    print(f"\nEvaluating trained model over {args.n_eval} episodes...")
    tester = Tester(holding_cost_per_unit=hc, order_cost_fixed=of, stockout_penalty=sp)
    
    rl_results = tester.test_trained(model=trainer.model, episodes=args.n_eval)
    random_results = tester.test_random(episodes=args.n_eval)
    fixed_results = tester.test_fixed_order(target=30.0, episodes=args.n_eval)
    
    # Print comparison
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nTrained RL Policy:")
    print(f"  Avg Reward: {rl_results['mean_reward']:.2f} ± {rl_results['std_reward']:.2f}")
    print(f"  Avg Stockouts: {rl_results['mean_stockout']:.2f}")
    print(f"  Avg Inventory: {rl_results['mean_inventory']:.2f}")
    
    print(f"\nRandom Policy (baseline):")
    print(f"  Avg Reward: {random_results['mean_reward']:.2f} ± {random_results['std_reward']:.2f}")
    print(f"  Avg Stockouts: {random_results['mean_stockout']:.2f}")
    print(f"  Avg Inventory: {random_results['mean_inventory']:.2f}")
    
    print(f"\nFixed Order Policy (baseline):")
    print(f"  Avg Reward: {fixed_results['mean_reward']:.2f} ± {fixed_results['std_reward']:.2f}")
    print(f"  Avg Stockouts: {fixed_results['mean_stockout']:.2f}")
    print(f"  Avg Inventory: {fixed_results['mean_inventory']:.2f}")
    print("="*60)
    
    # Save results
    eval_json = args.output + '_eval.json'
    all_results = {
        'Trained RL Policy': rl_results['per_episode'],
        'Random Policy': random_results['per_episode'],
        'Fixed Order (30 units)': fixed_results['per_episode']
    }
    
    with open(eval_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to: {eval_json}")
    
    # Visualize with training rewards
    vis = visulisor()
    if len(training_rewards) > 0:
        vis.set_training_rewards(training_rewards)
    vis.set_policy_comparison(all_results)
    image_path = args.output + '_comparison.png'
    vis.compare_and_plot(out_path=image_path, show=False)
    print(f"✓ Comparison plot saved to: {image_path}")


if __name__ == '__main__':
    main()
