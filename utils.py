"""Utility functions for evaluation and policy comparison.

This module centralizes evaluation helpers used across the project.
"""
import numpy as np


def evaluate_policy(model, env, num_episodes=10, use_trained=True):
    """Evaluate a trained policy (or random) on the provided env.

    Returns a dict with lists: 'rewards', 'stockouts', 'inventory_levels'.
    """
    results = {
        'rewards': [],
        'stockouts': [],
        'inventory_levels': []
    }

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_stockouts = 0
        ep_inventory = []

        while not done:
            if use_trained and model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += reward
            ep_stockouts += info.get('stockouts', 0)
            ep_inventory.append(info.get('inventory_level', 0.0))
            done = terminated or truncated

        results['rewards'].append(ep_reward)
        results['stockouts'].append(ep_stockouts)
        results['inventory_levels'].append(np.mean(ep_inventory) if len(ep_inventory) > 0 else 0.0)

    return results


def compare_policies(model, env, num_episodes=30):
    """Compare trained RL policy against random and fixed-order baselines.

    Returns a dict mapping policy name -> results dict (same shape as evaluate_policy output).
    """
    # Random policy
    random_results = evaluate_policy(None, env, num_episodes, use_trained=False)

    # Fixed-order baseline (order 30 units per product)
    fixed_results = {'rewards': [], 'stockouts': [], 'inventory_levels': []}

    # Determine num_products
    if hasattr(env, 'num_products'):
        num_products = env.num_products
    elif hasattr(env, 'env') and hasattr(env.env, 'num_products'):
        num_products = env.env.num_products
    else:
        num_products = 3

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        ep_stockouts = 0
        ep_inventory = []

        while not done:
            action = np.array([30.0] * num_products)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_stockouts += info.get('stockouts', 0)
            ep_inventory.append(info.get('inventory_level', 0.0))
            done = terminated or truncated

        fixed_results['rewards'].append(ep_reward)
        fixed_results['stockouts'].append(ep_stockouts)
        fixed_results['inventory_levels'].append(np.mean(ep_inventory) if len(ep_inventory) > 0 else 0.0)

    # Trained RL policy
    rl_results = evaluate_policy(model, env, num_episodes, use_trained=True) if model is not None else random_results

    policies = {
        'Random Policy': random_results,
        'Fixed Order (30 units)': fixed_results,
        'Trained RL Policy': rl_results
    }

    return policies
