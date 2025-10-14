"""Tester utilities for the Inventory agent.

Provides a Tester class that can:
- load a PPO model
- evaluate a trained model over multiple episodes
- evaluate baseline policies: random and fixed-order (order up to a target)
- save results as JSON

When run as a script it performs a tiny smoke test (random baseline, 2 episodes).
"""
from typing import Optional, Dict, Any
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from InventoryEnv import InventoryEnv
from utils import evaluate_policy


class Tester:
    """Helper to evaluate InventoryEnv agents and baselines.

    Args:
        holding_cost_per_unit, order_cost_fixed, stockout_penalty: economics to use when creating envs
        reward_scale: reward scaling used by env
    """

    def __init__(
        self,
        holding_cost_per_unit: float = 0.5,
        order_cost_fixed: float = 5.0,
        stockout_penalty: float = 50.0,
        reward_scale: float = 1000.0,
        num_products: int = 3
    ):
        self.holding_cost_per_unit = holding_cost_per_unit
        self.order_cost_fixed = order_cost_fixed
        self.stockout_penalty = stockout_penalty
        self.reward_scale = reward_scale
        self.num_products = num_products
        self.model: Optional[PPO] = None

    def load_model(self, path: str) -> PPO:
        """Load a PPO model from disk and keep a reference."""
        self.model = PPO.load(path)
        return self.model

    def _make_eval_env(self) -> Monitor:
        env = InventoryEnv(
            num_products=self.num_products,
            order_up_to=True,
            discrete_actions=False,
            holding_cost_per_unit=self.holding_cost_per_unit,
            order_cost_fixed=self.order_cost_fixed,
            order_cost_per_unit=1.0,
            stockout_penalty=self.stockout_penalty,
            reward_scale=self.reward_scale
        )
        return Monitor(env)

    def test_trained(self, model: Optional[PPO] = None, episodes: int = 100) -> Dict[str, Any]:
        """Evaluate a trained model (or the one previously loaded).

        If no model is available, raises ValueError.
        Returns a dict with lists: 'rewards','stockouts','inventory_levels' and aggregated means.
        """
        model = self.model if model is None else model
        if model is None:
            raise ValueError('No trained model provided or loaded')

        eval_env = self._make_eval_env()
        results = evaluate_policy(model, eval_env, num_episodes=episodes, use_trained=True)
        aggregated = {k: float(np.mean(v)) for k, v in results.items()}
        return {'per_episode': results, 'aggregated': aggregated}

    def test_random(self, episodes: int = 100) -> Dict[str, Any]:
        """Evaluate a random policy baseline using evaluate_policy with use_trained=False."""
        eval_env = self._make_eval_env()
        results = evaluate_policy(None, eval_env, num_episodes=episodes, use_trained=False)
        aggregated = {k: float(np.mean(v)) for k, v in results.items()}
        return {'per_episode': results, 'aggregated': aggregated}

    def test_fixed_order(self, target: float = 30.0, episodes: int = 100) -> Dict[str, Any]:
        """Evaluate a simple fixed policy: always order up to `target` units per product.

        Returns the same result structure as other test methods.
        """
        env = InventoryEnv(
            num_products=self.num_products,
            order_up_to=False,  # use order-quantity semantics for stepping
            discrete_actions=False,
            holding_cost_per_unit=self.holding_cost_per_unit,
            order_cost_fixed=self.order_cost_fixed,
            order_cost_per_unit=1.0,
            stockout_penalty=self.stockout_penalty,
            reward_scale=self.reward_scale
        )
        env = Monitor(env)

        per_episode = {'rewards': [], 'stockouts': [], 'inventory_levels': []}

        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_stockouts = 0.0
            ep_inv = []
            while not done:
                # Compute action (order quantity) to reach target inventory
                # Inventory stored on env.env.inventory because Monitor wraps it.
                current_inv = env.env.inventory
                target_arr = np.array([target] * self.num_products)
                action = np.maximum(0.0, target_arr - current_inv)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_stockouts += info.get('stockouts', 0)
                ep_inv.append(info.get('inventory_level', np.mean(env.env.inventory)))
                done = terminated or truncated
            per_episode['rewards'].append(ep_reward)
            per_episode['stockouts'].append(ep_stockouts)
            per_episode['inventory_levels'].append(np.mean(ep_inv))

        aggregated = {k: float(np.mean(v)) for k, v in per_episode.items()}
        return {'per_episode': per_episode, 'aggregated': aggregated}

    def save_results(self, results: Dict[str, Any], path: str):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved results to {path}')


if __name__ == '__main__':
    # Tiny smoke test: run random baseline for 2 episodes and print stats
    tester = Tester()
    print('Running smoke random baseline (2 episodes)')
    r = tester.test_random(episodes=2)
    print('Aggregated:', r['aggregated'])
