"""Trainer class for InventoryEnv.

Provides a simple programmatic interface to create envs, build a PPO agent,
train (with optional early stopping), evaluate, save and load models.

Example:
    from trainer import Trainer
    t = Trainer()
    t.train(timesteps=20000)
    results = t.evaluate(episodes=10)

"""
from typing import Optional
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from InventoryEnv import InventoryEnv
from utils import evaluate_policy


class Trainer:
    """A simple trainer for the Inventory management environment.

    Args:
        holding_cost_per_unit: holding cost used by the env
        order_cost_fixed: fixed ordering cost
        stockout_penalty: penalty per unit stockout
        n_envs: number of parallel envs for training
        hp: dict of PPO hyperparameters (learning_rate, n_steps, n_epochs, vf_coef, ent_coef, batch_size, gamma)
        reward_scale: reward scaling applied by the env
        device: 'cpu' or 'cuda'
        save_path: default model path when saving
    """

    def __init__(
        self,
        holding_cost_per_unit: float = 0.5,
        order_cost_fixed: float = 5.0,
        stockout_penalty: float = 50.0,
        n_envs: int = 4,
        # Explicit hyperparameters (preferred):
        learning_rate: Optional[float] = 3e-5,
        n_steps: Optional[int] = 2048,
        n_epochs: Optional[int] = 10,
        vf_coef: Optional[float] = 0.25,
        ent_coef: Optional[float] = 0.0,
        batch_size: Optional[int] = 64,
        gamma: Optional[float] = 0.995,
        reward_scale: float = 1000.0,
        device: str = 'cpu',
        save_path: str = 'inventory_ppo_model'
    ):
        self.holding_cost_per_unit = holding_cost_per_unit
        self.order_cost_fixed = order_cost_fixed
        self.stockout_penalty = stockout_penalty
        self.n_envs = n_envs
        self.reward_scale = reward_scale
        self.device = device
        self.save_path = save_path

        # default hyperparameters stored as scalars
        self.hp = {}
        self.hp['learning_rate'] = float(learning_rate)
        self.hp['n_steps'] = int(n_steps)
        self.hp['n_epochs'] = int(n_epochs)
        self.hp['vf_coef'] = float(vf_coef)
        self.hp['ent_coef'] = float(ent_coef)
        self.hp['batch_size'] = int(batch_size)
        self.hp['gamma'] = float(gamma)

        # placeholders
        self.model: Optional[PPO] = None
        self.vec_env = None

    def _make_env(self):
        """Create a single Monitor-wrapped InventoryEnv with configured economics."""
        def _init():
            env = InventoryEnv(
                num_products=3,
                order_up_to=True,
                discrete_actions=False,
                holding_cost_per_unit=self.holding_cost_per_unit,
                order_cost_fixed=self.order_cost_fixed,
                order_cost_per_unit=1.0,
                stockout_penalty=self.stockout_penalty,
                reward_scale=self.reward_scale
            )
            return Monitor(env)
        return _init

    def build_model(self):
        """Construct the PPO model with current hyperparameters and envs."""
        # DummyVecEnv expects a list of callables (functions that return envs).
        # self._make_env() returns such a callable, so create a list of callables
        # without invoking them here.
        env_fns = [self._make_env() for _ in range(self.n_envs)]
        self.vec_env = DummyVecEnv(env_fns)

        self.model = PPO(
            policy='MlpPolicy',
            env=self.vec_env,
            learning_rate=self.hp['learning_rate'],
            n_steps=self.hp['n_steps'],
            batch_size=self.hp['batch_size'],
            n_epochs=self.hp['n_epochs'],
            gamma=self.hp['gamma'],
            ent_coef=self.hp['ent_coef'],
            vf_coef=self.hp['vf_coef'],
            verbose=1,
            device=self.device
        )
        return self.model

    def train(self,
              timesteps: int = 200_000,
              early_stop: bool = False,
              eval_freq: int = 10000,
              patience: int = 3,
              min_delta: float = 1e-3,
              eval_episodes: int = 5,
              save_best: bool = False,
              save_path: Optional[str] = None
              ):
        """Train the PPO agent.

        Supports optional early stopping: train in chunks of size `eval_freq`, evaluate and stop if
        no improvement for `patience` checks.

        Returns: training time in seconds.
        """
        if self.model is None:
            self.build_model()

        save_path = self.save_path if save_path is None else save_path
        start = time.time()

        if early_stop:
            trained = 0
            best_score = -float('inf')
            no_improve = 0
            chunk = min(eval_freq, timesteps)

            while trained < timesteps:
                to_train = min(chunk, timesteps - trained)
                self.model.learn(total_timesteps=to_train)
                trained += to_train

                # quick evaluation
                eval_env = Monitor(InventoryEnv(
                    num_products=3,
                    order_up_to=True,
                    discrete_actions=False,
                    holding_cost_per_unit=self.holding_cost_per_unit,
                    order_cost_fixed=self.order_cost_fixed,
                    order_cost_per_unit=1.0,
                    stockout_penalty=self.stockout_penalty,
                    reward_scale=self.reward_scale
                ))
                rl = evaluate_policy(self.model, eval_env, num_episodes=eval_episodes, use_trained=True)
                score = float(np.mean(rl['rewards']))
                print(f"[early-stop] trained={trained}/{timesteps} eval_avg_reward={score:.2f} best={best_score:.2f}")

                if score > best_score + min_delta:
                    best_score = score
                    no_improve = 0
                    if save_best:
                        self.save(save_path + '.best')
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"Early stopping triggered (no improvement in {patience} evals)")
                    break

            train_time = time.time() - start
        else:
            self.model.learn(total_timesteps=timesteps)
            train_time = time.time() - start

        return train_time

    def evaluate(self, model: Optional[PPO] = None, episodes: int = 100):
        """Evaluate the trained model (or provided model) for `episodes` episodes.

        Returns dict with lists: rewards, stockouts, inventory_levels
        """
        model = self.model if model is None else model
        if model is None:
            raise ValueError('No model available to evaluate')

        eval_env = Monitor(InventoryEnv(
            num_products=3,
            order_up_to=True,
            discrete_actions=False,
            holding_cost_per_unit=self.holding_cost_per_unit,
            order_cost_fixed=self.order_cost_fixed,
            order_cost_per_unit=1.0,
            stockout_penalty=self.stockout_penalty,
            reward_scale=self.reward_scale
        ))

        results = evaluate_policy(model, eval_env, num_episodes=episodes, use_trained=True)
        return results

    def save(self, path: Optional[str] = None):
        """Save the current model to `path` (or default save_path)."""
        if self.model is None:
            raise ValueError('No model to save')
        path = self.save_path if path is None else path
        self.model.save(path)
        print(f"Saved model to: {path}.zip")

    def load(self, path: str):
        """Load a model from disk and attach it to this trainer."""
        self.model = PPO.load(path)
        print(f"Loaded model from: {path}")
        return self.model


if __name__ == '__main__':
    # quick smoke demo when running the file directly
    t = Trainer()
    print('Trainer created with hp =', t.hp)
    print('Running a tiny smoke training (2000 timesteps)')
    t.train(timesteps=2000)
    r = t.evaluate(episodes=2)
    print('Smoke eval:', {k: float(np.mean(v)) for k, v in r.items()})
