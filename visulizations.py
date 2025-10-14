"""Visualization helper that mirrors the plotting style in InventoryEnv.py

Creates a class `visulisor` that accepts training rewards and policy comparison
results and produces comparison figures (average profit, stockouts, inventory)
and training reward curves. It can load policy comparison results from JSON
or accept them as dictionaries.

The class intentionally follows the aesthetics used in InventoryEnv.plot_results.

Usage:
    from visulizations import visulisor
    v = visulisor()
    v.compare_and_plot(training_rewards, policy_comparison, out_path='myplot.png')

"""
from typing import Dict, Any, Optional
import json
import numpy as np
import matplotlib.pyplot as plt


class visulisor:
    """Visualization helper.

    Args:
        training_rewards: optional list of training checkpoint rewards (mean episode reward per checkpoint)
        policy_comparison: optional dict mapping policy name -> results dict, where each results dict
            must have lists under keys: 'rewards', 'stockouts', 'inventory_levels'
    """

    def __init__(self, training_rewards: Optional[list] = None, policy_comparison: Optional[Dict[str, Dict]] = None):
        self.training_rewards = [] if training_rewards is None else list(training_rewards)
        self.policy_comparison = {} if policy_comparison is None else dict(policy_comparison)

    @staticmethod
    def load_policy_comparison_from_json(path: str) -> Dict[str, Dict[str, list]]:
        """Load a saved policy comparison JSON file.

        Expected JSON structure:
        {
            "Random Policy": {"rewards": [...], "stockouts": [...], "inventory_levels": [...]},
            "Fixed Order (30 units)": { ... },
            "Trained RL Policy": { ... }
        }
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def set_training_rewards(self, rewards: list):
        self.training_rewards = list(rewards)

    def set_policy_comparison(self, policy_comparison: Dict[str, Dict[str, list]]):
        # Basic validation
        for name, results in policy_comparison.items():
            if not all(k in results for k in ('rewards', 'stockouts', 'inventory_levels')):
                raise ValueError(f"Policy '{name}' missing required keys. Expected 'rewards','stockouts','inventory_levels'.")
        self.policy_comparison = dict(policy_comparison)

    def compare_and_plot(self, out_path: str = 'inventory_training_results.png', show: bool = False):
        """Create the same 2x2 visualization as InventoryEnv.plot_results.

        - Top-left: Training progress (if available)
        - Top-right: Average profit bar chart by policy
        - Bottom-left: Average stockouts bar chart
        - Bottom-right: Average inventory bar chart

        Saves the figure to `out_path`. If show=True, also calls plt.show().
        """
        policy_comparison = self.policy_comparison

        # Prepare figure layout
        if policy_comparison:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            axes = list(axes) + [None, None]

        # Plot 1: Training progress
        if len(self.training_rewards) > 0 and axes[0] is not None:
            axes[0].plot(self.training_rewards, linewidth=2, color='blue', alpha=0.85)
            axes[0].set_xlabel('Training Checkpoints', fontsize=12)
            axes[0].set_ylabel('Mean Episode Reward ($)', fontsize=12)
            axes[0].set_title('Training Progress: Reward Over Time', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
        else:
            if axes[0] is not None:
                axes[0].text(0.5, 0.5, 'Training rewards not provided', ha='center', va='center', fontsize=12)
                axes[0].set_xticks([])
                axes[0].set_yticks([])

        # Plot policy comparisons if provided
        if policy_comparison and axes[1] is not None:
            policy_names = list(policy_comparison.keys())
            avg_rewards = [np.mean(policy_comparison[p]['rewards']) for p in policy_names]
            colors = ['#e74c3c', '#f39c12', '#2ecc71']

            bars = axes[1].bar(policy_names, avg_rewards, color=colors[:len(policy_names)], alpha=0.8, edgecolor='black')
            axes[1].set_ylabel('Average Profit ($)', fontsize=12)
            axes[1].set_title('Policy Comparison: Average Profit', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].tick_params(axis='x', rotation=15)

            for bar, val in zip(bars, avg_rewards):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width() / 2., height, f'${val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Stockouts
            avg_stockouts = [np.mean(policy_comparison[p]['stockouts']) for p in policy_names]
            bars = axes[2].bar(policy_names, avg_stockouts, color=colors[:len(policy_names)], alpha=0.8, edgecolor='black')
            axes[2].set_ylabel('Average Stockouts per Year', fontsize=12)
            axes[2].set_title('Policy Comparison: Stockout Frequency', fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3, axis='y')
            axes[2].tick_params(axis='x', rotation=15)
            for bar, val in zip(bars, avg_stockouts):
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width() / 2., height, f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            # Inventory
            avg_inventory = [np.mean(policy_comparison[p]['inventory_levels']) for p in policy_names]
            bars = axes[3].bar(policy_names, avg_inventory, color=colors[:len(policy_names)], alpha=0.8, edgecolor='black')
            axes[3].set_ylabel('Average Inventory Level', fontsize=12)
            axes[3].set_title('Policy Comparison: Inventory Management', fontsize=14, fontweight='bold')
            axes[3].grid(True, alpha=0.3, axis='y')
            axes[3].tick_params(axis='x', rotation=15)
            for bar, val in zip(bars, avg_inventory):
                height = bar.get_height()
                axes[3].text(bar.get_x() + bar.get_width() / 2., height, f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            if axes[1] is not None:
                axes[1].text(0.5, 0.5, 'No policy comparison data available', ha='center', va='center', fontsize=12)
                axes[1].set_xticks([])
                axes[1].set_yticks([])

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved as '{out_path}'")
        if show:
            plt.show()
        plt.close(fig)

    def quick_demo(self, out_path: str = 'vis_demo.png'):
        """Generate a quick synthetic demo plot and save it.

        Useful for smoke-testing the plotting pipeline.
        """
        # Create synthetic training rewards and policy comparison data
        if len(self.training_rewards) == 0:
            self.training_rewards = list(np.linspace(10, 300, num=30) + np.random.randn(30) * 20)

        if not self.policy_comparison:
            self.policy_comparison = {
                'Random Policy': {
                    'rewards': list(np.random.normal(100, 50, size=10)),
                    'stockouts': list(np.random.normal(200, 50, size=10)),
                    'inventory_levels': list(np.random.normal(80, 20, size=10))
                },
                'Fixed Order (30 units)': {
                    'rewards': list(np.random.normal(150, 40, size=10)),
                    'stockouts': list(np.random.normal(120, 40, size=10)),
                    'inventory_levels': list(np.random.normal(90, 15, size=10))
                },
                'Trained RL Policy': {
                    'rewards': list(np.random.normal(220, 30, size=10)),
                    'stockouts': list(np.random.normal(10, 5, size=10)),
                    'inventory_levels': list(np.random.normal(60, 10, size=10))
                }
            }

        self.compare_and_plot(out_path=out_path, show=False)


if __name__ == '__main__':
    # Quick smoke demo to validate the visualizer
    v = visulisor()
    print('Running quick visualization demo...')
    v.quick_demo(out_path='vis_demo.png')
    print('Saved sample visualization to vis_demo.png')
