"""Extract best configuration from grid search CSV results.

This script reads the grid search results CSV file, identifies the best
configuration (highest mean profit), and saves it to a JSON file.

Usage:
    python3 extract_best_config.py
    python3 extract_best_config.py --csv my_results.csv --output best_config.json
"""

import argparse
import csv
import json
import sys


def extract_best_config(csv_path: str, output_path: str):
    """Read CSV results and extract the best configuration."""
    
    print(f"Reading grid search results from: {csv_path}")
    
    # Read all rows from CSV
    rows = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    if not rows:
        print("Error: CSV file is empty or has no data rows")
        sys.exit(1)
    
    print(f"Found {len(rows)} configurations in grid search results")
    
    # Print available columns for debugging
    print(f"CSV columns: {list(rows[0].keys())}")
    
    # Find the row with maximum rl_avg_profit
    best_row = max(rows, key=lambda r: float(r.get('rl_avg_profit', float('-inf'))))
    
    # Extract parameters
    best_config = {
        # Economic parameters
        'holding_cost': float(best_row['holding_cost']),
        'order_fixed': float(best_row['order_fixed']),
        'stockout_penalty': float(best_row['stockout_penalty']),
        
        # Hyperparameters
        'learning_rate': float(best_row['learning_rate']),
        'n_steps': int(best_row['n_steps']),
        'n_epochs': int(best_row['n_epochs']),
        'vf_coef': float(best_row['vf_coef']),
        'ent_coef': float(best_row['ent_coef']),
        'batch_size': int(best_row['batch_size']),
        'gamma': float(best_row.get('gamma', 0.995)),
        
        # Performance metrics (for reference)
        'rl_avg_profit': float(best_row['rl_avg_profit']),
        'rl_avg_stockouts': float(best_row['rl_avg_stockouts']),
        'rl_avg_inventory': float(best_row['rl_avg_inventory']),
        'train_time_s': float(best_row['train_time_s'])
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n✓ Best configuration saved to: {output_path}")
    print(f"\nBest Configuration:")
    print(f"  Economic Parameters:")
    print(f"    holding_cost: {best_config['holding_cost']}")
    print(f"    order_fixed: {best_config['order_fixed']}")
    print(f"    stockout_penalty: {best_config['stockout_penalty']}")
    print(f"\n  Hyperparameters:")
    print(f"    learning_rate: {best_config['learning_rate']}")
    print(f"    n_steps: {best_config['n_steps']}")
    print(f"    n_epochs: {best_config['n_epochs']}")
    print(f"    vf_coef: {best_config['vf_coef']}")
    print(f"    ent_coef: {best_config['ent_coef']}")
    print(f"    batch_size: {best_config['batch_size']}")
    print(f"    gamma: {best_config['gamma']}")
    print(f"\n  Performance:")
    print(f"    rl_avg_profit: {best_config['rl_avg_profit']:.2f}")
    print(f"    rl_avg_stockouts: {best_config['rl_avg_stockouts']:.2f}")
    print(f"    rl_avg_inventory: {best_config['rl_avg_inventory']:.2f}")
    print(f"    train_time_s: {best_config['train_time_s']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract best configuration from grid search CSV results'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='pipeline_grid_results_cartesian.csv',
        help='Path to the grid search results CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='best_config.json',
        help='Path to save the best configuration JSON file'
    )
    
    args = parser.parse_args()
    
    extract_best_config(args.csv, args.output)


if __name__ == '__main__':
    main()
