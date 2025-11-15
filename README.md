# Inventory Management System with Reinforcement Learning

A sophisticated reinforcement learning system for optimizing inventory management across multiple products using PPO (Proximal Policy Optimization). The system learns optimal ordering policies by balancing inventory costs, stockout penalties, and order expenses.

## 🎯 Overview

This project implements an end-to-end RL pipeline for inventory management that:
- Simulates realistic multi-product inventory dynamics with seasonality
- Trains PPO agents to learn optimal ordering policies
- Performs comprehensive hyperparameter and economic parameter grid search
- Compares learned policies against baseline strategies
- Provides detailed visualizations and performance analysis

## 🏗️ Architecture

### Core Components

**InventoryEnv.py** - Custom Gymnasium environment
- Multi-product inventory simulation (default: 3 products)
- Order-up-to and order-quantity action semantics
- Realistic demand patterns with seasonal and day-of-week variations
- Configurable economic parameters (holding costs, stockout penalties, ordering costs)

**trainer.py** - Training orchestration
- PPO model configuration and training
- Periodic evaluation during training
- Early stopping support
- Model checkpoint management

**tester.py** - Policy evaluation
- Trained model evaluation
- Baseline policy testing (random, fixed-order, A2C)
- Performance metrics collection

**grid_search.py** - Hyperparameter optimization
- Cartesian grid search over economic and RL hyperparameters
- Parallel environment training
- CSV result logging with best configuration extraction

**pipeline.py** - End-to-end automation
- Automated grid search → retraining → evaluation workflow
- Configuration-driven execution
- Comprehensive result visualization

**visulizations.py** - Results visualization
- Training progress curves
- Policy comparison charts (profit, stockouts, inventory)
- Multi-series plotting support

## 📊 State & Action Spaces

### State Space (per timestep)
- Current inventory level (per product, normalized)
- Day of week (0-6, normalized)
- Week of year (1-52, normalized for seasonality)
- Recent sales history (7-day moving average)
- Days since last stockout (per product)

### Action Space
Two modes available:
1. **Order-up-to**: Target inventory level (recommended)
2. **Order-quantity**: Units to order

Both support continuous or discrete action spaces.

## 🚀 Quick Start

### Installation

```bash
## Create virtual environment
python3 -m venv venv
source venv/bin/activate  ## On Windows: venv\Scripts\activate

## Install dependencies from requirements file
pip install -r requirements.txt
```

### Basic Usage

#### 1. Train with Best Known Configuration

```bash
## Uses best_config.json (already included)
python3 train_from_best.py
```

This will:
- Load optimal hyperparameters from `best_config.json`
- Train a PPO agent with the best configuration
- Evaluate against baseline policies
- Generate comparison visualizations

#### 2. Run Full Pipeline (Grid Search + Training + Evaluation)

```bash
## Edit pipeline_config.json to customize parameters
python3 pipeline.py --config pipeline_config.json
```

This executes:
1. Cartesian grid search over economic and RL parameters
2. Identifies best configuration
3. Retrains with extended timesteps
4. Comprehensive evaluation and visualization

#### 3. Custom Grid Search

```bash
python3 grid_search.py \
  --timesteps 50000 \
  --holding-costs 0.5,1.5 \
  --order-fixed 5.0,20.0 \
  --stockout-penalties 50.0,100.0 \
  --learning-rate-list 3e-5,1e-4 \
  --n-steps-list 1024,2048 \
  --max-combinations 20 \
  --out-csv results.csv
```

#### 4. Extract Best Configuration from Grid Search

```bash
## After running grid search
python3 extract_best_config.py --csv results_cartesian.csv --output best_config.json
```

## 📁 Project Structure

```
InventoryManagementSystem/
├── InventoryEnv.py          ## Gymnasium environment
├── trainer.py               ## PPO training wrapper
├── tester.py               ## Policy evaluation
├── grid_search.py          ## Hyperparameter search
├── pipeline.py             ## End-to-end automation
├── visulizations.py        ## Plotting utilities
├── utils.py                ## Shared evaluation functions
├── extract_best_config.py  ## Extract best config from CSV
├── train_from_best.py      ## Direct training from config
├── best_config.json        ## Pre-tuned configuration
├── pipeline_config.json    ## Pipeline settings
├── requirements.txt        ## Python dependencies
├── output.log             ## Sample training log
└── .gitignore             ## Git ignore rules
```

## 🎛️ Configuration

### Economic Parameters

```python
holding_cost_per_unit = 0.5    ## Cost per unit per day
order_cost_fixed = 5.0         ## Fixed cost per order
order_cost_per_unit = 1.0      ## Variable cost per unit
stockout_penalty = 10.0        ## Penalty per unfulfilled unit
```

### Best PPO Hyperparameters (from grid search)

```json
{
  "holding_cost": 0.5,
  "order_fixed": 5.0,
  "stockout_penalty": 10.0,
  "learning_rate": 3e-05,
  "n_steps": 1024,
  "n_epochs": 8,
  "batch_size": 64,
  "gamma": 0.995,
  "vf_coef": 0.25,
  "ent_coef": 0.01
}
```

## 📈 Performance

Based on 365-day episodes with best configuration:

| Policy | Avg Profit | Avg Stockouts | Avg Inventory |
|--------|------------|---------------|---------------|
| **Trained PPO** | **$305.92** | **95.15** | **19.74** |
| Random | -$150+ | 15,000+ | 5-15 |
| Fixed Order | $200-250 | 500-2000 | 30-50 |

Performance metrics from grid search (combo 12):
- Training time: ~105 seconds
- Final average profit: $305.92
- Average stockouts per year: 95.15 units
- Average inventory level: 19.74 units

## 🔬 Advanced Features

### Custom Environment Configuration

```python
from InventoryEnv import InventoryEnv

env = InventoryEnv(
    num_products=5,
    max_inventory=300,
    max_order=150,
    order_up_to=True,
    discrete_actions=False,
    demand_mean=25,
    demand_std=10,
    seasonal_amplitude=0.4
)
```

### Programmatic Training

```python
from trainer import Trainer

trainer = Trainer(
    holding_cost_per_unit=0.5,
    order_cost_fixed=5.0,
    stockout_penalty=10.0,
    learning_rate=3e-5,
    n_steps=1024,
    n_epochs=8,
    batch_size=64,
    gamma=0.995
)

trainer.train(timesteps=200000)
trainer.save('my_model')

## Evaluate the trained model
results = trainer.evaluate(episodes=100)
```

### Evaluation & Comparison

```python
from tester import Tester

tester = Tester(
    holding_cost_per_unit=0.5,
    order_cost_fixed=5.0,
    stockout_penalty=10.0
)

## Test trained model
rl_results = tester.test_trained(model=trainer.model, episodes=100)

## Compare with baselines
baseline_results = tester.test_simple_rl(algo='A2C', timesteps=200000)
random_results = tester.test_random(episodes=100)
fixed_results = tester.test_fixed_order(target=30.0, episodes=100)

## Save results
tester.save_results(rl_results, 'rl_results.json')
```

### Custom Visualization

```python
from visulizations import visulisor

vis = visulisor()

## Set training rewards (list of checkpoint rewards)
vis.set_training_rewards(training_rewards)

## Set policy comparison data
policy_comparison = {
    'Trained PPO': rl_results['per_episode'],
    'Random Baseline': random_results['per_episode'],
    'Fixed Order': fixed_results['per_episode']
}
vis.set_policy_comparison(policy_comparison)

## Generate plots
vis.compare_and_plot(out_path='comparison.png', show=False)
```

## 📊 Visualization Examples

The system generates comprehensive visualizations including:
- Training progress curves (smoothed with 10% window)
- Policy comparison bar charts (profit, stockouts, inventory)
- Multi-baseline comparisons with standard deviations
- Stockout frequency analysis
- Inventory level distributions

Example outputs:
- `pipeline_comparison.png` - Full pipeline results
- `pipeline_best_model_comparison.png` - Model comparison
- `vis_demo.png` - Demo visualization

## 🧪 Grid Search Results

Results are saved in CSV format with columns:

**Economic Parameters:**
- `holding_cost` - Holding cost per unit per day
- `order_fixed` - Fixed ordering cost
- `stockout_penalty` - Penalty per unfulfilled unit

**Hyperparameters:**
- `learning_rate` - PPO learning rate
- `n_steps` - Steps per environment per update
- `n_epochs` - Training epochs per update
- `vf_coef` - Value function coefficient
- `ent_coef` - Entropy coefficient
- `batch_size` - Minibatch size
- `gamma` - Discount factor

**Performance Metrics:**
- `rl_avg_profit` - Average episode profit
- `rl_avg_stockouts` - Average stockouts per year
- `rl_avg_inventory` - Average inventory level
- `train_time_s` - Training time in seconds

Extract best configuration:
```bash
python3 extract_best_config.py --csv grid_results_cartesian.csv --output best_config.json
```

## 🐛 Troubleshooting

### Common Issues

**Low performance after training:**
- Increase training timesteps (500k-1M recommended)
- Ensure sufficient parallel environments (4-8)
- Verify economic parameters are reasonable
- Check reward_scale matches environment (default: 1000.0)

**Unstable training (NaN rewards):**
- Reduce learning rate (try 1e-5)
- Increase n_steps (2048-4096)
- Adjust reward_scale (try 100.0 or 10000.0)
- Check for invalid environment parameters

**Memory errors:**
- Reduce n_envs (try 2-4 instead of 8)
- Reduce batch_size (try 32 instead of 64)
- Reduce n_steps (try 512 instead of 2048)

**Grid search taking too long:**
- Use `--max-combinations` to limit search space
- Reduce `--timesteps` for initial exploration
- Reduce number of hyperparameter values
- Use `--n_envs` to increase parallelization

**Import errors:**
```bash
## Ensure all dependencies are installed
pip install -r requirements.txt

## Verify gymnasium (not gym)
pip uninstall gym
pip install gymnasium
```

## 📝 Pipeline Configuration

Edit `pipeline_config.json` to customize the full pipeline:

```json
{
  "timesteps_grid": 20000,
  "timesteps_final": 200000,
  "n_envs": 4,
  "n_eval": 5,
  "max_combinations": 50,
  "holding_costs": [0.5, 1.5, 3.0],
  "order_fixed": [5.0, 20.0, 50.0],
  "stockout_penalties": [10.0, 50.0, 200.0],
  "learning_rate_list": [3e-5, 1e-4],
  "n_steps_list": [512, 1024, 2048],
  "n_epochs_list": [5, 8],
  "vf_coef_list": [0.25, 0.5],
  "ent_coef_list": [0.0, 0.01],
  "batch_size": 64,
  "gamma": 0.995
}
```

## 🎓 How It Works

### 1. Environment Dynamics

Each episode simulates 365 days of inventory management:
1. Observe current state (inventory, time, sales history)
2. Decide order quantity for each product
3. Receive demand (with seasonality and randomness)
4. Fulfill demand from inventory
5. Calculate profit = revenue - holding costs - stockout penalties - order costs
6. Update state and repeat

### 2. Learning Algorithm (PPO)

The agent learns through:
- **Policy network**: Maps states to action probabilities
- **Value network**: Estimates expected future rewards
- **Clipped objective**: Prevents destructive policy updates
- **Multiple epochs**: Reuses collected experience efficiently

### 3. Reward Function

```
reward = revenue - total_cost

where:
  revenue = units_sold × selling_price (15.0)
  total_cost = holding_cost + stockout_cost + ordering_cost
  
  holding_cost = inventory_level × holding_cost_per_unit
  stockout_cost = unmet_demand × stockout_penalty
  ordering_cost = fixed_cost (if order > 0) + units_ordered × unit_cost
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Multi-warehouse scenarios
- Perishable goods with expiration dates
- Supply chain disruptions and lead time variability
- Alternative RL algorithms (SAC, TD3, DQN)
- Real-world demand data integration
- Multi-echelon inventory systems
- Stochastic lead times
- Supplier constraints and batch ordering

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🔗 References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Inventory Management Theory](https://en.wikipedia.org/wiki/Inventory_control)

## 📧 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review troubleshooting section above

---

**Status**: Production-ready | **Last Updated**: November 2024 | **Python**: 3.8+