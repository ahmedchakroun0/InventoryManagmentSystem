import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

class InventoryEnv(gym.Env):
    """
    Custom Inventory Management Environment
    
    State Space:
        - Current inventory level (for each product)
        - Day of week (0-6)
        - Week of year (1-52) for seasonality
        - Recent sales history (moving average)
        - Days since last stockout
    
    Action Space:
        - Order quantity (continuous: 0 to max_order)
    
    Reward:
        profit = revenue - holding_cost - stockout_penalty - ordering_cost
    """
    
    def __init__(self, 
                 num_products=3,
                 max_inventory=200,
                 max_order=100,
                 order_threshold=0.01,
                 discrete_actions=False,
                 n_action_bins=21,
                 order_up_to=False,
                 reward_scale=1000.0,
                 holding_cost_per_unit=0.5,
                 stockout_penalty=50.0,
                 order_cost_fixed=5.0,
                 order_cost_per_unit=1.0,
                 selling_price=15.0,
                 lead_time=0,  # Immediate delivery for simplicity
                 demand_mean=20,
                 demand_std=8,
                 seasonal_amplitude=0.3):
        
        super(InventoryEnv, self).__init__()
        
        # Environment parameters
        self.num_products = num_products
        self.max_inventory = max_inventory
        self.max_order = max_order
        # Minimum effective order quantity (to ignore tiny float outputs from policies)
        self.order_threshold = order_threshold
        self.holding_cost_per_unit = holding_cost_per_unit
        self.stockout_penalty = stockout_penalty
        self.order_cost_fixed = order_cost_fixed
        self.order_cost_per_unit = order_cost_per_unit
        self.selling_price = selling_price
        # Scale factor applied to the reward returned to the RL algorithm
        # (divide large profits/losses to keep training stable)
        self.reward_scale = reward_scale
        self.lead_time = lead_time
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.seasonal_amplitude = seasonal_amplitude
        
        # State variables
        self.inventory = None
        self.day = None
        self.week = None
        self.sales_history = None
        self.days_since_stockout = None
        self.total_revenue = 0
        self.total_cost = 0
        self.stockout_count = 0
        
        # Action space: order quantity for each product
        # Ensure discrete flags are available before configuring action space
        self.discrete_actions = discrete_actions
        self.n_action_bins = int(n_action_bins)
        self.order_up_to = order_up_to
        if self.order_up_to:
            # action represents target inventory level (fraction if continuous, bin if discrete)
            if self.discrete_actions:
                self.action_space = spaces.MultiDiscrete([self.n_action_bins] * self.num_products)
                self._bin_to_target = np.linspace(0, self.max_inventory, self.n_action_bins).astype(np.float32)
            else:
                # continuous normalized fraction 0..1 representing target fraction of max_inventory
                self.action_space = spaces.Box(low=0.0, high=1.0, shape=(num_products,), dtype=np.float32)
        else:
            if self.discrete_actions:
                self.action_space = spaces.MultiDiscrete([self.n_action_bins] * self.num_products)
                # precompute mapping from bin index to order qty
                self._bin_to_qty = np.linspace(0, self.max_order, self.n_action_bins).astype(np.float32)
            else:
                self.action_space = spaces.Box(
                    low=0, 
                    high=max_order, 
                    shape=(num_products,), 
                    dtype=np.float32
                )
        
        # Observation space
        obs_dim = (
            num_products +  # inventory levels
            1 +             # day of week (normalized)
            1 +             # week of year (normalized)
            num_products +  # recent sales average
            num_products    # days since stockout per product
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_stockouts = []
        self.episode_sales = []
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize inventory at random levels
        self.inventory = np.random.randint(
            self.demand_mean, 
            self.max_inventory // 2, 
            size=self.num_products
        ).astype(np.float32)
        
        self.day = 0
        self.week = 1
        self.sales_history = [deque(maxlen=7) for _ in range(self.num_products)]
        for hist in self.sales_history:
            hist.extend([self.demand_mean] * 7)
        
        self.days_since_stockout = np.zeros(self.num_products, dtype=np.float32)
        self.total_revenue = 0
        self.total_cost = 0
        self.stockout_count = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Construct observation vector"""
        # Normalize inventory
        inv_norm = self.inventory / self.max_inventory
        
        # Normalize time features
        day_norm = (self.day % 7) / 7.0
        week_norm = (self.week % 52) / 52.0
        
        # Calculate recent sales average
        sales_avg = np.array([
            np.mean(hist) if len(hist) > 0 else self.demand_mean 
            for hist in self.sales_history
        ]) / self.demand_mean
        
        # Normalize days since stockout
        days_stockout_norm = np.clip(self.days_since_stockout / 30.0, 0, 1)
        
        obs = np.concatenate([
            inv_norm,
            [day_norm],
            [week_norm],
            sales_avg,
            days_stockout_norm
        ]).astype(np.float32)
        
        return obs
    
    def _generate_demand(self):
        """Generate demand with seasonality and day-of-week patterns"""
        demands = []
        
        for product_idx in range(self.num_products):
            # Base demand
            base_demand = self.demand_mean
            
            # Seasonal variation (sinusoidal)
            seasonal_factor = 1 + self.seasonal_amplitude * np.sin(
                2 * np.pi * self.week / 52.0
            )
            
            # Day of week variation (weekend boost)
            dow = self.day % 7
            if dow in [5, 6]:  # Weekend
                dow_factor = 1.2
            elif dow == 0:  # Monday
                dow_factor = 0.9
            else:
                dow_factor = 1.0
            
            # Random variation
            demand = base_demand * seasonal_factor * dow_factor
            demand = max(0, np.random.normal(demand, self.demand_std))
            
            demands.append(demand)
        
        return np.array(demands)
    
    def step(self, action):
        """Execute one time step"""
        # Normalize incoming action to numpy array for indexing
        action = np.array(action)

        # If using order-up-to semantics, interpret action as a target inventory level
        if self.order_up_to:
            if self.discrete_actions:
                # action contains bin indices -> map to target inventory (units)
                action = self._bin_to_target[action.astype(int)]
            else:
                # continuous action assumed to be fraction 0..1 -> scale to inventory units
                action = np.clip(action, 0.0, 1.0) * float(self.max_inventory)

            # Compute order quantities needed to reach target inventory
            order_qty = np.maximum(0.0, action - self.inventory)
            # Threshold tiny quantities and round to integers
            order_qty = np.where(order_qty >= self.order_threshold, order_qty, 0.0)
            order_qty = np.round(order_qty).astype(np.float32)
            action = order_qty
        else:
            # Regular order-quantity semantics
            if self.discrete_actions:
                action = self._bin_to_qty[action.astype(int)]
            action = np.clip(action, 0, self.max_order)
            # Treat tiny fractional actions as no-op (policy may output tiny non-zero values)
            action = np.where(action >= self.order_threshold, action, 0.0)
            # Work with integer order quantities (units)
            action = np.round(action).astype(np.float32)
        
        # Calculate ordering costs
        # Charge fixed ordering cost once per time-step (one transaction for all products)
        order_cost = 0.0
        total_order_qty = float(np.sum(action))
        if total_order_qty > 0.0:
            order_cost = self.order_cost_fixed + total_order_qty * self.order_cost_per_unit
        
        # Receive order (immediate delivery for simplicity)
        self.inventory = np.clip(
            self.inventory + action, 
            0, 
            self.max_inventory
        )
        
        # Generate demand
        demand = self._generate_demand()
        
        # Process sales
        sales = np.minimum(demand, self.inventory)
        self.inventory -= sales

        # Calculate stockouts (in units)
        stockout = demand - sales
        stockout_units = np.sum(stockout)

        # Update metrics: count units of unmet demand instead of boolean per-product flags
        self.stockout_count += float(stockout_units)

        # Update days since stockout per product
        stockout_occurred = stockout > 0
        self.days_since_stockout = np.where(
            stockout_occurred,
            0,
            self.days_since_stockout + 1
        )
        
        # Update sales history
        for i, sale in enumerate(sales):
            self.sales_history[i].append(sale)
        
        # Calculate revenue
        revenue = np.sum(sales) * self.selling_price
        self.total_revenue += revenue
        
        # Calculate costs
        holding_cost = np.sum(self.inventory) * self.holding_cost_per_unit
        stockout_cost = stockout_units * self.stockout_penalty
        
        total_cost = order_cost + holding_cost + stockout_cost
        self.total_cost += total_cost
        
        # Reward = profit
        reward = revenue - total_cost
        # Scale reward for RL stability
        scaled_reward = float(reward) / float(self.reward_scale)
        
        # Update time
        self.day += 1
        self.week = (self.day // 7) + 1
        
        # Episode termination (simulate one year)
        terminated = self.day >= 365
        truncated = False
        
        if terminated:
            # store unscaled episode profit for reporting
            self.episode_rewards.append(self.total_revenue - self.total_cost)
            self.episode_stockouts.append(self.stockout_count)
            self.episode_sales.append(self.total_revenue / self.selling_price)
        
        return self._get_obs(), scaled_reward, terminated, truncated, {
            'revenue': revenue,
            'costs': total_cost,
            'stockouts': float(stockout_units),
            'inventory_level': float(np.mean(self.inventory))
        }
    
    def render(self):
        """Optional: visualize current state"""
        print(f"Day {self.day}, Week {self.week}")
        print(f"Inventory: {self.inventory}")
        print(f"Total Revenue: ${self.total_revenue:.2f}")
        print(f"Total Cost: ${self.total_cost:.2f}")
        print(f"Net Profit: ${self.total_revenue - self.total_cost:.2f}")
        print(f"Stockouts: {self.stockout_count}")
        print("-" * 50)







