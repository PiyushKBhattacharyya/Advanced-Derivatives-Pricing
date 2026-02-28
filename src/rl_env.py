import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FrictionalHedgingEnv(gym.Env):
    """
    OpenAI Gymnasium Environment modeling a live trading desk's Delta-Hedging pipeline.
    Unlike Black-Scholes, this environment explicitly penalizes the Agent for crossing
    the Bid/Ask spread (Transaction Costs). The AI must learn to balance Delta-Risk 
    against Capital destruction.
    """
    def __init__(self, spx_trajectories, bsde_delta_trajectories, transaction_cost=0.0002):
        super(FrictionalHedgingEnv, self).__init__()
        
        # [N_paths, T_steps] Matrices
        self.spx_paths = spx_trajectories
        self.bsde_deltas = bsde_delta_trajectories
        
        # Tier-1 Institutional Slippage (e.g., 2 basis points)
        self.transaction_cost = transaction_cost
        
        self.n_paths, self.n_steps = self.spx_paths.shape
        
        # ACTION: The Agent chooses a continuous portfolio delta holding between [0.0, 1.0]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # OBSERVATION: [Time_Remaining, Current_Spot, Deep_BSDE_Delta, Current_Inventory]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.current_step = 0
        self.current_path = 0
        self.inventory = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_path = np.random.randint(0, self.n_paths)
        self.inventory = 0.0
        return self._get_obs(), {}
        
    def _get_obs(self):
        t_rem = 1.0 - (self.current_step / float(self.n_steps))
        S_current = self.spx_paths[self.current_path, self.current_step]
        S_start = self.spx_paths[self.current_path, 0]
        S_norm = S_current / S_start
        delta_nn = self.bsde_deltas[self.current_path, self.current_step]
        return np.array([t_rem, S_norm, delta_nn, self.inventory], dtype=np.float32)
        
    def step(self, action):
        target_inventory = np.clip(action[0], 0.0, 1.0)
        
        S_current = self.spx_paths[self.current_path, self.current_step]
        S_next = self.spx_paths[self.current_path, min(self.current_step + 1, self.n_steps - 1)]
        
        # 1. Trading Friction (Cost of physically rebalancing) — normalized by spot
        trade_size = np.abs(target_inventory - self.inventory)
        friction_cost = trade_size * self.transaction_cost  # ~0.00002 per unit trade
        
        # 2. Portfolio PnL — normalized by spot so return is unit-free (~[-0.02, +0.02]/day)
        daily_return = (S_next - S_current) / S_current
        pnl = target_inventory * daily_return
        
        # 3. Base reward
        step_reward = pnl - friction_cost
        
        # 4. Risk Penalty — now on the same scale as normalized PnL
        optimal_delta = self.bsde_deltas[self.current_path, self.current_step]
        deviation = np.abs(target_inventory - optimal_delta)
        risk_penalty = 1.0 * deviation  # same unit as pnl now (~0.0–0.3)
        
        # 5. Convergence bonus: reward the agent extra for being very close to target
        convergence_bonus = 0.005 if deviation < 0.02 else 0.0
        
        total_reward = step_reward - risk_penalty + convergence_bonus
        
        self.inventory = target_inventory
        self.current_step += 1
        
        done = self.current_step >= self.n_steps - 1
        truncated = False
        
        return self._get_obs(), float(total_reward), done, truncated, {}
