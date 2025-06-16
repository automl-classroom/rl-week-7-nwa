"""
Deep Q-Learning with RND implementation.
"""
import matplotlib.pyplot as plt
import torch
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_4.networks import QNetwork


class RNDDQNAgent(DQNAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        rnd_hidden_size: int = 128,
        rnd_lr: float = 1e-3,
        rnd_update_freq: int = 1000,
        rnd_n_layers: int = 2,
        rnd_reward_weight: float = 0.1,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.seed = seed
        # TODO: initialize the RND networks
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        self.rnd_update_freq = target_update_freq
        self.target_ntwrk = QNetwork(obs_dim, n_actions, rnd_hidden_size)
        self.pred_ntwrk = QNetwork(obs_dim, n_actions, rnd_hidden_size)
        self.rnd_optimizer = torch.optim.Adam(self.pred_ntwrk.parameters(), lr=lr)   
        
        self.rnd_reward_weight = rnd_reward_weight

    def update_rnd(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on the RND network on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).
        """
        # TODO: get states and next_states from the batch
        states = np.array([t[0] for t in training_batch])
        states_tensor = torch.tensor(states, dtype=torch.float32)

        next_states = np.array([t[3] for t in training_batch])
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        
        with torch.no_grad():
            target_output = self.target_ntwrk(next_states_tensor)

        
        predictor_output = self.pred_ntwrk(next_states_tensor)
        loss = torch.nn.functional.mse_loss(predictor_output, target_output)

        self.rnd_optimizer.zero_grad()
        loss.backward()
        self.rnd_optimizer.step()
        
        return loss.item()
        
    def get_rnd_bonus(self, state: np.ndarray) -> float:
        """Compute the RND bonus for a given state.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.

        Returns
        -------
        float
            The RND bonus for the state.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            target_output = self.target_ntwrk(state_tensor)
            predictor_output = self.pred_ntwrk(state_tensor)

        bonus = torch.nn.functional.mse_loss(predictor_output, target_output).item()
        return bonus

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        episode_rewards = []
        steps = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # TODO: apply RND bonus
            reward += self.rnd_reward_weight * self.get_rnd_bonus(next_state)
            
            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

                if self.total_steps % self.rnd_update_freq == 0:
                    self.update_rnd(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                episode_rewards.append(ep_reward)
                steps.append(frame)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards)
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        # Saving to .csv for simplicity
        # Could also be e.g. npz
        print("Training complete.")
        training_data = pd.DataFrame({"steps": steps, "rewards": episode_rewards})
        training_data.to_csv(f"training_data_seed_{self.seed}.csv", index=False)


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 3) TODO: instantiate & train the agent
    agent = RNDDQNAgent(env=env)
    agent.train(num_frames=20000)
 
if __name__ == "__main__":
    main()
