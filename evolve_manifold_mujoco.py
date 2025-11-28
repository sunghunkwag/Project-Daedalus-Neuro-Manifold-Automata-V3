"""
Evolve Neuro-Manifold in MuJoCo (Advanced)

This script runs the evolutionary/RL loop for the Neuro-Manifold Automata V2.
Includes Intrinsic Motivation (Curiosity) and improved PPO stability.
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from collections import deque

from neuro_manifold.agent import ManifoldAgent

def train_manifold():
    print("=" * 80)
    print("Neuro-Manifold Automata V2 - Advanced Evolution in HalfCheetah-v4")
    print("=" * 80)

    env_id = "HalfCheetah-v4"
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Environment
    try:
        env = gym.make(env_id)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Please ensure gymnasium[mujoco] is installed.")
        return

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Organism (Using Hierarchical Micro/Macro structure)
    agent = ManifoldAgent(obs_dim, action_dim, num_micro=8, num_macro=2).to(device)

    # Reduced LR for stability
    optimizer = torch.optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=1e-4)

    print(agent)

    # Evolution Loop
    n_generations = 5
    steps_per_gen = 512
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    ent_coef = 0.01

    history = {'returns': [], 'curiosity': []}

    for gen in range(1, n_generations + 1):
        # 1. Gather Experience (Life)
        obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
        state_buf = [] # Store agent internal state at each step

        obs, _ = env.reset(seed=seed + gen)
        agent.reset()

        ep_ret = 0
        ep_len = 0

        agent.eval()
        for t in range(steps_per_gen):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Store State (Teacher Forcing Context)
            current_state = agent.get_state()
            state_buf.append(current_state)

            with torch.no_grad():
                mean, logstd, val = agent(obs_t, mode='act')
                std = torch.exp(logstd)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                logp = dist.log_prob(action).sum(axis=-1)

            act_np = action.cpu().numpy()[0]
            val_np = val.item()
            logp_np = logp.item()

            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            done = terminated or truncated

            # Store
            obs_buf.append(obs)
            act_buf.append(act_np)
            rew_buf.append(reward)
            val_buf.append(val_np)
            logp_buf.append(logp_np)
            done_buf.append(done)

            obs = next_obs
            ep_ret += reward
            ep_len += 1

            if done or (t == steps_per_gen - 1):
                if done:
                    print(f"Gen {gen} | Episode Return: {ep_ret:.2f}")
                    history['returns'].append(ep_ret)

                obs, _ = env.reset()
                agent.reset()
                ep_ret = 0
                ep_len = 0

        # 2. Compute Advantages (GAE)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, last_val = agent(obs_t, mode='act')
            last_val = last_val.item()

        adv_buf = np.zeros_like(rew_buf, dtype=np.float32)
        last_gae = 0

        for t in reversed(range(len(rew_buf))):
            if t == len(rew_buf) - 1:
                next_non_terminal = 1.0 - float(done_buf[t])
                next_val = last_val
            else:
                next_non_terminal = 1.0 - float(done_buf[t])
                next_val = val_buf[t+1]

            delta = rew_buf[t] + gamma * next_val * next_non_terminal - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            adv_buf[t] = last_gae

        ret_buf = adv_buf + np.array(val_buf)

        # 3. Optimize (Mutate/Update)
        def collate_states(states_list):
            batch = {}
            for k in states_list[0].keys():
                vals = [s[k] for s in states_list]
                if vals[0] is not None:
                    batch[k] = torch.cat(vals, dim=0)
                else:
                    batch[k] = None
            return batch

        obs_tens = torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_tens = torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=device)
        logp_tens = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=device)
        adv_tens = torch.as_tensor(adv_buf, dtype=torch.float32, device=device)
        ret_tens = torch.as_tensor(ret_buf, dtype=torch.float32, device=device)

        state_batch = collate_states(state_buf)

        # Normalize adv
        adv_tens = (adv_tens - adv_tens.mean()) / (adv_tens.std() + 1e-8)

        agent.train()
        for i in range(10): # Epochs
            optimizer.zero_grad()

            # Forward with explicit state
            mean, logstd, values, prediction, flat_state, energy = agent(obs_tens, initial_state=state_batch, mode='train')

            # Prediction Loss
            pred_current = prediction[:-1]
            target_next = flat_state[1:].detach()
            loss_pred = F.mse_loss(pred_current, target_next)

            # PPO Loss
            std = torch.exp(logstd)
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(act_tens).sum(axis=-1)
            ratio = torch.exp(logp - logp_tens)

            surr1 = ratio * adv_tens
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_tens
            loss_pi = -torch.min(surr1, surr2).mean()

            loss_v = F.mse_loss(values.squeeze(-1), ret_tens)

            # Entropy Loss
            loss_ent = -dist.entropy().mean()

            # Total Loss
            loss = loss_pi + 0.5 * loss_v + 0.5 * loss_pred + ent_coef * loss_ent

            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

        print(f"Gen {gen} | Loss: {loss.item():.4f} | Pred: {loss_pred.item():.4f} | E: {energy.mean().item():.4f}")

    # Save results
    with open("metrics_manifold_evolution.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    train_manifold()
