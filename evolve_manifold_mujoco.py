"""
Evolve Neuro-Manifold Hive Mind (Swarm Intelligence)

This script simulates a population of AI agents evolving via Swarm Intelligence.
Key Mechanism: "Instant Knowledge Transfer"
- Multiple agents explore in parallel.
- The best experience/brain is instantly broadcast to others.
- Divergent thinking is maintained via mutation.
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import copy
from collections import deque
from neuro_manifold.agent import ManifoldAgent

def train_hive_mind():
    print("=" * 80)
    print("Neuro-Manifold 'Hive Mind' Evolution - Swarm Intelligence")
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
        return

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 1. Initialize Swarm Population
    POP_SIZE = 4 # Small swarm for demo (conceptually 1M)
    population = [ManifoldAgent(obs_dim, action_dim, num_micro=8, num_macro=2).to(device) for _ in range(POP_SIZE)]

    # Shared Optimizer
    optimizer = torch.optim.AdamW(population[0].parameters(), lr=3e-4, weight_decay=1e-4)

    print(f"Swarm Initialized: {POP_SIZE} Agents")

    # Evolution Loop
    n_generations = 5
    steps_per_gen = 512
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    ent_coef = 0.01
    mutation_rate = 0.02

    history = {'returns': [], 'elite_return': []}

    for gen in range(1, n_generations + 1):

        # --- CURRICULUM CONTROL (The Edge of Chaos Strategy) ---
        if gen <= 1:
            # Phase 1: Stability (Euclidean Space, No Plasticity)
            print(f"Gen {gen} [Phase 1]: Warm-up (Frozen Geometry & Plasticity)")
            for agent in population:
                agent.brain.geometry.freeze()
                agent.brain.micro_layer.eta.data.fill_(0.0) # Disable plasticity
                agent.brain.macro_layer.eta.data.fill_(0.0)
        elif gen <= 3:
            # Phase 2: Flexibility (Enable Geometry, No Plasticity)
            print(f"Gen {gen} [Phase 2]: Geometry Unlocked")
            for agent in population:
                agent.brain.geometry.unfreeze()
                # Plasticity still off
        else:
            # Phase 3: Full Awakening (Enable Plasticity)
            print(f"Gen {gen} [Phase 3]: Full Edge of Chaos")
            for agent in population:
                agent.brain.geometry.unfreeze()
                # Enable plasticity (Set to small initial value)
                if agent.brain.micro_layer.eta.item() == 0.0:
                    agent.brain.micro_layer.eta.data.fill_(0.01)
                    agent.brain.macro_layer.eta.data.fill_(0.01)

        # --- PHASE 1: Parallel Exploration (Swarm Rollout) ---
        swarm_experience = []
        swarm_returns = []

        for i, agent in enumerate(population):
            if i > 0:
                agent.mutate(noise_scale=0.005)

            obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
            state_buf = []

            obs, _ = env.reset(seed=seed + gen*100 + i)
            agent.reset()
            ep_ret = 0

            agent.eval()
            for t in range(steps_per_gen):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

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

                obs_buf.append(obs)
                act_buf.append(act_np)
                rew_buf.append(reward)
                val_buf.append(val_np)
                logp_buf.append(logp_np)
                done_buf.append(done)

                obs = next_obs
                ep_ret += reward

                if done:
                    obs, _ = env.reset()
                    agent.reset()

            # Finish Path (GAE)
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

            swarm_experience.append({
                'obs': torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device),
                'act': torch.as_tensor(np.array(act_buf), dtype=torch.float32, device=device),
                'logp': torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=device),
                'adv': torch.as_tensor(adv_buf, dtype=torch.float32, device=device),
                'ret': torch.as_tensor(ret_buf, dtype=torch.float32, device=device),
                'state_buf': state_buf
            })
            swarm_returns.append(ep_ret)

        # --- PHASE 2: Hive Mind Broadcast ---
        elite_idx = np.argmax(swarm_returns)
        elite_return = swarm_returns[elite_idx]
        elite_agent = population[elite_idx]

        print(f"Gen {gen} | Elite: Agent {elite_idx} (Return: {elite_return:.2f}) | Swarm Avg: {np.mean(swarm_returns):.2f}")
        history['returns'].append(swarm_returns)
        history['elite_return'].append(elite_return)

        if elite_idx != 0:
            population[0].load_brain_from(elite_agent)

        for i in range(1, POP_SIZE):
            population[i].load_brain_from(population[0])
            population[i].mutate(noise_scale=mutation_rate)

        # --- PHASE 3: Collective Learning ---
        obs_all = torch.cat([e['obs'] for e in swarm_experience])
        act_all = torch.cat([e['act'] for e in swarm_experience])
        logp_all = torch.cat([e['logp'] for e in swarm_experience])
        adv_all = torch.cat([e['adv'] for e in swarm_experience])
        ret_all = torch.cat([e['ret'] for e in swarm_experience])

        all_state_buf = [s for e in swarm_experience for s in e['state_buf']]

        def collate_states(states_list):
            batch = {}
            for k in states_list[0].keys():
                vals = [s[k] for s in states_list]
                if vals[0] is not None:
                    batch[k] = torch.cat(vals, dim=0)
                else:
                    batch[k] = None
            return batch

        state_batch = collate_states(all_state_buf)

        adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

        agent = population[0]
        agent.train()

        for i in range(10):
            optimizer.zero_grad()

            mean, logstd, values, prediction, flat_state, energy = agent(obs_all, initial_state=state_batch, mode='train')

            # Loss
            pred_current = prediction[:-1]
            target_next = flat_state[1:].detach()
            if pred_current.shape[0] != target_next.shape[0]:
                 min_len = min(pred_current.shape[0], target_next.shape[0])
                 pred_current = pred_current[:min_len]
                 target_next = target_next[:min_len]

            loss_pred = F.mse_loss(pred_current, target_next)

            std = torch.exp(logstd)
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(act_all).sum(axis=-1)
            ratio = torch.exp(logp - logp_all)

            surr1 = ratio * adv_all
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_all
            loss_pi = -torch.min(surr1, surr2).mean()

            loss_v = F.mse_loss(values.squeeze(-1), ret_all)
            loss_ent = -dist.entropy().mean()

            loss = loss_pi + 0.5 * loss_v + 0.5 * loss_pred + ent_coef * loss_ent

            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

        print(f"      Loss: {loss.item():.4f} | Pred: {loss_pred.item():.4f}")

    with open("metrics_hive_mind.json", "w") as f:
        json_history = {k: [np.array(x).tolist() for x in v] for k, v in history.items()}
        json.dump(json_history, f)

if __name__ == "__main__":
    train_hive_mind()
