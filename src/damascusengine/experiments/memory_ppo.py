from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
import functools
from pathlib import Path

import gymnasium
from gymnasium import spaces
import matplotlib
os.environ["MPLCONFIGDIR"] = str(Path(".matplotlib").resolve())
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pufferlib
import pufferlib.emulation
import pufferlib.vector
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


@dataclass
class Config:
    total_timesteps: int = 100_000
    num_envs: int = 32
    num_workers: int = 8
    rollout_steps: int = 64
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_envs: int = 8
    hidden_size: int = 128
    seed: int = 7
    backend: str = "multiprocessing"
    agent: str = "ff"
    cue_delay: int = 8
    output_dir: str = "results/memory"


class CueRecallEnv(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cue_delay: int = 8, render_mode: str = "rgb_array") -> None:
        super().__init__()
        self.cue_delay = cue_delay
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._rng = np.random.default_rng()
        self.cue = 0
        self.step_index = 0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.cue = int(self._rng.integers(0, 2))
        self.step_index = 0
        return self._observation(), {}

    def _observation(self) -> np.ndarray:
        if self.step_index == 0:
            return np.array(
                [1.0 if self.cue == 0 else 0.0, 1.0 if self.cue == 1 else 0.0, 0.0],
                dtype=np.float32,
            )
        if self.step_index <= self.cue_delay:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def step(self, action: int):
        terminated = False
        reward = 0.0
        info = {}
        if self.step_index > self.cue_delay:
            terminated = True
            reward = 1.0 if int(action) == self.cue else 0.0
            info["success"] = reward
        self.step_index += 1
        return self._observation(), reward, terminated, False, info

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)


class FeedforwardPolicy(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def act(self, obs: torch.Tensor, hidden: torch.Tensor | None = None):
        h = self.net(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1), hidden

    def evaluate_sequence(self, obs_seq: torch.Tensor, done_seq: torch.Tensor):
        t, n, obs_size = obs_seq.shape
        flat = obs_seq.reshape(t * n, obs_size)
        h = self.net(flat)
        logits = self.policy_head(h).reshape(t, n, -1)
        values = self.value_head(h).reshape(t, n)
        return logits, values


class RecurrentPolicy(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.Tanh())
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def act(self, obs: torch.Tensor, hidden: torch.Tensor | None):
        if hidden is None:
            hidden = self.initial_state(obs.shape[0], obs.device)
        x = self.encoder(obs).unsqueeze(0)
        out, hidden = self.rnn(x, hidden)
        out = out.squeeze(0)
        return self.policy_head(out), self.value_head(out).squeeze(-1), hidden

    def evaluate_sequence(self, obs_seq: torch.Tensor, done_seq: torch.Tensor):
        steps, batch, _ = obs_seq.shape
        hidden = self.initial_state(batch, obs_seq.device)
        logits = []
        values = []
        prev_done = torch.zeros(batch, device=obs_seq.device)
        for step in range(steps):
            mask = (1.0 - prev_done).view(1, batch, 1)
            hidden = hidden * mask
            x = self.encoder(obs_seq[step]).unsqueeze(0)
            out, hidden = self.rnn(x, hidden)
            out = out.squeeze(0)
            logits.append(self.policy_head(out))
            values.append(self.value_head(out).squeeze(-1))
            prev_done = done_seq[step]
        return torch.stack(logits), torch.stack(values)


def make_memory_env(buf=None, seed: int = 0, cue_delay: int = 8):
    env = CueRecallEnv(cue_delay=cue_delay)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, seed=seed)


def resolve_backend(name: str):
    if name == "serial":
        return pufferlib.vector.Serial
    if name == "multiprocessing":
        return pufferlib.vector.Multiprocessing
    raise ValueError(f"unsupported backend: {name}")


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_output_dir(root: str, agent: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(root) / agent / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_gae(rewards, dones, values, next_value, gamma, gae_lambda):
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(next_value)
    for step in reversed(range(steps)):
        if step == steps - 1:
            next_non_terminal = 1.0 - dones[step]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_values = values[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def collect_episode_stats(infos, episode_returns, episode_lengths, successes):
    if not isinstance(infos, list):
        return
    for info in infos:
        if not isinstance(info, dict):
            continue
        if "episode_return" in info:
            episode_returns.append(float(info["episode_return"]))
        if "episode_length" in info:
            episode_lengths.append(int(info["episode_length"]))
        if "success" in info:
            successes.append(float(info["success"]))


def save_plots(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return
    updates = [row["update"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(updates, [row["mean_return_50"] for row in rows], color="#1f77b4")
    axes[0, 0].set_title("Mean Return (50-episode window)")

    axes[0, 1].plot(updates, [row["success_rate_50"] for row in rows], color="#d62728")
    axes[0, 1].set_title("Success Rate (50-episode window)")

    axes[1, 0].plot(updates, [row["policy_loss"] for row in rows], label="policy")
    axes[1, 0].plot(updates, [row["value_loss"] for row in rows], label="value")
    axes[1, 0].plot(updates, [row["entropy"] for row in rows], label="entropy")
    axes[1, 0].legend()
    axes[1, 0].set_title("Optimization Signals")

    axes[1, 1].plot(updates, [row["sps"] for row in rows], color="#2ca02c")
    axes[1, 1].set_title("Steps Per Second")

    for ax in axes.flat:
        ax.set_xlabel("Update")

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def build_policy(config: Config, obs_size: int, action_count: int):
    if config.agent == "ff":
        return FeedforwardPolicy(obs_size, action_count, config.hidden_size)
    if config.agent == "gru":
        return RecurrentPolicy(obs_size, action_count, config.hidden_size)
    raise ValueError(f"unsupported agent: {config.agent}")


def run_experiment(config: Config) -> dict:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = select_device()
    output_dir = make_output_dir(config.output_dir, config.agent)

    backend = resolve_backend(config.backend)
    vec_kwargs = {}
    if backend is pufferlib.vector.Multiprocessing:
        vec_kwargs["num_workers"] = config.num_workers

    env_factory = functools.partial(make_memory_env, cue_delay=config.cue_delay)
    vecenv = pufferlib.vector.make(env_factory, backend=backend, num_envs=config.num_envs, seed=config.seed, **vec_kwargs)

    obs, _ = vecenv.reset(seed=config.seed)
    obs_size = int(np.prod(vecenv.single_observation_space.shape))
    action_count = int(vecenv.single_action_space.n)
    policy = build_policy(config, obs_size, action_count).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    batch_size = config.num_envs * config.rollout_steps
    updates = config.total_timesteps // batch_size

    obs_buffer = torch.zeros((config.rollout_steps, config.num_envs, obs_size), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.int64, device=device)
    logprob_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    rewards_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    dones_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    values_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)

    metrics_path = output_dir / "metrics.csv"
    rows = []
    recent_returns: deque[float] = deque(maxlen=50)
    recent_lengths: deque[int] = deque(maxlen=50)
    recent_successes: deque[float] = deque(maxlen=50)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    successes: list[float] = []
    processed_return_count = 0
    processed_length_count = 0
    processed_success_count = 0
    hidden = policy.initial_state(config.num_envs, device) if isinstance(policy, RecurrentPolicy) else None
    global_step = 0
    start_time = time.time()

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "update",
                "global_step",
                "mean_return_50",
                "mean_length_50",
                "success_rate_50",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "sps",
            ],
        )
        writer.writeheader()

        for update in range(1, updates + 1):
            for step in range(config.rollout_steps):
                global_step += config.num_envs
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                obs_buffer[step] = obs_tensor

                with torch.no_grad():
                    logits, value, hidden = policy.act(obs_tensor, hidden)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action)

                actions_buffer[step] = action
                logprob_buffer[step] = logprob
                values_buffer[step] = value

                next_obs, rewards, terminals, truncations, infos = vecenv.step(action.cpu().numpy())
                done = np.logical_or(terminals, truncations)
                done_tensor = torch.as_tensor(done, dtype=torch.float32, device=device)
                rewards_buffer[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                dones_buffer[step] = done_tensor

                if hidden is not None:
                    hidden = hidden * (1.0 - done_tensor).view(1, config.num_envs, 1)

                collect_episode_stats(infos, episode_returns, episode_lengths, successes)
                for value_item in episode_returns[processed_return_count:]:
                    recent_returns.append(value_item)
                processed_return_count = len(episode_returns)
                for length_item in episode_lengths[processed_length_count:]:
                    recent_lengths.append(length_item)
                processed_length_count = len(episode_lengths)
                for success in successes[processed_success_count:]:
                    recent_successes.append(success)
                processed_success_count = len(successes)

                obs = next_obs

            with torch.no_grad():
                next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                next_value = policy.act(next_obs_tensor, hidden)[1]

            advantages, returns = compute_gae(
                rewards_buffer, dones_buffer, values_buffer, next_value, config.gamma, config.gae_lambda
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            clip_fractions = []
            policy_loss_value = 0.0
            value_loss_value = 0.0
            entropy_value = 0.0
            approx_kl_value = 0.0

            env_indices = np.arange(config.num_envs)
            for _ in range(config.update_epochs):
                np.random.shuffle(env_indices)
                for start in range(0, config.num_envs, config.minibatch_envs):
                    mb_env = env_indices[start:start + config.minibatch_envs]
                    mb_obs = obs_buffer[:, mb_env]
                    mb_actions = actions_buffer[:, mb_env]
                    mb_logprob = logprob_buffer[:, mb_env]
                    mb_advantages = advantages[:, mb_env]
                    mb_returns = returns[:, mb_env]
                    mb_dones = dones_buffer[:, mb_env]

                    logits, new_values = policy.evaluate_sequence(mb_obs, mb_dones)
                    dist = Categorical(logits=logits.reshape(-1, action_count))
                    new_logprob = dist.log_prob(mb_actions.reshape(-1)).reshape_as(mb_actions)
                    entropy = dist.entropy().mean()

                    logratio = new_logprob - mb_logprob
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                        clip_fractions.append(clip_fraction)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()
                    value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()
                    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

                    policy_loss_value = float(policy_loss.item())
                    value_loss_value = float(value_loss.item())
                    entropy_value = float(entropy.item())
                    approx_kl_value = float(approx_kl.item())

            row = {
                "update": update,
                "global_step": global_step,
                "mean_return_50": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "mean_length_50": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
                "success_rate_50": float(np.mean(recent_successes)) if recent_successes else 0.0,
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approx_kl": approx_kl_value,
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "sps": int(global_step / max(time.time() - start_time, 1e-6)),
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(
                f"agent={config.agent} update={update:04d} step={global_step:07d} "
                f"return50={row['mean_return_50']:.3f} success50={row['success_rate_50']:.3f} sps={row['sps']}"
            )

    vecenv.close()
    save_plots(rows, output_dir)

    summary = {
        "config": asdict(config),
        "device": str(device),
        "final_mean_return_50": rows[-1]["mean_return_50"] if rows else 0.0,
        "best_mean_return_50": max((row["mean_return_50"] for row in rows), default=0.0),
        "final_success_rate_50": rows[-1]["success_rate_50"] if rows else 0.0,
        "best_success_rate_50": max((row["success_rate_50"] for row in rows), default=0.0),
        "episodes_completed": len(episode_returns),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run a PufferLib-backed cue-recall memory PPO experiment.")
    parser.add_argument("--agent", choices=["ff", "gru"], default="ff")
    parser.add_argument("--backend", choices=["serial", "multiprocessing"], default="multiprocessing")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-envs", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--cue-delay", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results/memory")
    args = parser.parse_args()
    return Config(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_envs=args.minibatch_envs,
        hidden_size=args.hidden_size,
        cue_delay=args.cue_delay,
        seed=args.seed,
        backend=args.backend,
        agent=args.agent,
        output_dir=args.output_dir,
    )


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    config = parse_args()
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
