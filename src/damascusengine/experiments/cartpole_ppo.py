from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
import matplotlib
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
    total_timesteps: int = 200_000
    num_envs: int = 32
    num_workers: int = 8
    rollout_steps: int = 128
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 128
    seed: int = 7
    backend: str = "multiprocessing"
    output_dir: str = "results/cartpole"


class ActorCritic(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


def make_cartpole(buf=None, seed: int = 0):
    env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
    env = gymnasium.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -1, 1))
    env = gymnasium.wrappers.TransformReward(env, lambda reward: np.clip(reward, -1, 1))
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


def make_output_dir(root: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(root) / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_episode_stats(infos, episode_returns: list[float], episode_lengths: list[int]) -> None:
    if not isinstance(infos, list):
        return
    for info in infos:
        if not isinstance(info, dict):
            continue
        if "episode_return" in info:
            episode_returns.append(float(info["episode_return"]))
        if "episode_length" in info:
            episode_lengths.append(int(info["episode_length"]))


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def save_plots(rows: list[dict], output_dir: Path) -> None:
    if not rows:
        return

    update = [row["update"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(update, [row["mean_return_20"] for row in rows], color="#1f77b4")
    axes[0, 0].set_title("Mean Episode Return (20-episode window)")
    axes[0, 0].set_xlabel("Update")
    axes[0, 0].set_ylabel("Return")

    axes[0, 1].plot(update, [row["mean_length_20"] for row in rows], color="#ff7f0e")
    axes[0, 1].set_title("Mean Episode Length (20-episode window)")
    axes[0, 1].set_xlabel("Update")
    axes[0, 1].set_ylabel("Length")

    axes[1, 0].plot(update, [row["policy_loss"] for row in rows], label="policy")
    axes[1, 0].plot(update, [row["value_loss"] for row in rows], label="value")
    axes[1, 0].plot(update, [row["entropy"] for row in rows], label="entropy")
    axes[1, 0].set_title("Optimization Signals")
    axes[1, 0].set_xlabel("Update")
    axes[1, 0].legend()

    axes[1, 1].plot(update, [row["sps"] for row in rows], color="#2ca02c")
    axes[1, 1].set_title("Steps Per Second")
    axes[1, 1].set_xlabel("Update")
    axes[1, 1].set_ylabel("SPS")

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def run_experiment(config: Config) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = select_device()
    output_dir = make_output_dir(config.output_dir)

    backend = resolve_backend(config.backend)
    vec_kwargs = {}
    if backend is pufferlib.vector.Multiprocessing:
        vec_kwargs["num_workers"] = config.num_workers

    vecenv = pufferlib.vector.make(
        make_cartpole,
        backend=backend,
        num_envs=config.num_envs,
        seed=config.seed,
        **vec_kwargs,
    )

    obs, _ = vecenv.reset(seed=config.seed)
    obs_size = int(np.prod(vecenv.single_observation_space.shape))
    action_count = int(vecenv.single_action_space.n)

    model = ActorCritic(obs_size, action_count, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    batch_size = config.num_envs * config.rollout_steps
    updates = config.total_timesteps // batch_size

    obs_buffer = torch.zeros((config.rollout_steps, config.num_envs, obs_size), dtype=torch.float32, device=device)
    actions_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.int64, device=device)
    logprob_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    rewards_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    dones_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)
    values_buffer = torch.zeros((config.rollout_steps, config.num_envs), dtype=torch.float32, device=device)

    metrics_path = output_dir / "metrics.csv"
    rows: list[dict] = []
    recent_returns: deque[float] = deque(maxlen=20)
    recent_lengths: deque[int] = deque(maxlen=20)
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    processed_return_count = 0
    processed_length_count = 0
    global_step = 0
    start_time = time.time()

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "update",
                "global_step",
                "mean_return_20",
                "mean_length_20",
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
                    logits, value = model(obs_tensor)
                    distribution = Categorical(logits=logits)
                    action = distribution.sample()
                    logprob = distribution.log_prob(action)

                actions_buffer[step] = action
                logprob_buffer[step] = logprob
                values_buffer[step] = value

                next_obs, rewards, terminals, truncations, infos = vecenv.step(action.cpu().numpy())
                done = np.logical_or(terminals, truncations)
                rewards_buffer[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                dones_buffer[step] = torch.as_tensor(done, dtype=torch.float32, device=device)

                collect_episode_stats(infos, episode_returns, episode_lengths)
                for value_item in episode_returns[processed_return_count:]:
                    recent_returns.append(value_item)
                processed_return_count = len(episode_returns)
                for length_item in episode_lengths[processed_length_count:]:
                    recent_lengths.append(length_item)
                processed_length_count = len(episode_lengths)
                obs = next_obs

            with torch.no_grad():
                next_value = model(torch.as_tensor(obs, dtype=torch.float32, device=device))[1]

            advantages, returns = compute_gae(
                rewards_buffer,
                dones_buffer,
                values_buffer,
                next_value,
                config.gamma,
                config.gae_lambda,
            )

            b_obs = obs_buffer.reshape(batch_size, obs_size)
            b_actions = actions_buffer.reshape(batch_size)
            b_logprob = logprob_buffer.reshape(batch_size)
            b_advantages = advantages.reshape(batch_size)
            b_returns = returns.reshape(batch_size)
            b_values = values_buffer.reshape(batch_size)

            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            clip_fractions = []
            policy_loss_value = 0.0
            value_loss_value = 0.0
            entropy_value = 0.0
            approx_kl_value = 0.0

            indices = np.arange(batch_size)
            for _ in range(config.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, batch_size, config.minibatch_size):
                    end = start + config.minibatch_size
                    mb_idx = indices[start:end]

                    logits, new_value = model(b_obs[mb_idx])
                    dist = Categorical(logits=logits)
                    new_logprob = dist.log_prob(b_actions[mb_idx])
                    entropy = dist.entropy().mean()

                    logratio = new_logprob - b_logprob[mb_idx]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                        clip_fractions.append(clip_fraction)

                    mb_adv = b_advantages[mb_idx]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    value_loss = 0.5 * ((new_value - b_returns[mb_idx]) ** 2).mean()
                    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    policy_loss_value = float(policy_loss.item())
                    value_loss_value = float(value_loss.item())
                    entropy_value = float(entropy.item())
                    approx_kl_value = float(approx_kl.item())

            sps = int(global_step / max(time.time() - start_time, 1e-6))
            row = {
                "update": update,
                "global_step": global_step,
                "mean_return_20": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "mean_length_20": float(np.mean(recent_lengths)) if recent_lengths else 0.0,
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approx_kl": approx_kl_value,
                "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                "sps": sps,
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(
                f"update={update:04d} step={global_step:07d} "
                f"return20={row['mean_return_20']:.2f} len20={row['mean_length_20']:.2f} "
                f"sps={row['sps']}"
            )

    vecenv.close()
    save_plots(rows, output_dir)

    summary = {
        "config": asdict(config),
        "device": str(device),
        "final_mean_return_20": rows[-1]["mean_return_20"] if rows else 0.0,
        "final_mean_length_20": rows[-1]["mean_length_20"] if rows else 0.0,
        "best_mean_return_20": max((row["mean_return_20"] for row in rows), default=0.0),
        "best_mean_length_20": max((row["mean_length_20"] for row in rows), default=0.0),
        "episodes_completed": len(episode_returns),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Run a PufferLib-backed CartPole PPO experiment.")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", choices=["serial", "multiprocessing"], default="multiprocessing")
    parser.add_argument("--output-dir", default="results/cartpole")
    args = parser.parse_args()
    return Config(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        rollout_steps=args.rollout_steps,
        learning_rate=args.learning_rate,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size,
        seed=args.seed,
        backend=args.backend,
        output_dir=args.output_dir,
    )


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    config = parse_args()
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
