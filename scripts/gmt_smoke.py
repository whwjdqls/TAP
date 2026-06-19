"""
Headless smoke test for GMT (humanoid-general-motion-tracking).

Reuses sim2sim's `HumanoidEnv` but stubs `mujoco_viewer.MujocoViewer` so the
sim runs without an X display. Steps the policy for a short window and prints
the policy output / sim state. Use this when the cluster node has no GLFW
display (the upstream sim2sim --record_video flag still opens a GLFW context).

Usage:
    python scripts/gmt_smoke.py --motion walk_stand.pkl --steps 200
"""
from __future__ import annotations

import argparse
import os
import sys
import types

import numpy as np
import torch

TAP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GMT_ROOT = os.path.join(TAP_ROOT, "GMT")


class _NoopViewer:
    """Stand-in for mujoco_viewer.MujocoViewer that does nothing."""

    def __init__(self, *_args, **_kwargs):
        self.cam = types.SimpleNamespace(distance=0.0, lookat=np.zeros(3))

    def read_pixels(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def render(self):
        pass

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", default="walk_stand.pkl",
                        help="Motion pkl name under GMT/assets/motions/")
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of sim_dt steps to run.")
    parser.add_argument("--device", default="cpu",
                        help="cpu (default) or cuda. GMT's pinned torch 2.4.1 "
                             "doesn't support Blackwell sm_120; CPU is safe.")
    args = parser.parse_args()

    sys.path.insert(0, GMT_ROOT)
    # Stub mujoco_viewer before sim2sim imports it.
    fake_module = types.ModuleType("mujoco_viewer")
    fake_module.MujocoViewer = _NoopViewer
    sys.modules["mujoco_viewer"] = fake_module

    os.chdir(GMT_ROOT)

    import sim2sim  # noqa: E402  imports HumanoidEnv

    jit_policy_pth = "assets/pretrained_checkpoints/pretrained.pt"
    motion_file = os.path.join("assets/motions", args.motion)

    device = args.device
    env = sim2sim.HumanoidEnv(
        policy_path=jit_policy_pth,
        motion_path=motion_file,
        robot_type=args.robot,
        device=device,
        record_video=True,  # uses the stubbed viewer; no real rendering
    )

    print(f"[gmt_smoke] device={device} motion={args.motion} steps={args.steps}")
    print(f"[gmt_smoke] num_dofs={env.num_dofs} n_proprio={env.n_proprio} "
          f"history_len={env.history_len}")

    # Inline the inference loop without touching the viewer logic.
    import mujoco
    last_actions = []
    for i in range(args.steps):
        dof_pos, dof_vel, quat, ang_vel = env.extract_data()
        if i % env.sim_decimation == 0:
            curr_timestep = i // env.sim_decimation
            mimic_obs = env._get_mimic_obs(curr_timestep)
            rpy = sim2sim.quatToEuler(quat)
            obs_dof_vel = dof_vel.copy()
            obs_dof_vel[[4, 5, 10, 11]] = 0.0
            obs_prop = np.concatenate([
                ang_vel * env.ang_vel_scale,
                rpy[:2],
                (dof_pos - env.default_dof_pos) * env.dof_pos_scale,
                obs_dof_vel * env.dof_vel_scale,
                env.last_action,
            ])
            obs_hist = np.array(env.proprio_history_buf).flatten()
            obs_buf = np.concatenate([mimic_obs, obs_prop, obs_hist])
            obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(env.device)
            with torch.no_grad():
                raw_action = env.policy_jit(obs_tensor).cpu().numpy().squeeze()
            env.last_action = raw_action.copy()
            raw_action = np.clip(raw_action, -10.0, 10.0)
            scaled = raw_action * env.action_scale
            pd_target = scaled + env.default_dof_pos
            env.proprio_history_buf.append(obs_prop)
            last_actions.append(raw_action)
        torque = (pd_target - dof_pos) * env.stiffness - dof_vel * env.damping
        torque = np.clip(torque, -env.torque_limits, env.torque_limits)
        env.data.ctrl = torque
        mujoco.mj_step(env.model, env.data)

    last_actions = np.stack(last_actions)
    print(f"[gmt_smoke] ran {args.steps} sim steps "
          f"({last_actions.shape[0]} policy calls)")
    print(f"[gmt_smoke] action stats: mean={last_actions.mean():+.4f} "
          f"std={last_actions.std():.4f} "
          f"min={last_actions.min():+.4f} max={last_actions.max():+.4f}")
    print(f"[gmt_smoke] final qpos[:7]={env.data.qpos[:7]}")
    print(f"[gmt_smoke] OK")


if __name__ == "__main__":
    main()
