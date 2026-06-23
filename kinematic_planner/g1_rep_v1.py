"""g1_rep_v1 — a HumanML3D-style kinematic motion representation for the G1.

Built from Bones-SEED GT G1 data. Used as the generation target of a
text-to-motion MDM diffusion model. Per-frame layout (136-D):

    idx        name            width  meaning
    [0,1)      rot_velocity      1     per-frame root yaw delta  Δψ  (canonical)
    [1,3)      lin_velocity      2     root planar velocity in the heading frame (dx, dz)
    [3,4)      root_height       1     root (pelvis) world height (Y, kimodo Y-up)
    [4,10)     root_orient_6d    6     full root orientation, heading-removed (6D), so the
                                       executable decode reproduces pelvis pitch/roll, not
                                       just yaw
    [10,39)    joint_angles     29     normalized G1 actuated joint angles (qpos order)
    [39,138)   ric_data         99     joints 1..33 positions, root-relative XZ + world Y,
                                       heading-removed (33 = 34 G1 joints minus the root)
    [138,142)  foot_contacts     4     [L_ankle, L_toe, R_ankle, R_toe] in {0,1}

Decode (the executable path): integrate (rot_velocity, lin_velocity,
root_height) -> global root trajectory + yaw, then **root + 29 qpos angles ->
G1 forward kinematics** (via the MuJoCo qpos converter + G1Skeleton34.fk).
``ric_data`` is a redundant position target (HumanML3D-style) that also gives a
fast, rotation-free position decode for visualization.

Conventions (kimodo Y-up): up axis = +Y, ground plane = X-Z. Heading is the
root yaw about +Y, taken from the hip vector (``compute_heading_angle``). The
representation is canonical: frame 0 is shifted to the origin and rotated to
zero heading, so absolute world placement is dropped (standard for generation).

The encode/decode pair below uses explicit full-angle Y-rotations and is an
*exact algebraic inverse by construction* on the position side; the qpos->FK
path additionally incurs the G1 1-DOF-per-joint projection (~0.04 m, intrinsic)
and the yaw-only-root approximation (pelvis pitch/roll is carried by the waist
joints, not the root).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
N_JOINTS = 34          # G1Skeleton34
N_ACTUATED = 29        # actuated qpos DOFs
_RIC = (N_JOINTS - 1) * 3  # 99
FEAT_DIM = 1 + 2 + 1 + 6 + N_ACTUATED + _RIC + 4   # = 142


def feature_layout() -> "OrderedDict[str, slice]":
    o = 0
    d = OrderedDict()
    d["rot_velocity"] = slice(o, o + 1); o += 1
    d["lin_velocity"] = slice(o, o + 2); o += 2
    d["root_height"] = slice(o, o + 1); o += 1
    d["root_orient_6d"] = slice(o, o + 6); o += 6
    d["joint_angles"] = slice(o, o + N_ACTUATED); o += N_ACTUATED
    d["ric_data"] = slice(o, o + _RIC); o += _RIC
    d["foot_contacts"] = slice(o, o + 4); o += 4
    assert o == FEAT_DIM, (o, FEAT_DIM)
    return d


def _cont6d_to_matrix(c6: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(...,6) -> (...,3,3) via Gram-Schmidt; columns [x, y, z]."""
    x_raw, y_raw = c6[..., 0:3], c6[..., 3:6]
    x = x_raw / (x_raw.norm(dim=-1, keepdim=True) + eps)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / (z.norm(dim=-1, keepdim=True) + eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def _matrix_to_cont6d(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) -> (...,6): first two columns (x, y axes)."""
    return torch.cat([R[..., 0], R[..., 1]], dim=-1)


SLICE_DICT = feature_layout()


# ----------------------------------------------------------------------------
# Full-angle Y-axis rotation (kimodo Y-up). v' = R_y(theta) @ v.
# ----------------------------------------------------------------------------
def _Ry(theta: torch.Tensor) -> torch.Tensor:
    """theta (...) -> (..., 3, 3) rotation about +Y by theta."""
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([c, z, s, z, o, z, -s, z, c], dim=-1).reshape(theta.shape + (3, 3))


def _apply(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R (...,3,3) applied to v (...,3) -> (...,3)."""
    return torch.einsum("...ij,...j->...i", R, v)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + torch.pi) % (2 * torch.pi) - torch.pi


def heading_from_hips(posed_joints: torch.Tensor, r_hip: int, l_hip: int) -> torch.Tensor:
    """Per-frame root heading yaw from the hip vector (kimodo convention).

    posed_joints: (T, J, 3) -> (T,) heading angle. Matches
    ``kimodo.motion_rep.feature_utils.compute_heading_angle``:
    atan2(diff_z, -diff_x) on (right_hip - left_hip).
    """
    diff = posed_joints[:, r_hip] - posed_joints[:, l_hip]
    return torch.atan2(diff[:, 2], -diff[:, 0])


def _unwrapped_canonical_heading(posed_joints: torch.Tensor, r_hip: int, l_hip: int) -> torch.Tensor:
    """Heading made continuous (cumsum of wrapped deltas) and shifted so
    frame 0 = 0. Returns alpha (T,)."""
    head = heading_from_hips(posed_joints, r_hip, l_hip)            # (T,)
    if head.shape[0] == 1:
        return torch.zeros_like(head)
    dpsi = _wrap(head[1:] - head[:-1])                              # (T-1,)
    alpha = torch.zeros_like(head)
    alpha[1:] = torch.cumsum(dpsi, dim=0)                           # alpha[0]=0
    return alpha


# ----------------------------------------------------------------------------
# Encode
# ----------------------------------------------------------------------------
def encode(
    posed_joints: torch.Tensor,    # (T, 34, 3)  kimodo Y-up global joint positions
    root_positions: torch.Tensor,  # (T, 3)      pelvis xyz (== posed_joints[:,0])
    root_rot: torch.Tensor,        # (T, 3, 3)   root local rotation (local_rot_mats[:,0])
    joint_angles: torch.Tensor,    # (T, 29)     actuated qpos angles (radians)
    foot_contacts: torch.Tensor,   # (T, 4)      precomputed {0,1}
    r_hip: int,
    l_hip: int,
) -> torch.Tensor:
    """Encode one motion to (T, 136). Exact inverse of :func:`decode_positions`
    on the position side."""
    T = posed_joints.shape[0]
    dev, dt = posed_joints.device, posed_joints.dtype
    # Absolute, continuous (unwrapped) heading. Using the *absolute* heading for
    # the ric/lin_vel rotation makes the rep heading-invariant (a global Y
    # rotation of the input leaves the features unchanged); rot_velocity (its
    # per-frame delta) is what the decoder integrates, reconstructing the motion
    # canonicalized to zero heading at frame 0.
    head0 = heading_from_hips(posed_joints, r_hip, l_hip)[0]
    head = head0 + _unwrapped_canonical_heading(posed_joints, r_hip, l_hip)  # (T,)

    # rot_velocity = per-frame delta of the continuous heading (last duplicated)
    rot_vel = torch.zeros(T, device=dev, dtype=dt)
    if T > 1:
        rot_vel[:-1] = head[1:] - head[:-1]
        rot_vel[-1] = rot_vel[-2]

    root_xz = root_positions.clone()
    root_xz[:, 1] = 0.0                                                # (T,3) planar root

    Rneg = _Ry(-head)                                                  # (T,3,3) remove heading

    # lin_velocity: ego-frame planar root delta. lin_vel[t-1] inverts the
    # decode increment root[t]-root[t-1] = R_y(alpha[t]) @ ego[t]; last dup.
    lin_vel = torch.zeros(T, 2, device=dev, dtype=dt)
    if T > 1:
        droot = root_xz[1:] - root_xz[:-1]                            # (T-1,3)
        ego = _apply(Rneg[1:], droot)                                 # rotate by -alpha[t]
        lin_vel[:-1] = ego[:, [0, 2]]
        lin_vel[-1] = lin_vel[-2]

    root_height = root_positions[:, 1:2]                              # (T,1)

    # root orientation, heading-removed: R_y(-head) @ R_root -> 6D. Faithful
    # pelvis pitch/roll (not just yaw), heading-invariant.
    root_orient_6d = _matrix_to_cont6d(Rneg @ root_rot.to(dt))        # (T,6)

    # ric: non-root joints, root-XZ removed (keep Y), heading-removed.
    rel = posed_joints[:, 1:, :] - root_xz[:, None, :]               # (T,33,3)
    ric = _apply(Rneg[:, None, :, :], rel)                            # rotate each by -alpha[t]
    ric = ric.reshape(T, _RIC)

    feats = torch.cat(
        [rot_vel[:, None], lin_vel, root_height, root_orient_6d,
         joint_angles, ric, foot_contacts.to(dt)],
        dim=1,
    )
    assert feats.shape == (T, FEAT_DIM), feats.shape
    return feats


# ----------------------------------------------------------------------------
# Decode — positions (HumanML3D recover_from_ric analog; fast, rotation-free)
# ----------------------------------------------------------------------------
def _integrate_root(feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B,T,136)|(T,136) -> alpha (B,T), root_xyz (B,T,3), R_y(alpha) (B,T,3,3)."""
    sq = feats.dim() == 2
    if sq:
        feats = feats.unsqueeze(0)
    B, T, _ = feats.shape
    rot_vel = feats[..., SLICE_DICT["rot_velocity"]].squeeze(-1)      # (B,T)
    lin_vel = feats[..., SLICE_DICT["lin_velocity"]]                  # (B,T,2)
    root_h = feats[..., SLICE_DICT["root_height"]].squeeze(-1)        # (B,T)

    alpha = torch.zeros(B, T, device=feats.device, dtype=feats.dtype)
    if T > 1:
        alpha[:, 1:] = torch.cumsum(rot_vel[:, :-1], dim=1)
    R = _Ry(alpha)                                                    # (B,T,3,3)

    ego = torch.zeros(B, T, 3, device=feats.device, dtype=feats.dtype)
    if T > 1:
        ego[:, 1:, 0] = lin_vel[:, :-1, 0]
        ego[:, 1:, 2] = lin_vel[:, :-1, 1]
    droot = _apply(R, ego)                                            # rotate by +alpha[t]
    root = torch.cumsum(droot, dim=1)                                # (B,T,3) planar (y=0)
    root[..., 1] = root_h
    return (alpha, root, R) if not sq else (alpha[0], root[0], R[0])


def decode_positions(feats: torch.Tensor) -> torch.Tensor:
    """(B,T,136)|(T,136) -> world joints (B,T,34,3)|(T,34,3). Uses ric_data only."""
    sq = feats.dim() == 2
    if sq:
        feats = feats.unsqueeze(0)
    B, T, _ = feats.shape
    alpha, root, R = _integrate_root(feats)
    ric = feats[..., SLICE_DICT["ric_data"]].reshape(B, T, N_JOINTS - 1, 3)
    joints = _apply(R[:, :, None, :, :], ric)                        # rotate by +alpha[t]
    root_xz = root.clone(); root_xz[..., 1] = 0.0
    joints = joints + root_xz[:, :, None, :]
    world = torch.cat([root[:, :, None, :], joints], dim=2)          # (B,T,34,3)
    return world.squeeze(0) if sq else world


# ----------------------------------------------------------------------------
# Decode — executable: root + 29 qpos angles -> G1 forward kinematics.
# ----------------------------------------------------------------------------
def decode_qpos_to_joints(
    feats: torch.Tensor,           # (T, 136) single motion (unnormalized)
    converter,                     # kimodo.exports.mujoco.MujocoQposConverter
    skeleton,                      # kimodo.skeleton.G1Skeleton34 instance
) -> Dict[str, torch.Tensor]:
    """Reconstruct G1 pose from the rep using the qpos path.

    Joint local rotations come from the 29 angles via the MuJoCo converter
    (driven with an identity root); the root local rotation is set to the
    yaw-only R_y(alpha) in kimodo frame; root translation from the integrated
    trajectory. Returns dict with ``local_rot_mats``, ``root_positions``,
    ``posed_joints``.
    """
    feats = feats.detach()
    T = feats.shape[0]
    alpha, root, _R = _integrate_root(feats)                          # (T,), (T,3)
    angles = feats[:, SLICE_DICT["joint_angles"]]                     # (T,29)
    # Full root rotation = R_y(heading) @ heading-removed-orientation.
    R_root = _Ry(alpha) @ _cont6d_to_matrix(feats[:, SLICE_DICT["root_orient_6d"]])  # (T,3,3)

    # Build a qpos with identity root + the 29 angles; ask the converter for
    # per-joint local rotations (root-independent), in kimodo frame.
    qpos = np.zeros((T, 7 + N_ACTUATED), dtype=np.float64)
    qpos[:, 3] = 1.0                                                  # root quat w=1 (identity)
    qpos[:, 7:] = angles.cpu().numpy().astype(np.float64)
    mdict = converter.qpos_to_motion_dict(
        torch.from_numpy(qpos).float().unsqueeze(0),
        source_fps=20, root_quat_w_first=True, mujoco_rest_zero=False,
    )
    local = mdict["local_rot_mats"]
    if local.dim() == 5:
        local = local[0]                                             # (T,34,3,3)
    local = local.to(feats.dtype)

    # Override root (joint 0) local rotation with the reconstructed full root.
    local = local.clone()
    local[:, 0] = R_root.to(local.dtype)

    _, posed, _ = skeleton.fk(local.unsqueeze(0), root.unsqueeze(0))
    posed = posed[0]                                                  # (T,34,3)
    return {"local_rot_mats": local, "root_positions": root, "posed_joints": posed}


# ----------------------------------------------------------------------------
# Motion-rep wrapper (mirrors HumanML3DNativeMotionRep's interface)
# ----------------------------------------------------------------------------
class G1RepV1MotionRep(nn.Module):
    """Thin wrapper exposing what the MDM denoiser + loss need:
    ``motion_rep_dim``, ``nbjoints``, ``fps``, ``slice_dict``, ``skeleton``,
    ``normalize`` / ``unnormalize``. Mean/std are flat (FEAT_DIM,) arrays."""

    motion_rep_dim: int = FEAT_DIM

    def __init__(
        self,
        mean_path: str | Path,
        std_path: str | Path,
        skeleton,
        fps: int = 20,
        eps: float = 1e-6,
        feat_bias: float = 5.0,
        stats_path: Optional[str | Path] = None,   # accepted & ignored (helper injects it)
    ):
        super().__init__()
        self.skeleton = skeleton
        self.nbjoints = int(skeleton.nbjoints)
        if self.nbjoints != N_JOINTS:
            raise ValueError(f"G1RepV1MotionRep requires a 34-joint G1 skeleton; got {self.nbjoints}")
        self.fps = int(fps)
        self.slice_dict: Dict[str, slice] = feature_layout()
        self.feature_names: List[str] = list(self.slice_dict.keys())
        self.feat_bias = float(feat_bias)

        mean = torch.from_numpy(np.load(str(mean_path))).float()
        std = torch.from_numpy(np.load(str(std_path))).float()
        if mean.shape[-1] != FEAT_DIM or std.shape[-1] != FEAT_DIM:
            raise ValueError(f"mean/std must be ({FEAT_DIM},); got {tuple(mean.shape)}/{tuple(std.shape)}")

        # MDM/MoMask feat_bias: shrink std on root + contact blocks to up-weight
        # them in the normalized loss.
        if self.feat_bias != 1.0:
            std = std.clone()
            for name in ("rot_velocity", "lin_velocity", "root_height", "foot_contacts"):
                std[self.slice_dict[name]] = std[self.slice_dict[name]] / self.feat_bias

        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self._eps = float(eps)

    def _safe_std(self) -> torch.Tensor:
        return torch.sqrt(self.std * self.std + self._eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self._safe_std().to(device=x.device, dtype=x.dtype)
        return (x - m) / s

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mean.to(device=x.device, dtype=x.dtype)
        s = self._safe_std().to(device=x.device, dtype=x.dtype)
        return x * s + m


__all__ = [
    "FEAT_DIM", "N_JOINTS", "N_ACTUATED", "SLICE_DICT", "feature_layout",
    "encode", "decode_positions", "decode_qpos_to_joints", "heading_from_hips",
    "G1RepV1MotionRep",
]
