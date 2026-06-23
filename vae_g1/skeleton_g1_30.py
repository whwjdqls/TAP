"""G1Skeleton30 — matches the 30 bodies in the kimodo `g1skel34/xml/g1.xml`
MJCF. Kimodo's G1Skeleton34 also tracks 4 virtual endpoints (toe tips + palm
hand-roll), but the MJCF doesn't model those as bodies. For TMR we use the
30 actual MJCF bodies so `posed_joints` from MuJoCo FK lines up 1:1.

Joint hierarchy mirrors the MJCF body parent list.
"""
from __future__ import annotations

from kimodo.skeleton.base import SkeletonBase


class G1Skeleton30(SkeletonBase):
    """Unitree G1 skeleton, 30-joint MJCF subset.

    `SkeletonBase.__init__` normally loads `joints.p` / `lengths.p` rest-pose
    tensors from `kimodo/assets/skeletons/<name>/`. We don't ship those for
    `g1skel30`, and TMR only needs `posed_joints` at runtime — no FK from
    rotations. So we pass `load=False` and the parent skips the file load.
    """

    name = "g1skel30"

    def __init__(self, **kwargs):
        kwargs.setdefault("load", False)
        super().__init__(**kwargs)

    # Foot / hand joint names — used by TMRMotionRep's foot-contact detector.
    # Picked to be valid MJCF joints (no virtual toe-tip / palm).
    right_foot_joint_names = ["right_ankle_pitch_joint", "right_ankle_roll_joint"]
    left_foot_joint_names  = ["left_ankle_pitch_joint",  "left_ankle_roll_joint"]
    right_hand_joint_names = ["right_wrist_pitch_joint", "right_wrist_yaw_joint"]
    left_hand_joint_names  = ["left_wrist_pitch_joint",  "left_wrist_yaw_joint"]

    hip_joint_names = ["right_hip_pitch_joint", "left_hip_pitch_joint"]

    bone_order_names_with_parents = [
        ("pelvis", None),
        ("left_hip_pitch_joint", "pelvis"),
        ("left_hip_roll_joint", "left_hip_pitch_joint"),
        ("left_hip_yaw_joint", "left_hip_roll_joint"),
        ("left_knee_joint", "left_hip_yaw_joint"),
        ("left_ankle_pitch_joint", "left_knee_joint"),
        ("left_ankle_roll_joint", "left_ankle_pitch_joint"),
        ("right_hip_pitch_joint", "pelvis"),
        ("right_hip_roll_joint", "right_hip_pitch_joint"),
        ("right_hip_yaw_joint", "right_hip_roll_joint"),
        ("right_knee_joint", "right_hip_yaw_joint"),
        ("right_ankle_pitch_joint", "right_knee_joint"),
        ("right_ankle_roll_joint", "right_ankle_pitch_joint"),
        ("waist_yaw_joint", "pelvis"),
        ("waist_roll_joint", "waist_yaw_joint"),
        ("waist_pitch_joint", "waist_roll_joint"),
        ("left_shoulder_pitch_joint", "waist_pitch_joint"),
        ("left_shoulder_roll_joint", "left_shoulder_pitch_joint"),
        ("left_shoulder_yaw_joint", "left_shoulder_roll_joint"),
        ("left_elbow_joint", "left_shoulder_yaw_joint"),
        ("left_wrist_roll_joint", "left_elbow_joint"),
        ("left_wrist_pitch_joint", "left_wrist_roll_joint"),
        ("left_wrist_yaw_joint", "left_wrist_pitch_joint"),
        ("right_shoulder_pitch_joint", "waist_pitch_joint"),
        ("right_shoulder_roll_joint", "right_shoulder_pitch_joint"),
        ("right_shoulder_yaw_joint", "right_shoulder_roll_joint"),
        ("right_elbow_joint", "right_shoulder_yaw_joint"),
        ("right_wrist_roll_joint", "right_elbow_joint"),
        ("right_wrist_pitch_joint", "right_wrist_roll_joint"),
        ("right_wrist_yaw_joint", "right_wrist_pitch_joint"),
    ]


__all__ = ["G1Skeleton30"]
