import einops
import numpy as np
from lerobot_robot_yams import BiYamsFollower, BiYamsFollowerConfig
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

class YamRealEnvironment(_environment.Environment):
    """Environment wrapper for BiYamsFollower robot."""

    def __init__(
        self,
        config: BiYamsFollowerConfig | None = None,
        render_height: int = 224,
        render_width: int = 224,
        prompt='Fold the towel.'
    ) -> None:
        if config is None:
            config = BiYamsFollowerConfig()

        self._robot = BiYamsFollower(config)
        self._render_height = render_height
        self._render_width = render_width
        self._connected = False
        self.prompt=prompt

    @override
    def reset(self) -> None:
        if not self._connected:
            self._robot.connect()
            self._robot.configure()
            self._connected = True

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if not self._connected:
            raise RuntimeError("Robot is not connected. Call reset() first.")

        obs = self._robot.get_observation()

        # Extract joint states for both arms (14 DoF total)
        # BiYamsFollower returns: left_joint_1.pos through left_gripper.pos, right_joint_1.pos through right_gripper.pos
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        left_joints = np.array([obs[f"left_{name}.pos"] for name in joint_names])
        right_joints = np.array([obs[f"right_{name}.pos"] for name in joint_names])
        state = np.concatenate([left_joints, right_joints])

        # Process images - resize and convert to CHW format
        # Camera names should match what yam_policy expects: "stereo", "wrist.left", "wrist.right"
        images = {}
        cam_mapping = {
            "topdown": "stereo",
            "left_wrist": "wrist.left",
            "right_wrist": "wrist.right",
        }
        for cam_key, output_key in cam_mapping.items():
            if cam_key in obs:
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs[cam_key], self._render_height, self._render_width)
                )
                images[output_key] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": state,
            **images,
            "prompt": self.prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        # Convert from array to dict format expected by BiYamsFollower
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        action_dict = {}
        for i, name in enumerate(joint_names):
            action_dict[f"left_{name}.pos"] = float(action['actions_arm0'][i])
            action_dict[f"right_{name}.pos"] = float(action['actions_arm1'][i])

        self._robot.send_action(action_dict)

    def disconnect(self) -> None:
        if self._connected:
            self._robot.disconnect()
            self._connected = False