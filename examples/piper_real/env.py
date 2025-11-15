import einops
import numpy as np
from lerobot_robot_piper import LerobotPiper, LerobotPiperConfig
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

cam_map = {
    "stereo": "stereo",
    "wrist1": "wrist1",
    "wrist2": "wrist2",
}

class PiperRealEnvironment(_environment.Environment):
    """Environment wrapper for LerobotPiper robot."""

    def __init__(
        self,
        config: LerobotPiperConfig | None = None,
        render_height: int = 224,
        render_width: int = 224,
    ) -> None:
        if config is None:
            config = LerobotPiperConfig()

        self._robot = LerobotPiper(config)
        self._render_height = render_height
        self._render_width = render_width
        self._connected = False

    @override
    def reset(self) -> None:
        if not self._connected:
            self._robot.connect()
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
        left_joints = np.array([obs[f"L.joint_{i}"] for i in range(7)])
        right_joints = np.array([obs[f"R.joint_{i}"] for i in range(7)])
        state = np.concatenate([left_joints, right_joints])

        # Process images - resize and convert to CHW format
        images = {}
        for cam_name in ["stereo", "wrist1", "wrist2"]:
            if cam_name in obs:
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(obs[cam_name], self._render_height, self._render_width)
                )
                images[cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": state,
            **images,
            "prompt": "pick and place",
        }

    @override
    def apply_action(self, action: dict) -> None:
        # Convert from array to dict format expected by LerobotPiper
        action_dict = {}
        for i, v in enumerate(action['actions_arm0']):
            action_dict[f"L.joint_{i}"] = float(v)
        for i, v in enumerate(action['actions_arm1']):
            action_dict[f"R.joint_{i}"] = float(v)

        self._robot.send_action(action_dict)

    def disconnect(self) -> None:
        if self._connected:
            self._robot.disconnect()
            self._connected = False