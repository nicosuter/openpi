from pathlib import Path
import i2rt.robot_models
import mujoco
import mujoco.viewer
import numpy as np


class RobotYamSim:
    def __init__(self):
        """Initialize robot and start visualization."""
        # Load MuJoCo model
        model_path = Path(i2rt.robot_models.__path__[0]) / "yam" / "yam.xml"
        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)

        # Launch viewer immediately
        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

        # Initial sync
        self._viewer.sync()

    def apply_action(self, action: np.ndarray):
        """
        Apply action to robot and update visualization.

        Args:
            action: Array of joint positions
        """
        # Directly set joint positions for immediate response (kinematic mode)
        n_joints = min(len(action), self._model.nq)
        self._data.qpos[:n_joints] = action[:n_joints]

        # Forward kinematics to update dependent quantities
        mujoco.mj_forward(self._model, self._data)

        # Update visualization
        if self._viewer.is_running():
            self._viewer.sync()

    def __del__(self):
        """Cleanup viewer on deletion."""
        if hasattr(self, '_viewer') and self._viewer is not None:
            self._viewer.close()