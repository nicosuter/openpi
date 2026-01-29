import logging

from lerobot.datasets import lerobot_dataset
import numpy as np

from examples.yam_sim.robot_sim import RobotYamSim
from openpi.policies.yam_policy import _parse_image
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment



class YamSimEnv(_environment.Environment):
    def __init__(
        self,
        repo_id:str,
        render_height: int = 224,
        render_width: int = 224,
        sim_robot:bool = True,
        prompt='Fold the towel.',
        eps_idx=0
    ) -> None:
        self._render_height = render_height
        self._render_width = render_width
        self.sim_robot=sim_robot
        self.prompt=prompt
        self.repo_id = repo_id
        self.dataset = lerobot_dataset.LeRobotDataset(repo_id=repo_id)
        self.idx=self.dataset.meta.episodes[eps_idx]['dataset_from_index']
        
        if sim_robot:
            self.robot = RobotYamSim()
        


    
    def get_observation(self) -> dict:
        obs = self.dataset[self.idx]
        images = {}
        cam_mapping = {
            "observation.images.topdown": "stereo",
            "observation.images.left_wrist": "wrist.left",
            "observation.images.right_wrist": "wrist.right",
        }
        for cam_key, output_key in cam_mapping.items():
            if cam_key in obs:
                img  = obs[cam_key]
                img = _parse_image(img)
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, self._render_height, self._render_width)
                )
                images[output_key] = img

        return {
            "state": obs['observation.state'].numpy(),
            **images,
            "prompt": self.prompt,
        }
    
    def apply_action(self, action:dict):
        obs = self.dataset[self.idx]
        target_act = obs['action'].numpy()
        print(f'[{self.idx}] L1 norm {np.linalg.norm(action["actions"] - target_act, ord=1) :1.3f}')
        self.idx+=1
        logging.info(f"Server infer time: {action['server_timing']['infer_ms']/1000:1.3f}s")
        if self.sim_robot:
            self.robot.apply_action(action['actions'])

    def reset(self) -> None:
        pass
    
    def is_episode_complete(self) -> bool:
        return False