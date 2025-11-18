import dataclasses
import pathlib
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
from openpi.models.tokenizer import PaligemmaTokenizer
from pathlib import Path
import jax.numpy as jnp
import math
import numpy as np
import torch
from sarm.dataset.gap_dataset import GapLerobotDataset
from sarm.model.reward_sarm import RewardSarm

import openpi.models.pi0_config as pi0_config
from openpi.shared.download import DEFAULT_CACHE_DIR
from openpi.training.config import LeRobotPiperDataConfig, AssetsConfig, DataConfig, _CONFIGS_DICT, \
    LeRobotPiperSarmDataConfig
from openpi.training.data_loader import create_torch_dataset, transform_dataset, create_torch_data_loader
from openpi_client import image_tools
from scripts import compute_norm_stats, train


def test_happy_path_get_rewards_from_sarm():
    class SarmMock:
        def __init__(self, T=9):
            self.n = 0
            self.T = T
            
    
        def __call__(self, batch):
            self.n += 1
            shape = batch['observation.state'].shape
            return jnp.ones((shape[0], shape[1])) * self.n
        
    sarm_mock = SarmMock()
    reward_model = RewardSarm(sarm=sarm_mock)
    repo_id = 'ETHRC/piper_towel_v0'
    dataset_gab = GapLerobotDataset(repo_id=repo_id, action_horizon=25, frame_gap=30, t_step_lookback=8)
    batch_size = 4
    data_loader = torch.utils.data.DataLoader(
        dataset_gab,
        batch_size=batch_size,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    weights = reward_model(batch)
    assert len(weights) == batch_size
    assert math.isclose(weights.sum(), 1, abs_tol=1e-5)
    
def test_data_configuration():
    action_horizon = 42
    piper_config = LeRobotPiperDataConfig(
        repo_id='ETHRC/piper_towel_v0',
        assets=AssetsConfig(
            assets_dir="",
            asset_id=None,
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    )
    data_config = piper_config.create(assets_dirs=Path('/'), model_config=pi0_config.Pi0Config())
    lerobot_dataset = create_torch_dataset(data_config=data_config, action_horizon=action_horizon, model_config=None)
    frame = lerobot_dataset[12]
    assert frame is not None
    assert 'observation.images.wrist1' in frame
    assert 'action' in frame
    assert 'observation.state' in frame
    assert 'prompt' in frame
    assert frame['action'].shape == (action_horizon, 14)
    assert frame['prompt'] == frame['task']

    # Apply transformations
    dataset_transformed = transform_dataset(lerobot_dataset, data_config, skip_norm_stats=True)
    frame_transformed = dataset_transformed[12]

    # Compare image inputs, transformed dataset resizes to 224,224,3
    image_map = {'observation.images.wrist1': 'right_wrist_0_rgb',
                 'observation.images.wrist2': 'left_wrist_0_rgb',
                 'observation.images.stereo': 'base_0_rgb',
                 }
    for key_lerobot, key_transformed in image_map.items():
        image_original = frame[key_lerobot]
        image_original = image_original.permute(1, 2, 0).cpu().numpy()  #
        image_original = image_tools.convert_to_uint8(image_original)
        image_original = image_tools.resize_with_pad(image_original, 224, 224)  # (224, 224, 3)

        image_transformed = frame_transformed['image'][key_transformed]
        diff = image_original.astype(np.float32) - image_transformed.astype(np.float32)
        mse = np.mean(diff ** 2)
        assert image_original.mean() > 10
        assert mse < 0.01
    
    # Compare state
    np.testing.assert_array_equal(frame_transformed['state'][:14], frame['observation.state'])
    np.testing.assert_array_equal(frame_transformed['state'][14:], np.zeros_like(frame_transformed['state'][14:]))
    
    # Compare actions
    assert frame_transformed['actions'].shape == (action_horizon, 32)
    for n in range(action_horizon):
        np.testing.assert_array_equal(frame_transformed['actions'][n][:6]+frame_transformed['state'][:6], frame['action'][n][:6])
        np.testing.assert_array_equal(frame_transformed['actions'][n][7:13]+frame_transformed['state'][7:13], frame['action'][n][7:13])
        # Grippers
        assert frame_transformed['actions'][n][6] == frame['action'][n][6]
        assert frame_transformed['actions'][n][13] == frame['action'][n][13]
        np.testing.assert_array_equal(frame_transformed['actions'][n][14:],np.zeros_like(frame_transformed['actions'][n][14:]))
    
    # Tokenizer
    tokenizer = PaligemmaTokenizer()
    valid_tokens = frame_transformed['tokenized_prompt'][frame_transformed['tokenized_prompt_mask']]
    decoded_prompt = tokenizer._tokenizer.decode(valid_tokens.tolist())
    assert decoded_prompt.strip() == frame['task']

def test_data_configuration_with_normalization():
    repo_id = 'ETHRC/piper_towel_v0'
    train_config_id = 'pi0_piper_debug'
    assets_dir = pathlib.Path(DEFAULT_CACHE_DIR).expanduser() / train_config_id
    if not (assets_dir / repo_id / 'norm_stats.json').exists():
        compute_norm_stats.main(train_config_id)
    action_horizon = 42
    piper_config = LeRobotPiperDataConfig(
        repo_id=repo_id,
        assets=AssetsConfig(
            assets_dir=str(assets_dir),
            asset_id=repo_id,
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    )
    data_config = piper_config.create(assets_dirs=assets_dir, model_config=pi0_config.Pi0Config())
    lerobot_dataset = create_torch_dataset(data_config=data_config, action_horizon=action_horizon, model_config=None)
    dataset_transformed = transform_dataset(lerobot_dataset, data_config, skip_norm_stats=True)
    dataset_transformed_norm = transform_dataset(lerobot_dataset, data_config, skip_norm_stats=False)
    frame_1 = dataset_transformed[100]
    frame_2 = dataset_transformed_norm[100]

    for key in (set(frame_1.keys()) - {'state', 'actions', 'image'}):
        np.testing.assert_array_equal(frame_1[key], frame_2[key])
    for key in frame_1['image']:
        np.testing.assert_array_equal(frame_1['image'][key], frame_2['image'][key])
    # normalized values should not be equal
    assert np.array_equal(frame_1['state'], frame_2['state']) == False
    assert np.array_equal(frame_1['actions'], frame_2['actions']) == False


@pytest.mark.parametrize("config_name", ["pi0_piper_debug"])
def test_train_policy(config_name, tmp_path, monkeypatch):
    config = dataclasses.replace(
        _CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=1,
        log_interval=1,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),

    )
    train.main(config)


def test_get_reward_data():
    action_horizon = 42
    piper_config = LeRobotPiperSarmDataConfig(
        repo_id='ETHRC/piper_towel_v0',
        assets=AssetsConfig(
            assets_dir="",
            asset_id=None,
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
        reward_model='sarm'
    )
    sarm_keys = {'gap_data_0.observation.state', 'gap_data_0.observation.images.wrist1',
                 'gap_data_0.observation.images.wrist2', 'gap_data_0.observation.images.stereo',
                 'gap_data_1.observation.state', 'gap_data_1.observation.images.wrist1',
                 'gap_data_1.observation.images.wrist2', 'gap_data_1.observation.images.stereo'}
    data_config = piper_config.create(assets_dirs=Path('/'), model_config=pi0_config.Pi0Config())
    lerobot_dataset = create_torch_dataset(data_config=data_config, action_horizon=action_horizon, model_config=None)
    dataset_transformed = transform_dataset(lerobot_dataset, data_config, skip_norm_stats=True)
    data_item = dataset_transformed[0]
    assert sarm_keys < set(data_item.keys())

    # Testing Dataloader
    data_loader = create_torch_data_loader(data_config=data_config,
                                           model_config=pi0_config.Pi0Config(),
                                           action_horizon=action_horizon,
                                           batch_size=2,
                                           skip_norm_stats=True)
    
    batch = iter(data_loader)
    obs, action, reward_inputs = next(batch)
    reward_inputs_dict = reward_inputs.to_dict()
    assert set(reward_inputs_dict.keys()) == sarm_keys
    assert reward_inputs_dict['gap_data_0.observation.state'].shape == (2, *tuple(data_item['gap_data_0.observation.state'].shape))
    
@pytest.mark.parametrize("config_name", ["pi0_piper_debug_reward"])
def test_train_policy_reward(config_name, tmp_path, monkeypatch):
    config = dataclasses.replace(
        _CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=1,
        log_interval=1,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),

    )
    train.main(config)