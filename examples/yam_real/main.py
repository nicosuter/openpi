import dataclasses
import logging

from lerobot_robot_yams import BiYamsFollowerConfig
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.yam_real import env as _env


@dataclasses.dataclass
class Args:
    host: str = 'localhost'
    port: int = 7000

    action_horizon: int = 10

    num_episodes: int = 1
    max_episode_steps: int = 1000

    # BiYamsFollower config
    left_can_port: str = "can_follower_l"
    left_server_port: int = 11333
    right_can_port: str = "can_follower_r"
    right_server_port: int = 11334


def main(args: Args) -> None:
    # Connect to policy server
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    # Create robot config
    robot_config = BiYamsFollowerConfig(
        left_arm_can_port=args.left_can_port,
        left_arm_server_port=args.left_server_port,
        right_arm_can_port=args.right_can_port,
        right_arm_server_port=args.right_server_port,
    )

    # Create runtime with Yam environment
    runtime = _runtime.Runtime(
        environment=_env.YamRealEnvironment(config=robot_config),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=30,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        # Ensure robot is disconnected on exit
        if hasattr(runtime._environment, "disconnect"):
            runtime._environment.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)