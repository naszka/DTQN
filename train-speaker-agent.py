import argparse
import os
import torch
import numpy as np
from torch.nn import Module

from utils import env_processing, epsilon_anneal
from utils.agent_utils import MODEL_MAP, get_agent
from utils.random import set_global_seed, RNG
from utils.logging_utils import RunningAverage, get_logger, timestamp
from run import evaluate, train
from envs.speaker_mushroom_forest import create_speaker_env, SpeakerMushroomForest
from envs.mushroom_forest import MushroomForest, encode_state


def load_listener_agent(model_path, model_type, base_env, device):
    """
    Load a trained listener agent from a model file

    Args:
        model_path (str): Path to the trained model
        model_type (str): Type of model (e.g., "DTQN")
        base_env (gym.Env): The environment the listener will interact with
        device (torch.device): Device to run the model on
    """
    # Ensure RNG is initialized
    if not hasattr(RNG, 'rng') or RNG.rng is None:
        RNG.seed(42)

    # Get environment dimensions
    env_obs_length = base_env.observation_space.shape[0]
    num_actions = base_env.action_space.n

    # Create a listener agent with the same hyperparameters as during training
    listener = get_agent(
        model_type,
        [base_env],  # Environment
        8,  # obs_embed
        0,  # a_embed
        128,  # in_embed
        500_000,  # buf_size
        device,
        3e-4,  # learning_rate
        32,  # batch_size
        50,  # context_len
        100,  # max_episode_steps
        50,  # history
        10_000,  # tuf
        0.99,  # discount
        # DTQN specific params
        8,  # heads
        2,  # layers
        0.0,  # dropout
        False,  # identity
        "res",  # gate
        "learned",  # pos
        0,  # bag_size
    )

    # Load the trained weights
    listener.load_mini_checkpoint(model_path)

    # Set to evaluation mode
    listener.eval_on()

    # Verify the listener is compatible with the environment
    assert listener.env_obs_length == env_obs_length, f"Listener observation size {listener.env_obs_length} doesn't match environment {env_obs_length}"
    assert listener.num_actions == num_actions, f"Listener action space {listener.num_actions} doesn't match environment {num_actions}"

    # Initialize context with a dummy observation
    dummy_obs = np.zeros(listener.env_obs_length, dtype=np.float32)
    listener.context_reset(dummy_obs)

    print(f"Loaded listener agent with obs_size={listener.env_obs_length}, action_size={listener.num_actions}")

    return listener


def train_speaker_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listener-model-path", type=str, required=True,
                        help="Path to the trained listener agent model")
    parser.add_argument("--listener-model-type", type=str, default="DTQN",
                        help="Type of the listener agent model")
    parser.add_argument("--n-cells", type=int, default=10,
                        help="Number of cells in the mushroom forest")
    parser.add_argument("--max-features", type=int, default=3,
                        help="Maximum number of features")
    parser.add_argument("--max-features-per-cell", type=int, default=2,
                        help="Maximum number of features per cell")
    parser.add_argument("--num-steps", type=int, default=1_000_000,
                        help="Number of steps to train the speaker agent")

    # Add all the arguments from run.py for consistency
    parser.add_argument("--project-name", type=str, default="SpeakerAgent",
                        help="Project name for wandb")
    parser.add_argument("--disable-wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--model", type=str, default="DTQN",
                        choices=list(MODEL_MAP.keys()),
                        help="Network model to use for the speaker agent")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--eval-frequency", type=int, default=5_000,
                        help="How many training timesteps between evaluations")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of episodes for each evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Pytorch device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Define feature weights (should match what was used to train the listener)
    feature_weights = np.array([1.0, 20.0, -1.0])

    # Set random seed first
    set_global_seed(args.seed)

    print(f"[{timestamp()}] Creating base environment...")

    # First, create the shared base environment that both listener and speaker will use
    base_mushroom_env = MushroomForest(
        n_cells=args.n_cells,
        max_features=args.max_features,
        feature_weights=feature_weights,
        max_features_per_cell=args.max_features_per_cell
    )

    print(f"[{timestamp()}] Created base environment with {args.n_cells} cells and {args.max_features} features")

    # Load the trained listener agent
    print(f"[{timestamp()}] Loading listener agent from {args.listener_model_path}...")
    listener = load_listener_agent(
        args.listener_model_path,
        args.listener_model_type,
        base_mushroom_env,
        device
    )

    # Now create the speaker environment using the same base environment and the loaded listener
    print(f"[{timestamp()}] Creating speaker environment...")
    speaker_env = create_speaker_env(
        n_cells=args.n_cells,
        max_features=args.max_features,
        feature_weights=feature_weights,
        max_features_per_cell=args.max_features_per_cell,
        trained_listener=listener,
        base_env=base_mushroom_env  # Pass the same base environment
    )

    # Create epsilon annealing schedule
    eps = epsilon_anneal.LinearAnneal(1.0, 0.1, args.num_steps // 10)

    # Create the speaker agent
    speaker_agent = get_agent(
        args.model,
        [speaker_env],  # List of environments
        8,  # obs_embed - May need tuning for the speaker's observation space
        0,  # a_embed
        128,  # in_embed
        500_000,  # buf_size
        device,
        args.lr,
        args.batch,
        50,  # context_len
        100,  # max_episode_steps
        50,  # history
        10_000,  # tuf (target update frequency)
        0.99,  # discount
        # DTQN specific params
        8,  # heads
        2,  # layers
        0.0,  # dropout
        False,  # identity
        "res",  # gate
        "learned",  # pos
        0,  # bag_size
    )

    print(f"[{timestamp()}] Created {args.model} Speaker agent with "
          f"{sum(p.numel() for p in speaker_agent.policy_network.parameters())} parameters")

    # Create directories for saving models
    policy_save_dir = os.path.join(os.getcwd(), "policies", "speaker_agent")
    os.makedirs(policy_save_dir, exist_ok=True)

    policy_path = os.path.join(
        policy_save_dir,
        f"speaker_agent_ncells={args.n_cells}_features={args.max_features}_seed={args.seed}"
    )

    # Setup for logging
    mean_success_rate = RunningAverage(10)
    mean_reward = RunningAverage(10)
    mean_episode_length = RunningAverage(10)

    # Create logger
    args.envs = "SpeakerMushroomForest"
    logger = get_logger(policy_path, args, {"name": "speaker_agent_training"})

    # Train the speaker agent
    print(f"[{timestamp()}] Starting speaker agent training...")
    train(
        speaker_agent,
        [speaker_env],  # Train environments
        [speaker_env],  # Eval environments (same as train for now)
        ["speaker_env"],  # Environment strings
        ["speaker_env"],  # Eval environment strings
        args.num_steps,
        eps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path,
        True,  # save_policy
        logger,
        mean_success_rate,
        mean_reward,
        mean_episode_length,
        None,  # time_remaining
        args.verbose,
    )

    print(f"[{timestamp()}] Speaker agent training completed!")


if __name__ == "__main__":
    train_speaker_agent()