import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .mushroom_forest import MushroomForest, encode_state


class SpeakerMushroomForest(gym.Wrapper):
    """
    Environment for a speaker agent that learns to send messages to a listener agent
    in the MushroomForest environment.

    The speaker:
    - Can fully observe the forest features and reward weights
    - Chooses what information to share with the listener
    - Is rewarded based on the listener's performance

    Args:
        base_env (MushroomForest): The base MushroomForest environment
        trained_listener: A trained agent that will receive the messages
    """

    def __init__(self, n_cells, max_features, feature_weights, max_features_per_cell=None,
                 trained_listener=None, base_env=None):
        # Use the provided base environment or create a new one
        if base_env is not None:
            self.forest_env = base_env
        else:
            self.forest_env = MushroomForest(n_cells, max_features, feature_weights, max_features_per_cell)

        super().__init__(self.forest_env)

        self.n_cells = n_cells
        self.max_features = max_features
        self.feature_weights = feature_weights
        self.max_features_per_cell = max_features_per_cell or max_features
        self.trained_listener = trained_listener

        # Define action space for the speaker agent
        # Actions:
        # - 0 to max_features-1: Send reward weight for feature i
        # - max_features to max_features + (n_cells * max_features_per_cell) - 1:
        #   Send observation about cell/feature pair
        self.action_space = spaces.Discrete(max_features + (n_cells * max_features_per_cell))

        # Create a more complete observation space that includes full forest knowledge
        # This contains:
        # - Position of the listener (one-hot)
        # - Complete features matrix of the forest (n_cells x max_features)
        # - Feature weights vector
        # - Last action taken by the listener

        obs_size = (
                n_cells +  # Position (one-hot)
                (n_cells * max_features) +  # Full forest features matrix flattened
                max_features +  # Feature weights
                n_cells + 1  # Last listener action (n_cells visit actions + 1 pick action)
        )

        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(obs_size,),
            dtype=np.float32
        )

        # Keep track of the listener's state for reward calculation
        self.listener_prev_reward = 0
        self.cumulative_reward = 0
        self.listener_position = 0
        self.listener_last_action = None
        self.current_message = None

    def reset(self, seed=None, options=None):
        """Reset the environment and the listener's context"""
        # Reset the underlying environment
        obs = self.forest_env.reset(seed=seed)

        # Reset listener state tracking
        self.listener_prev_reward = 0
        self.cumulative_reward = 0
        self.listener_position = 0
        self.listener_last_action = None
        self.current_message = None

        # Initialize the listener's context if available
        if self.trained_listener is not None:
            # Get initial observation for the listener - without any message yet
            raw_obs = self.forest_env._get_dict_obs()
            listener_obs = encode_state(raw_obs, self.n_cells, self.max_features)
            self.trained_listener.context_reset(listener_obs)

        # Return the speaker's observation
        return self._get_speaker_obs()

    def _get_speaker_obs(self):
        """Create the speaker's observation vector that has full knowledge of the environment"""
        # Position (one-hot)
        position = np.zeros(self.n_cells)
        position[self.forest_env.position] = 1

        # Full forest features matrix (flattened)
        forest_features = self.forest_env.features.flatten()

        # Feature weights
        weights = np.array(self.forest_env.feature_weights)

        # Last listener action (one-hot, including the pick action)
        listener_action = np.zeros(self.n_cells + 1)
        if self.listener_last_action is not None:
            listener_action[self.listener_last_action] = 1

        # Combine all features
        return np.concatenate([
            position,
            forest_features,
            weights,
            listener_action
        ], dtype=np.float32)

    def _create_message(self, action):
        """
        Create a message based on the speaker's action.

        Args:
            action (int): The speaker's action

        Returns:
            dict: A message dictionary compatible with encode_state
        """
        if action < self.max_features:
            # Send reward weight for feature action
            message = {
                'type': 'reward',
                'feature_idx': int(action),
                'weight': float(self.feature_weights[action])
            }
        else:
            # Send observation about a cell/feature pair
            adjusted_action = action - self.max_features
            cell_idx = adjusted_action // self.max_features_per_cell
            feature_idx = adjusted_action % self.max_features_per_cell

            # Check if the feature exists in the cell
            if cell_idx < self.n_cells and feature_idx < self.max_features and self.forest_env.features[
                cell_idx, feature_idx] == 1:
                # Feature exists in the cell
                message = {
                    'type': 'state',
                    'cell_idx': int(cell_idx),
                    'feature_idx': int(feature_idx)
                }
            else:
                # Feature doesn't exist in the cell
                message = {
                    'type': 'state',
                    'cell_idx': int(cell_idx),
                    'feature_idx': None
                }

        self.current_message = message
        return message

    def step(self, action):
        """
        Take a step in the environment using the speaker's action to create a message
        for the listener agent.
        """
        # Create message based on speaker's action
        message = self._create_message(action)

        # Get the original observation
        raw_obs = self.forest_env._get_dict_obs()

        # Add the message to the observation for the listener
        raw_obs['message'] = message

        # Encode the observation for the listener agent
        listener_obs = encode_state(raw_obs, self.n_cells, self.max_features)

        if self.trained_listener:
            # Update the listener with the new observation
            if self.listener_last_action is not None:
                # Only send reward and done info when we've taken an action
                self.trained_listener.observe(
                    listener_obs,
                    self.listener_last_action,
                    self.listener_prev_reward,
                    False  # Not done yet
                )

            # Get the listener's action based on this observation
            listener_action = self.trained_listener.get_action(epsilon=0.0)

            # Take the listener's action in the environment
            next_obs, reward, done, truncated, info = self.forest_env.step(listener_action)

            # Update listener state tracking
            self.listener_prev_reward = reward
            self.cumulative_reward += reward
            self.listener_position = self.forest_env.position
            self.listener_last_action = listener_action

            # The speaker's reward is the same as the listener's
            speaker_reward = reward

            # Add message to info for debugging
            info["message"] = message
            info["listener_action"] = listener_action
            info["cumulative_reward"] = self.cumulative_reward
        else:
            # If no listener provided (e.g. during testing), just return some basic info
            next_obs, reward, done, truncated, info = self.forest_env.step(0)  # Take a default action
            speaker_reward = -1  # Default penalty for each step

        # Get the speaker's observation for the next state
        speaker_obs = self._get_speaker_obs()

        return speaker_obs, speaker_reward, done, truncated, info

    def render(self):
        """Render the environment with additional speaker-specific information"""
        print(f"Speaker Environment - Listener position: {self.listener_position}")
        if self.current_message is not None:
            print(f"Last message sent: {self.current_message}")
        print(f"Listener cumulative reward: {self.cumulative_reward}")
        self.forest_env.render()


# Function to create the environment for the speaker agent
def create_speaker_env(n_cells, max_features, feature_weights, max_features_per_cell,
                       trained_listener=None, base_env=None):
    """
    Create a SpeakerMushroomForest environment.

    Args:
        n_cells (int): Number of cells in the forest
        max_features (int): Maximum number of features
        feature_weights (list): Weights for the linear reward function
        max_features_per_cell (int): Maximum number of features per cell
        trained_listener: A trained listener agent
        base_env (MushroomForest, optional): A base environment to wrap

    Returns:
        SpeakerMushroomForest: The speaker environment
    """
    return SpeakerMushroomForest(
        n_cells=n_cells,
        max_features=max_features,
        feature_weights=feature_weights,
        max_features_per_cell=max_features_per_cell,
        trained_listener=trained_listener,
        base_env=base_env
    )