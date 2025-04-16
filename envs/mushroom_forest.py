import gymnasium as gym
import numpy as np
from gymnasium import spaces


def encode_state(obs, n_cells, max_features):
    """
    Encode the observation into a comprehensive feature vector

    Args:
    - obs: Dictionary observation from the environment
    - n_cells: Number of cells in the environment
    - max_features: Maximum number of features per cell

    Returns:
    - Numpy array representing the encoded state
    """
    # Position (one-hot)
    position = np.zeros(n_cells)
    position[obs['position']] = 1

    # Current cell features (supporting variable number of features)
    current_features = obs['current_features']

    # Message encoding
    message = obs.get('message', {'type': None})

    # Message type (one-hot encoding, now with three types)
    message_type = np.zeros(3)
    if message['type'] == 'empty':
        message_type[0] = 1
    elif message['type'] == 'state':
        message_type[1] = 1
    elif message['type'] == 'reward':
        message_type[2] = 1

    # State message specific features
    state_cell_idx = np.zeros(n_cells)
    state_feature_idx = np.zeros(max_features)
    if message['type'] == 'state':
        if message['cell_idx'] is not None:
            state_cell_idx[message['cell_idx']] = 1
        if message['feature_idx'] is not None:
            state_feature_idx[message['feature_idx']] = 1

    # Reward message specific features
    reward_feature_idx = np.zeros(max_features)
    reward_weight = np.zeros(1)
    if message['type'] == 'reward':
        if 'feature_idx' in message:
            reward_feature_idx[message['feature_idx']] = 1
        if 'weight' in message:
            reward_weight[0] = message['weight']

    # Combine all features
    state_vector = np.concatenate([
        position,
        current_features,
        message_type,
        state_cell_idx,
        state_feature_idx,
        reward_feature_idx,
        reward_weight,
    ], dtype=np.float32)

    return state_vector

class MushroomForest(gym.Env):
    """
    Env with flexible features and linear rewards

    Args:
        n_cells (int): Number of cells in the grid
        max_features (int): Maximum number of features per cell
        feature_weights (np.array): Weight vector for the linear reward function
        max_features_per_cell (int, optional): Maximum number of features a cell can have.
                                               Defaults to max_features.
    """

    def __init__(self, n_cells, max_features, feature_weights, max_features_per_cell=None):
        super(MushroomForest, self).__init__()

        self.n_cells = n_cells
        self.max_features = max_features
        self.feature_weights = np.array(feature_weights)

        # If max_features_per_cell not specified, default to max_features
        self.max_features_per_cell = max_features_per_cell or max_features

        # Validate feature weights
        assert len(self.feature_weights) == max_features, "Feature weights must match max number of features"

        # Validate max features per cell
        assert self.max_features_per_cell <= max_features, "Max features per cell cannot exceed total max features"

        # Action space: N visit actions + 1 pick action
        self.action_space = spaces.Discrete(n_cells + 1)
        self.PICK_ACTION = n_cells  # The last action is the pick action

        # Calculate the size of the encoded state vector
        self.encoded_state_size = (
                n_cells +  # Position (one-hot)
                max_features +  # Current cell features
                3 +  # Message type (one-hot)
                n_cells +  # State message cell index (one-hot)
                max_features +  # State message feature index (one-hot)
                max_features +  # Reward message feature index (one-hot)
                1  # Reward message weight
        )

        # Update observation space to be a Box for the encoded state
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.encoded_state_size,),
            dtype=np.float32
        )

        # Initialize state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize features with a flexible number of features per cell
        self.features = np.zeros((self.n_cells, self.max_features), dtype=int)
        for i in range(self.n_cells):
            # Randomly choose the number of features for this cell
            n_cell_features = self.np_random.integers(0, self.max_features_per_cell + 1)

            # Randomly select which features to activate
            feature_indices = self.np_random.choice(
                self.max_features,
                size=n_cell_features,
                replace=False
            )
            self.features[i, feature_indices] = 1

        self.position = 0  # Start at cell 0

        return self._get_encoded_obs()

    def _get_dict_obs(self):
        """Returns the raw dictionary observation (for internal use)"""
        return {
            'position': self.position,
            'current_features': self.features[self.position]
        }

    def _get_encoded_obs(self):
        """Returns the encoded observation vector"""
        dict_obs = self._get_dict_obs()
        return encode_state(dict_obs, self.n_cells, self.max_features)

    def _calculate_reward(self, cell_idx):
        """Calculate reward for a cell based on its features"""
        return np.dot(self.features[cell_idx], self.feature_weights)

    def step(self, action):
        reward = -1  # Base cost for any action
        done = False

        if action == self.PICK_ACTION:  # Pick current cell
            reward += self._calculate_reward(self.position)
            self.features[self.position] = 0  # Empty the cell
        else:  # Visit action
            self.position = action

        # Check if all cells are empty
        cell_rewards = [self._calculate_reward(i) for i in range(self.n_cells)]
        if all(reward <= 2 for reward in cell_rewards):
            done = True

        return self._get_encoded_obs(), reward, done, False, {}

    def render(self):
        print(f"Current position: {self.position}")
        raw_obs = self._get_dict_obs()
        print(f"Raw observation: {raw_obs}")
        print(f"Encoded observation shape: {self._get_encoded_obs().shape}")
        print("Potential rewards:", [self._calculate_reward(i) for i in range(self.n_cells)])


class Speaker0MushroomForest(MushroomForest):
    def __init__(self, n_cells, max_features, feature_weights,
                 max_features_per_cell,
                 message_type_probs=(1 / 3, 1 / 3, 1 / 3)):
        """
        Initialize the environment with configurable message type probabilities

        Args:
        - message_type_probs (tuple): Probabilities for (empty, state, reward) messages
                                      Must sum to 1
        """
        # Validate message type probabilities
        assert len(message_type_probs) == 3, "Must provide exactly 3 probabilities"
        assert np.isclose(sum(message_type_probs), 1.0), "Probabilities must sum to 1"

        self.message_type_probs = message_type_probs
        super().__init__(n_cells, max_features, feature_weights, max_features_per_cell)



    def _get_dict_obs(self):
        # Get the basic observation from the parent class
        obs = super()._get_dict_obs()

        # Choose message type based on provided probabilities
        message_type = self.np_random.choice(
            ['empty', 'state', 'reward'],
            p=self.message_type_probs
        )

        if message_type == 'empty':
            # Empty message with all values set to 0
            message = {
                'type': 'empty'
            }
        elif message_type == 'state':
            # Message about state (same as before)
            cell_idx = self.np_random.integers(0, self.n_cells)
            true_feature_indices = np.where(self.features[cell_idx] == 1)[0]

            if len(true_feature_indices) > 0:
                feature_idx = self.np_random.choice(true_feature_indices)
                message = {
                    'type': 'state',
                    'cell_idx': cell_idx,
                    'feature_idx': int(feature_idx)
                }
            else:
                message = {
                    'type': 'state',
                    'cell_idx': cell_idx,
                    'feature_idx': None
                }
        else:  # reward message
            feature_idx = self.np_random.integers(0, self.max_features)
            message = {
                'type': 'reward',
                'feature_idx': int(feature_idx),
                'weight': float(self.feature_weights[feature_idx])
            }

        # Add the message to the observation
        obs['message'] = message

        return obs


# Example usage would look like:
if __name__ == "__main__":
    # Create environment with custom message type probabilities
    weights = np.array([1.0, 20.0, -1.0])
    env = Speaker0MushroomForest(
        n_cells=10,
        max_features=3,
        feature_weights=weights,
        max_features_per_cell=2,
        message_type_probs=(0.1, 0.6, 0.3)  # 10% empty, 60% state, 30% reward
    )