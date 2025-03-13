import gymnasium as gym
import numpy as np
from gymnasium import spaces


def encode_state(obs, n_cells, m_features):
    """
    Encode the observation into a comprehensive feature vector

    Args:
    - obs: Dictionary observation from the environment
    - n_cells: Number of cells in the environment
    - m_features: Number of features per cell

    Returns:
    - Numpy array representing the encoded state
    """
    # Position (one-hot)
    position = np.zeros(n_cells)
    position[obs['position']] = 1

    # Current cell features (binary)
    current_features = obs['current_features']

    # Message encoding
    message = obs.get('message', {'type': None})

    # Message type (one-hot)
    message_type = np.zeros(2)
    if message['type'] == 'state':
        message_type[0] = 1
    elif message['type'] == 'reward':
        message_type[1] = 1

    # State message specific features
    state_cell_idx = np.zeros(n_cells)
    state_feature_idx = np.zeros(m_features)
    if message['type'] == 'state':
        if message['cell_idx'] is not None:
            state_cell_idx[message['cell_idx']] = 1
        if message['feature_idx'] is not None:
            state_feature_idx[message['feature_idx']] = 1

    # Reward message specific features
    reward_feature_idx = np.zeros(m_features)
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
    Env with binary features and linear rewards

    Args:
        n_cells (int): Number of cells in the grid
        m_features (int): Number of binary features per cell
        feature_weights (np.array): Weight vector for the linear reward function
    """

    def __init__(self, n_cells, m_features, feature_weights):
        super(MushroomForest, self).__init__()

        self.n_cells = n_cells
        self.m_features = m_features
        self.feature_weights = np.array(feature_weights)

        # Validate feature weights
        assert len(self.feature_weights) == m_features, "Feature weights must match number of features"

        # Action space: N visit actions + 1 pick action
        self.action_space = spaces.Discrete(n_cells + 1)
        self.PICK_ACTION = n_cells  # The last action is the pick action

        # Calculate the size of the encoded state vector
        self.encoded_state_size = (
            n_cells +                # Position (one-hot)
            m_features +             # Current cell features
            2 +                      # Message type (one-hot)
            n_cells +                # State message cell index (one-hot)
            m_features +             # State message feature index (one-hot)
            m_features +             # Reward message feature index (one-hot)
            1                        # Reward message weight
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

        # Randomly initialize binary features
        self.features = self.np_random.integers(0, 2, size=(self.n_cells, self.m_features))
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
        return encode_state(dict_obs, self.n_cells, self.m_features)

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
    def _get_dict_obs(self):
        # Get the basic observation from the parent class
        obs = super()._get_dict_obs()

        # 50% chance to send a message about state or reward function
        if self.np_random.random() < 0.9:
            # Message about state
            # Randomly choose a cell
            cell_idx = self.np_random.integers(0, self.n_cells)

            # Find indices of features that are true (1) in the chosen cell
            true_feature_indices = np.where(self.features[cell_idx] == 1)[0]

            # If there are any true features, pick one randomly
            if len(true_feature_indices) > 0:
                feature_idx = self.np_random.choice(true_feature_indices)
                message = {
                    'type': 'state',
                    'cell_idx': cell_idx,
                    'feature_idx': int(feature_idx)
                }
            else:
                # If no true features, still share the cell but with a None feature
                message = {
                    'type': 'state',
                    'cell_idx': cell_idx,
                    'feature_idx': None
                }
        else:
            # Message about reward function
            # Randomly select a feature index
            feature_idx = self.np_random.integers(0, self.m_features)
            message = {
                'type': 'reward',
                'feature_idx': int(feature_idx),
                'weight': float(self.feature_weights[feature_idx])
            }

        # Add the message to the observation
        obs['message'] = message

        return obs

    def _get_encoded_obs(self):
        """Returns the encoded observation vector"""
        dict_obs = self._get_dict_obs()
        return encode_state(dict_obs, self.n_cells, self.m_features)


if __name__ == "__main__":
    # Create environment with 4 cells, 3 features each, and random weights
    weights = np.array([1.0, 20.0, -1.0])
    env = Speaker0MushroomForest(n_cells=10, m_features=3, feature_weights=weights)
    print("Forest features:")
    print(env.features)

    # Reset environment
    obs = env.reset()
    print(f"Encoded observation shape: {obs.shape}")
    env.render()

    # Example episode
    for _ in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, truncated, info = env.step(action)
        print(f"\nAction: {'pick' if action == env.PICK_ACTION else f'visit {action}'}")
        print(f"Reward: {reward}")
        print(f"Encoded observation shape: {obs.shape}")
        env.render()

        if done:
            break