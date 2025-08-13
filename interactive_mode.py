import os
import argparse
import torch
import numpy as np
from utils import env_processing
from utils.agent_utils import get_agent
from utils.random import set_global_seed
from envs.mushroom_forest import encode_state


def unwrap_env(env):
    """Unwrap environment to get to the actual MushroomForest environment"""
    while hasattr(env, 'env'):
        env = env.env
    return env

class InteractiveMushroomForest:
    """Interactive wrapper for MushroomForest that allows manual message control"""

    def __init__(self, base_env):
        self.base_env = base_env
        self.unwrapped_env = unwrap_env(base_env)
        self.current_obs = None

        # Verify we have the right environment
        if not hasattr(self.unwrapped_env, 'position'):
            raise ValueError(f"Unwrapped environment {type(self.unwrapped_env)} doesn't have 'position' attribute")
        if not hasattr(self.unwrapped_env, 'features'):
            raise ValueError(f"Unwrapped environment {type(self.unwrapped_env)} doesn't have 'features' attribute")

        print(f"✅ Unwrapped environment: {type(self.unwrapped_env)}")

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation"""
        dict_obs = self.base_env.reset(seed=seed, options=options)
        # Convert back to dict format for manipulation
        self.current_obs = {
            'position': self.unwrapped_env.position,
            'current_features': self.unwrapped_env.features[self.unwrapped_env.position].copy(),
            'message': {'type': 'empty'},  # Start with empty message
            'encoded_state': None
        }
        return self._encode_current_obs()

    def step(self, action, custom_message=None):
        """Step the environment with optional custom message"""


        _, reward, done, truncated, info = self.base_env.step(action)

        # Store the message we want to inject
        if custom_message is not None:
            injected_message = custom_message
        else:
            injected_message = info["raw_obs"]['message']



        # Update our observation with the current state and injected message
        self.current_obs = {
            'position': self.unwrapped_env.position,
            'current_features': self.unwrapped_env.features[self.unwrapped_env.position].copy(),
            'message': injected_message,
        }

        return self._encode_current_obs(), reward, done, truncated, info

    def _encode_current_obs(self):
        """Encode the current observation"""
        return encode_state(self.current_obs, self.unwrapped_env.n_cells, self.unwrapped_env.max_features)

    def get_current_state_info(self):
        """Get detailed information about current state"""
        return {
            'position': self.unwrapped_env.position,
            'current_features': self.unwrapped_env.features[self.unwrapped_env.position].copy(),
            'all_features': self.unwrapped_env.features.copy(),
            'potential_rewards': [self.unwrapped_env._calculate_reward(i) for i in range(self.unwrapped_env.n_cells)],
            'feature_weights': self.unwrapped_env.feature_weights,
            'current_message': self.current_obs.get('message', {'type': 'empty'}),
            'n_cells': self.unwrapped_env.n_cells,
        }

    def render(self):
        """Enhanced rendering with more detail"""
        state_info = self.get_current_state_info()
        print("\n" + "=" * 60)
        print(f"🍄 MUSHROOM FOREST STATE 🍄")
        print("=" * 60)
        print(f"Current Position: {state_info['position']}")
        print(f"Current Cell Features: {state_info['current_features']}")
        print(f"Current Message: {state_info['current_message']}")
        print()
        print("All Cells Overview:")
        for i, features in enumerate(state_info['all_features']):
            marker = "👤" if i == state_info['position'] else "  "
            reward = state_info['potential_rewards'][i]
            features_str = np.array2string(features, separator=', ')
            print(f"{marker} Cell {i}: {features_str} → Reward: {reward:.1f}")
        print(f"\nFeature Weights: {state_info['feature_weights']}")
        print("=" * 60)



def create_custom_message(env_info):
    """Interactive function to create custom messages"""
    print("\n📨 MESSAGE CREATOR 📨")
    print("Choose message type:")
    print("1. Empty message")
    print("2. State message (about a cell's features)")
    print("3. Reward message (about feature weights)")

    #try:
    choice = input("Enter choice (1-3): ").strip()

    if choice == '1':
        return {'type': 'empty'}

    elif choice == '2':
        print(f"\nAvailable cells: 0-{env_info['n_cells'] - 1}")
        cell_idx = int(input("Enter cell index: "))
        if not (0 <= cell_idx < env_info['n_cells']):
            print("Invalid cell index, using 0")
            cell_idx = 0

        # Show features in that cell
        cell_features = env_info['all_features'][cell_idx]
        active_features = np.where(cell_features == 1)[0]
        print(f"Cell {cell_idx} has active features: {list(active_features)}")

        if len(active_features) > 0:
            print(f"Available feature indices: {list(active_features)}")
            feature_idx = int(input("Enter feature index (or -1 for None): "))
            if feature_idx == -1 or feature_idx not in active_features:
                feature_idx = None
        else:
            print("No active features in this cell")
            feature_idx = None

        return {
            'type': 'state',
            'cell_idx': cell_idx,
            'feature_idx': feature_idx
        }

    elif choice == '3':
        print(f"\nAvailable features: 0-{len(env_info['feature_weights']) - 1}")
        print(f"Feature weights: {env_info['feature_weights']}")
        feature_idx = int(input("Enter feature index: "))
        if not (0 <= feature_idx < len(env_info['feature_weights'])):
            print("Invalid feature index, using 0")
            feature_idx = 0

        return {
            'type': 'reward',
            'feature_idx': feature_idx,
            'weight': float(env_info['feature_weights'][feature_idx])
        }

    else:
        print("Invalid choice, using empty message")
        return {'type': 'empty'}

    #except (ValueError, KeyError) as e:
    #    print(f"Error creating message: {e}")
    #    print("Using empty message")
    #    return {'type': 'empty'}


def interactive_mode(agent, env, max_steps=100):
    """Main interactive loop"""
    print("\n🎮 INTERACTIVE AGENT MODE 🎮")
    print("Commands:")
    print("  'auto' - Let agent choose action")
    print("  'msg' - Create custom message")
    print("  'state' - Show current state")
    print("  'quit' - Exit")
    print("  0-N - Take specific action (0-N are visit actions, N+1 is pick)")

    obs = env.reset()
    agent.context_reset(obs)
    env.render()

    step_count = 0
    total_reward = 0

    while step_count < max_steps:
        print(f"\n--- Step {step_count + 1} ---")

        # Get agent's suggested action
        suggested_action = agent.get_action(epsilon=0.0)  # Greedy action
        action_names = [f"Visit cell {i}" for i in range(env.unwrapped_env.n_cells)] + ["Pick current cell"]
        print(f"🤖 Agent suggests: Action {suggested_action} ({action_names[suggested_action]})")

        # Get user input
        user_input = input("\nYour command: ").strip().lower()

        if user_input == 'quit':
            break
        elif user_input == 'state':
            env.render()
            continue
        elif user_input == 'msg':
            state_info = env.get_current_state_info()
            custom_message = create_custom_message(state_info)
            print(f"Created message: {custom_message}")
            # Don't take action, just show what would happen with this message
            continue
        elif user_input == 'auto':
            action = suggested_action
            custom_message = None
        else:
            try:
                action = int(user_input)
                if not (0 <= action <= env.unwrapped_env.n_cells):
                    print(f"Invalid action. Use 0-{env.unwrapped_env.n_cells}")
                    continue
                custom_message = None
            except ValueError:
                print("Invalid command")
                continue

        # Ask if user wants to inject a custom message
        if custom_message is None:
            msg_choice = input("Inject custom message? (y/n): ").strip().lower()
            if msg_choice == 'y':
                state_info = env.get_current_state_info()
                custom_message = create_custom_message(state_info)

        # Take the action
        next_obs, reward, done, truncated, info = env.step(action, custom_message)
        print(f"obs_next: {next_obs}")
        # Update agent's context
        agent.observe(next_obs, action, reward, done)

        # Show results
        print(f"\n✅ Action taken: {action} ({action_names[action]})")
        if custom_message and custom_message['type'] != 'empty':
            print(f"📨 Message sent: {custom_message}")
        print(f"💰 Reward: {reward}")
        total_reward += reward
        print(f"💎 Total reward: {total_reward}")

        env.render()

        if done or truncated:
            print(f"\n🏁 Episode finished! Final reward: {total_reward}")
            break

        step_count += 1

    print(f"\nSession ended after {step_count} steps with total reward: {total_reward}")


def main():
    parser = argparse.ArgumentParser(description="Interactive mode for trained agent")
    parser.add_argument("--policy-path", type=str, required=True,
                        help="Path to the trained policy file")
    parser.add_argument("--env", type=str, default="MushroomForest-v1",
                        help="Environment name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")

    # Agent parameters (should match training)
    parser.add_argument("--model", type=str, default="DTQN", help="Model type")
    parser.add_argument("--obs-embed", type=int, default=8)
    parser.add_argument("--a-embed", type=int, default=0)
    parser.add_argument("--in-embed", type=int, default=128)
    parser.add_argument("--context", type=int, default=50)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gate", type=str, default="res")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--pos", type=str, default="learned")
    parser.add_argument("--bag-size", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Create environment
    base_env = env_processing.make_env(args.env)
    env = InteractiveMushroomForest(base_env)

    # Set seed
    set_global_seed(args.seed, env.base_env)

    # Create agent (matching training parameters)
    device = torch.device(args.device)
    agent = get_agent(
        args.model,
        [env.base_env],  # Pass as list
        args.obs_embed,
        args.a_embed,
        args.in_embed,
        buffer_size=1000,  # Not important for inference
        device=device,
        learning_rate=1e-4,  # Not important for inference
        batch_size=32,  # Not important for inference
        context_len=args.context,
        max_env_steps=args.max_steps,
        history=50,  # Default from training
        target_update_frequency=10000,  # Not important for inference
        gamma=0.99,  # Not important for inference
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        identity=args.identity,
        gate=args.gate,
        pos=args.pos,
        bag_size=args.bag_size,
    )

    # Load trained policy
    try:
        agent.load_mini_checkpoint(args.policy_path)
        agent.eval_on()  # Set to evaluation mode
        print(f"✅ Loaded policy from {args.policy_path}")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    # Start interactive mode
    try:
        interactive_mode(agent, env, args.max_steps)
    except KeyboardInterrupt:
        print("\n\n👋 Interactive session ended by user")
    except Exception as e:
        print(f"\n❌ Error during interactive session: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()