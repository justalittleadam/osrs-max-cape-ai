import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from environments.mining_env import MiningEnvironment
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
def main():
    # Create log directory
    log_dir = "../logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    print("Creating enhanced mining environment...")
    env = MiningEnvironment()

    # Wrap environment with Monitor for logging
    env = Monitor(env, log_dir)

    print("\n=== Testing Enhanced Navigation ===")

    # Test the enhanced environment first
    obs, info = env.reset()
    print(f"Starting area observation index [26]: {obs[26]}")  # Should show castle area
    print(f"Starting position: ({obs[0]:.0f}, {obs[1]:.0f})")  # Should be castle coords
    print(f"Distance to rocks: {obs[19]:.1f}")  # Should be far (50.0)
    print(f"Rocks in area: {obs[22]:.0f}")  # Should be 0 at castle

    # Test navigation to mine (action 1)
    print("\n1. Testing navigation to mine...")
    obs, reward, done, truncated, info = env.step(1)  # navigate_to_mine
    print(f"Action: navigate_to_mine")
    print(f"Reward: {reward:.3f}")
    print(f"Message: {info.get('server_response', {}).get('message', 'No message')}")
    print(f"New position: ({obs[0]:.0f}, {obs[1]:.0f})")

    # Continue navigating until we reach the mine
    navigation_steps = 0
    while obs[26] != 1.0 and navigation_steps < 15:  # 1.0 should represent lumbridge_mine
        obs, reward, done, truncated, info = env.step(1)  # Keep navigating
        navigation_steps += 1
        message = info.get('server_response', {}).get('message', '')
        if 'Arrived' in message:
            print(f"✅ {message}")
            break
        elif navigation_steps % 3 == 0:  # Print every 3rd step
            print(f"  Still walking... {message}")

    # Test mining once we're at the mine
    print("\n2. Testing mining...")
    obs, reward, done, truncated, info = env.step(7)  # mine_nearest
    print(f"Action: mine_nearest")
    print(f"Reward: {reward:.3f}")
    print(f"Message: {info.get('server_response', {}).get('message', 'No message')}")
    print(f"Inventory count: {obs[9]:.0f}")

    # Test navigation to bank (action 2)
    print("\n3. Testing navigation to bank...")
    obs, reward, done, truncated, info = env.step(2)  # navigate_to_bank
    print(f"Action: navigate_to_bank")
    print(f"Reward: {reward:.3f}")
    print(f"Message: {info.get('server_response', {}).get('message', 'No message')}")

    print("\n=== Enhanced Navigation Tests Complete! ===\n")

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,  # Save every 2000 steps
        save_path=log_dir + "checkpoints/",
        name_prefix="mining_model"
    )

    # Create PPO model with enhanced settings for navigation and TensorBoard logging
    print("Creating enhanced PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=64,  # More steps for complex navigation
        batch_size=16,
        n_epochs=4,
        gamma=0.99,  # Important for multi-step rewards
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        tensorboard_log=log_dir,  # Enable TensorBoard logging,
        device=device
    )

    # Train for longer since navigation is more complex
    print("Starting enhanced training...")
    print("Agent must learn: Castle → Mine → Mining → Bank → Repeat")
    print("TensorBoard logs will be saved to:", log_dir)
    print("Start TensorBoard with: tensorboard --logdir", log_dir)

    model.learn(
        total_timesteps=10000,  # Increased from 1000
        callback=checkpoint_callback,
        # progress_bar=True
    )

    # Save the final model
    model.save(log_dir + "final_mining_model")
    print(f"Final model saved to: {log_dir}final_mining_model")

    print("Training complete! Testing trained model...")

    # Test the trained model with detailed logging
    obs, info = env.reset()
    total_reward = 0
    step_count = 0

    print(f"\n=== Testing Trained Agent ===")
    print(f"Starting position: ({obs[0]:.0f}, {obs[1]:.0f})")

    for i in range(50):  # More steps to see full cycle
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Get action name for display
        action_names = {
            0: "no_move", 1: "navigate_to_mine", 2: "navigate_to_bank",
            3: "move_to_iron", 4: "move_to_coal", 5: "move_to_bank",
            6: "no_mine", 7: "mine_nearest", 8: "mine_iron", 9: "mine_tin",
            10: "mine_copper", 11: "mine_coal", 12: "no_inventory",
            13: "drop_ore", 14: "bank_all", 15: "bank_ore_only",
            16: "wait_tick", 17: "wait_mining", 18: "wait_movement",
            19: "move_to_mithril", 20: "move_to_adamant", 21: "move_to_runite"
        }
        action_name = action_names.get(action, f"action_{action}")

        # Print interesting events
        message = info.get('server_response', {}).get('message', '')
        if any(keyword in message for keyword in ['Arrived', 'mined', 'Banked', 'Walking', 'Successfully']):
            print(f"Step {step_count}: {action_name} → {message} (Reward: {reward:.2f})")

        # Render every 10 steps
        if i % 10 == 0:
            env.render()

        if done or truncated:
            print(f"Episode ended: {info.get('reason', 'Unknown reason')}")
            obs, info = env.reset()
            break

    print(f"\nFinal Results:")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward / step_count:.3f}")

    env.close()
    print("✅ Enhanced training and testing complete!")
    print(f"View training progress: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()