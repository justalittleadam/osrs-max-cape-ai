import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from environments import MiningEnvironment

def main():
    # Create environment
    print("Creating mining environment...")
    env = MiningEnvironment()

    # Create PPO model
    print("Creating PPO model...")
    model = PPO("MlpPolicy", env, verbose=1)

    # Train for a short test
    print("Starting training...")
    model.learn(total_timesteps=1000)

    print("Training complete! Testing trained model...")

    # Test the trained model
    obs, info = env.reset()
    for i in range(10):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done or truncated:
            obs, info = env.reset()

    env.close()
    print("✅ Training and testing complete!")

if __name__ == "__main__":
    main()