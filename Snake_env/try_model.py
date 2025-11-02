"""
Visualize Best Snake Model - CPU Version
"""
from stable_baselines3 import PPO
from snakeenv import SnakeEnv
import time

# Create environment with rendering enabled
env = SnakeEnv(curriculum=0.0, render_mode='human')

# Load your best model - FORCE CPU USAGE
print("Loading best model on CPU...")
model = PPO.load('models/PPO_curriculum/best_model/best_model', env=env, device='cpu')
print("Model loaded! Starting visualization...\n")

# Play 5 episodes
for episode in range(1, 6):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    print(f"Episode {episode} - Starting...")
    
    while not (done or truncated):
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # FIX: Convert numpy array to int
        action = int(action)
        
        # Take action in environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Render the game (you'll see the snake moving)
        env.render()
        
        # Slow down so you can watch (adjust or remove this)
        time.sleep(0.1)
    
    # Print episode results
    print(f"Episode {episode} finished - Score: {info['score']}, "
          f"Length: {info['length']}, Steps: {steps}, Reward: {total_reward:.2f}\n")
    
    # Short pause between episodes
    time.sleep(1)

env.close()
print("Done!")
