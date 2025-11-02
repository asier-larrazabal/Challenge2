import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
from snakeenv import SnakeEnv

# Configuration
ALGORITHM = "PPO"  # Options: PPO, A2C, DQN
USE_CURRICULUM = True
NUM_ENVS = 4  # Parallel environments (reduced for stability)
TOTAL_TIMESTEPS = 2_000_000

models_dir = f"models/{ALGORITHM}_curriculum" if USE_CURRICULUM else f"models/{ALGORITHM}"
log_dir = f"logs/{ALGORITHM}_curriculum" if USE_CURRICULUM else f"logs/{ALGORITHM}"


# Curriculum schedule
class CurriculumSchedule:
    def __init__(self, start_curriculum=0.3, decay_rate=0.95):
        """
        Start with easier curriculum (0.3 = 30% chance of survival on collision)
        Decay to 0 as agent improves
        """
        self.curriculum = start_curriculum
        self.decay_rate = decay_rate
        self.best_score = 0
    
    def update(self, score):
        """Decrease curriculum difficulty when agent improves"""
        if score > self.best_score:
            self.best_score = score
            self.curriculum = max(0.0, self.curriculum * self.decay_rate)
            print(f"Curriculum updated: {self.curriculum:.4f}, Best score: {self.best_score}")
        return self.curriculum


def make_env(rank, curriculum=0.0):
    """Create environment with curriculum"""
    def _init():
        env = SnakeEnv(curriculum=curriculum, render_mode=None)
        env = Monitor(env, log_dir)
        return env
    return _init


def train():
    """Main training function"""
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    curriculum_schedule = CurriculumSchedule()
    
    # Create vectorized environments for parallel training
    print(f"Creating {NUM_ENVS} parallel environments...")
    
    if USE_CURRICULUM:
        # Start with easier curriculum
        env = SubprocVecEnv([make_env(i, curriculum_schedule.curriculum) for i in range(NUM_ENVS)])
    else:
        env = SubprocVecEnv([make_env(i, 0.0) for i in range(NUM_ENVS)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(0, 0.0)])

    # Optimized hyperparameters for Snake PPO
    if ALGORITHM == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,  # Steps per environment per update
            batch_size=64,
            n_epochs=10,
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            device="cpu"  # Auto-detect CUDA/CPU
        )

    elif ALGORITHM == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            device="auto"
        )

    elif ALGORITHM == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=100000,
            learning_starts=10000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log=log_dir,
            device="auto"
        )

    # Callbacks for saving best models and checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{models_dir}/best_model",
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="checkpoint"
    )

    callback = CallbackList([eval_callback, checkpoint_callback])

    # Training information
    print(f"\n{'='*60}")
    print(f"Starting training with {ALGORITHM}")
    print(f"{'='*60}")
    print(f"Curriculum Learning: {USE_CURRICULUM}")
    print(f"Number of parallel environments: {NUM_ENVS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Models directory: {models_dir}")
    print(f"Logs directory: {log_dir}")
    print(f"{'='*60}\n")

    if USE_CURRICULUM:
        # Train in stages with progressive difficulty
        stages = 10
        timesteps_per_stage = TOTAL_TIMESTEPS // stages
        
        for stage in range(stages):
            print(f"\n{'='*60}")
            print(f"Training Stage {stage + 1}/{stages}")
            print(f"{'='*60}")
            print(f"Curriculum difficulty: {curriculum_schedule.curriculum:.4f}")
            print(f"Timesteps this stage: {timesteps_per_stage:,}")
            
            # Update environment curriculum
            env.close()
            env = SubprocVecEnv([
                make_env(i, curriculum_schedule.curriculum) 
                for i in range(NUM_ENVS)
            ])
            model.set_env(env)
            
            # Train for this stage
            model.learn(
                total_timesteps=timesteps_per_stage,
                callback=callback,
                reset_num_timesteps=False,
                tb_log_name=f"{ALGORITHM}_curriculum"
            )
            
            # Evaluate and update curriculum
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=20
            )
            print(f"\nStage {stage + 1} Evaluation:")
            print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
            
            # Update curriculum based on performance
            curriculum_schedule.update(mean_reward)
            
            # Save stage model
            model.save(f"{models_dir}/stage_{stage + 1}")
            print(f"  Model saved: stage_{stage + 1}")
            
    else:
        # Standard training without curriculum
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            tb_log_name=ALGORITHM
        )

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"Final mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

    # Save final model
    model.save(f"{models_dir}/final_model")
    print(f"Final model saved: {models_dir}/final_model")

    # Cleanup
    env.close()
    eval_env.close()

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Models saved to: {models_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    train()
