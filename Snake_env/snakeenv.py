import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random

SNAKE_LEN_GOAL = 30

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, curriculum=0.0, render_mode=None):
        super(SnakeEnv, self).__init__()
        
        self.action_space = spaces.Discrete(4)
        
        # Compact observation: 11 features
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32
        )
        
        self.curriculum = curriculum
        self.render_mode = render_mode
        
        # Grid parameters
        self.grid_size = 500
        self.cell_size = 10
        self.grid_cells = self.grid_size // self.cell_size
        
        # Performance tracking
        self.episode_steps = 0
        self.max_steps = 1000
        self.steps_since_apple = 0
        self.max_steps_without_apple = 100
        
        # Visualization
        if render_mode == 'human':
            self.img = np.zeros((self.grid_size, self.grid_size, 3), dtype='uint8')
        else:
            self.img = None
            
        self.score = 0
        self.max_score = 0
        self.direction = 1

    def _get_observation(self):
        """Optimized observation with 11 features"""
        head_x, head_y = self.snake_head
        apple_x, apple_y = self.apple_position
        
        # Direction to apple (normalized)
        apple_delta_x = (apple_x - head_x) / self.grid_size
        apple_delta_y = (apple_y - head_y) / self.grid_size
        
        # Danger detection - check if NEXT position in each direction is dangerous
        danger_left = float(self._is_collision([head_x - self.cell_size, head_y]))
        danger_right = float(self._is_collision([head_x + self.cell_size, head_y]))
        danger_up = float(self._is_collision([head_x, head_y - self.cell_size]))
        danger_down = float(self._is_collision([head_x, head_y + self.cell_size]))
        
        # Current direction (one-hot)
        dir_left = 1.0 if self.direction == 0 else 0.0
        dir_right = 1.0 if self.direction == 1 else 0.0
        dir_down = 1.0 if self.direction == 2 else 0.0
        dir_up = 1.0 if self.direction == 3 else 0.0
        
        # Snake length (normalized)
        snake_length_norm = len(self.snake_position) / SNAKE_LEN_GOAL
        
        observation = np.array([
            apple_delta_x,
            apple_delta_y,
            danger_left,
            danger_right,
            danger_up,
            danger_down,
            dir_left,
            dir_right,
            dir_down,
            dir_up,
            snake_length_norm
        ], dtype=np.float32)
        
        return observation

    def _is_collision(self, position):
        """Check if position causes collision"""
        x, y = position
        # Boundary check
        if x >= self.grid_size or x < 0 or y >= self.grid_size or y < 0:
            return True
        # Self collision (check against full snake except last segment which will move)
        if position in self.snake_position[:-1]:
            return True
        return False

    def step(self, action):
        self.episode_steps += 1
        self.steps_since_apple += 1
        
        # Store previous distance to apple
        prev_distance = np.linalg.norm(
            np.array(self.snake_head, dtype=float) - np.array(self.apple_position, dtype=float)
        )
        
        # CRITICAL FIX: Calculate new head position WITHOUT modifying snake_head yet
        new_head = list(self.snake_head)
        
        # Prevent 180-degree turns
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if action != opposite.get(self.direction):
            self.direction = action
        
        # Calculate new head position based on direction
        if self.direction == 0:  # Left
            new_head[0] -= self.cell_size
        elif self.direction == 1:  # Right
            new_head[0] += self.cell_size
        elif self.direction == 2:  # Down
            new_head[1] += self.cell_size
        elif self.direction == 3:  # Up
            new_head[1] -= self.cell_size
        
        # Initialize reward
        reward = 0.0
        terminated = False
        truncated = False
        
        # Check if new position eats apple
        if new_head == self.apple_position:
            # Ate apple - grow snake
            self.snake_position.insert(0, new_head)
            self.snake_head = new_head
            self.apple_position = self._generate_apple()
            self.score += 1
            self.steps_since_apple = 0
            reward = 10.0
            
        else:
            # Normal move - no growth
            self.snake_position.insert(0, new_head)
            self.snake_position.pop()
            self.snake_head = new_head
            
            # Distance-based reward shaping
            current_distance = np.linalg.norm(
                np.array(self.snake_head, dtype=float) - np.array(self.apple_position, dtype=float)
            )
            reward = (prev_distance - current_distance) / self.grid_size
        
        # Check for collisions AFTER updating position
        if self._is_collision_after_move():
            # Curriculum learning: probabilistic survival
            if random.random() > self.curriculum:
                terminated = True
                reward = -10.0
            else:
                reward = -1.0
        
        # Timeout checks
        if self.steps_since_apple >= self.max_steps_without_apple:
            truncated = True
            reward -= 5.0
        
        if self.episode_steps >= self.max_steps:
            truncated = True
        
        # Victory condition
        if len(self.snake_position) >= SNAKE_LEN_GOAL:
            reward = 100.0
            terminated = True
        
        observation = self._get_observation()
        info = {
            'score': self.score,
            'length': len(self.snake_position),
            'steps': self.episode_steps
        }
        
        if self.render_mode == 'human':
            self._update_ui()
        
        return observation, reward, terminated, truncated, info

    def _is_collision_after_move(self):
        """Check if current head position is in collision"""
        x, y = self.snake_head
        # Boundary check
        if x >= self.grid_size or x < 0 or y >= self.grid_size or y < 0:
            return True
        # Self collision (check if head hit body - exclude head itself)
        if self.snake_head in self.snake_position[1:]:
            return True
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.score > self.max_score:
            self.max_score = self.score
            print(f"New maximum score: {self.max_score}")
        
        self.episode_steps = 0
        self.steps_since_apple = 0
        self.score = 0
        
        # Initialize snake at center
        center = self.grid_size // 2
        center = (center // self.cell_size) * self.cell_size
        
        self.snake_position = [
            [center, center],
            [center - self.cell_size, center],
            [center - 2 * self.cell_size, center]
        ]
        
        self.snake_head = list(self.snake_position[0])
        self.direction = 1  # Start moving right
        
        # Generate apple
        self.apple_position = self._generate_apple()
        
        if self.render_mode == 'human':
            self.img = np.zeros((self.grid_size, self.grid_size, 3), dtype='uint8')
            self._update_ui()
        
        observation = self._get_observation()
        return observation, {}

    def _generate_apple(self):
        """Generate apple at valid position"""
        max_tries = 100
        for _ in range(max_tries):
            apple_pos = [
                random.randint(0, self.grid_cells - 1) * self.cell_size,
                random.randint(0, self.grid_cells - 1) * self.cell_size
            ]
            if apple_pos not in self.snake_position:
                return apple_pos
        
        return [
            random.randint(0, self.grid_cells - 1) * self.cell_size,
            random.randint(0, self.grid_cells - 1) * self.cell_size
        ]

    def render(self):
        if self.render_mode == 'human':
            cv2.imshow('Snake Game', self.img)
            cv2.waitKey(1)
        elif self.render_mode == 'rgb_array':
            return self.img

    def close(self):
        if self.render_mode == 'human':
            cv2.destroyAllWindows()

    def _update_ui(self):
        """Update visualization"""
        self.img = np.zeros((self.grid_size, self.grid_size, 3), dtype='uint8')
        
        # Draw apple (red)
        cv2.rectangle(
            self.img,
            tuple(self.apple_position),
            (self.apple_position[0] + self.cell_size, 
             self.apple_position[1] + self.cell_size),
            (0, 0, 255), -1
        )
        
        # Draw snake
        for i, pos in enumerate(self.snake_position):
            brightness = max(100, 255 - i * 5)
            cv2.rectangle(
                self.img,
                tuple(pos),
                (pos[0] + self.cell_size, pos[1] + self.cell_size),
                (0, brightness, 0), -1
            )
        
        # Draw score
        cv2.putText(
            self.img,
            f'Score: {self.score}',
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
