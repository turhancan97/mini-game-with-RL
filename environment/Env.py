# pygame template - skeleton for a new pygame project
import pygame
import os
import numpy as np
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from .Player import Player
from .Enemy import Enemy
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.DQLAgent import DQLAgent, CNNDQLAgent

# window size
WIDTH = 360
HEIGHT = 360
multiply = 4
FPS = 60 * multiply  # how fast game is

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # RGB
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Env_enemy(pygame.sprite.Sprite):
    def __init__(self, TRAIN=False, scenario="eat", num_enemies=3, enemy_speed=3, max_steps=None, headless=False, obs_type="vector"):
        pygame.sprite.Sprite.__init__(self)
        # scenario can be: "eat" (catch objects) or "avoid" (avoid collisions)
        self.scenario = scenario.lower()
        if self.scenario not in ("eat", "avoid"):
            raise ValueError("scenario must be 'eat' or 'avoid'")

        self.all_sprite = pygame.sprite.Group()
        self.enemy_group = pygame.sprite.Group()
        self.player = Player(WIDTH, HEIGHT, BLUE, RED)
        self.all_sprite.add(self.player)
        # unified number of enemies (can be overridden)
        self.num_enemies = int(num_enemies)
        self.enemy_speed = int(enemy_speed)
        self.enemies = [Enemy(WIDTH, HEIGHT, RED, WHITE, speed=self.enemy_speed) for _ in range(self.num_enemies)]
        for enemy in self.enemies:
            self.all_sprite.add(enemy)
            self.enemy_group.add(enemy)
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.enemy_missed = False
        self.max_steps = max_steps  # None means no cap
        self.headless = headless
        self.obs_type = obs_type
        if self.obs_type == "pixels":
            self.agent = CNNDQLAgent(self.num_enemies)
            self.frame_stack = deque(maxlen=4)
        else:
            self.agent = DQLAgent(self.num_enemies)

        self.TRAIN = TRAIN  # Set to True if you want to train, otherwise False to just run the game
        if not self.TRAIN:
            if self.scenario == "eat":
                model_path = "model/model_pixels_eat.h5" if self.obs_type == "pixels" else "model/model_vector_eat.h5"
            elif self.scenario == "avoid":
                model_path = "model/model_pixels_avoid.h5" if self.obs_type == "pixels" else "model/model_vector_avoid.h5"
            self.agent.load_model(model_path)
            self.agent.epsilon = 0  # Set exploration rate to 0 to always choose the best action

    def findDistance(self, a, b):
        d = a - b
        return d

    def _get_frame(self, surface):
        arr = pygame.surfarray.array3d(surface)  # (W,H,3)
        arr = np.transpose(arr, (1, 0, 2))      # (H,W,3)
        gray = rgb2gray(arr)                    # (H,W) float [0,1]
        small = resize(gray, (84, 84), anti_aliasing=True)
        return small.astype(np.float32)

    def _get_pixel_state(self, surface):
        frame = self._get_frame(surface)
        if len(self.frame_stack) == 0:
            for _ in range(4):
                self.frame_stack.append(frame)
        else:
            self.frame_stack.append(frame)
        stacked = np.stack(list(self.frame_stack), axis=-1)  # (84,84,4)
        return np.expand_dims(stacked, axis=0)

    def step(self, action, surface=None):
        # get coordinate
        state_list = []

        # update
        self.player.update(action)
        missed_any = False
        for enemy in self.enemies:
            missed = enemy.update()
            # Track if any enemy has passed the player (Eat scenario uses this)
            missed_any = missed_any or missed
        self.enemy_missed = missed_any

        if self.obs_type == "pixels":
            # render to surface before capturing
            if surface is None:
                surface = pygame.Surface((WIDTH, HEIGHT))
            surface.fill(GREEN)
            self.all_sprite.draw(surface)
            return self._get_pixel_state(surface)
        else:
            player_coords = self.player.getCoordinates()
            for enemy in self.enemies:
                enemy_coords = enemy.getCoordinates()
                # normalized distance features per enemy: (Δx/WIDTH, Δy/HEIGHT)
                state_list.append(self.findDistance(player_coords[0], enemy_coords[0]) / WIDTH)
                state_list.append(self.findDistance(player_coords[1], enemy_coords[1]) / HEIGHT)
            return [state_list]

    # reset
    def initialStates(self):
        self.all_sprite = pygame.sprite.Group()
        self.enemy_group = pygame.sprite.Group()
        self.player = Player(WIDTH, HEIGHT, BLUE, RED)
        self.all_sprite.add(self.player)
        self.enemies = [Enemy(WIDTH, HEIGHT, RED, WHITE, speed=self.enemy_speed) for _ in range(self.num_enemies)]
        for enemy in self.enemies:
            self.all_sprite.add(enemy)
            self.enemy_group.add(enemy)

        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.enemy_missed = False

        if self.obs_type == "pixels":
            temp_surface = pygame.Surface((WIDTH, HEIGHT))
            temp_surface.fill(GREEN)
            self.all_sprite.draw(temp_surface)
            return self._get_pixel_state(temp_surface)
        else:
            state_list = []
            player_coords = self.player.getCoordinates()
            for enemy in self.enemies:
                enemy_coords = enemy.getCoordinates()
                state_list.append(self.findDistance(player_coords[0], enemy_coords[0]) / WIDTH)
                state_list.append(self.findDistance(player_coords[1], enemy_coords[1]) / HEIGHT)
            return [state_list]

    # euclidean distance (used by 'avoid' scenario)
    def euclideanDistance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def run(self):
        # initialize pygame and create window
        if self.headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        if self.headless:
            screen = pygame.Surface((WIDTH, HEIGHT))
        else:
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("RL-Game")
        clock = pygame.time.Clock()

        # game loop
        state = self.initialStates()
        running = True
        batch_size = 128  # How many experiences to use for each training step. 24 is the first value.

        # Initialize variables for 'avoid' scenario
        survival_reward = 0.001  # constant per-step reward for surviving (small magnitude)
        near_miss_threshold = 50
        max_penalty = -0.25

        steps = 0
        while running:
            # Default reward per step depends on scenario
            if self.scenario == "eat":
                self.reward = -0.01  # small step cost to encourage catching
            else:
                # In 'avoid', use a constant per-step survival reward; near-miss penalty computed below
                self.reward = 0.0
            # keep loop running at the right speed (or run as fast as possible in headless)
            if not self.headless:
                clock.tick(FPS)
            # process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # update
            action = self.agent.act(state)
            next_state = self.step(action, surface=screen if self.obs_type == "pixels" else None)
            steps += 1
            # Scenario-specific rewards and terminations
            hits = pygame.sprite.spritecollide(self.player, self.enemy_group, False, pygame.sprite.collide_circle)

            if self.scenario == "eat":
                # Reward for catching enemies; continue episode
                if hits:
                    self.reward = 1.0
                    # Respawn enemies after a catch
                    self.all_sprite = pygame.sprite.Group()
                    self.enemy_group = pygame.sprite.Group()
                    self.all_sprite.add(self.player)
                    self.enemies = [Enemy(WIDTH, HEIGHT, RED, WHITE, speed=self.enemy_speed) for _ in range(self.num_enemies)]
                    for enemy in self.enemies:
                        self.all_sprite.add(enemy)
                        self.enemy_group.add(enemy)

                # If any enemy was missed (went off screen), end episode with penalty
                if self.enemy_missed:
                    self.reward = -1.0
                    self.done = True
                    running = False
                    print("Missed an enemy! Total reward: ", self.total_reward + self.reward)

            else:  # avoid
                # near-miss penalty based on closest enemies
                near_miss_penalty = 0.0
                player_coords = self.player.getCoordinates()
                for enemy in self.enemies:
                    enemy_coords = enemy.getCoordinates()
                    distance = self.euclideanDistance(player_coords[0], player_coords[1], enemy_coords[0], enemy_coords[1])
                    if distance < near_miss_threshold:
                        # Closer => more negative penalty
                        near_miss_penalty += (max_penalty * (1 - distance / near_miss_threshold))
                # Apply shaping: constant survival reward + near-miss penalty
                self.reward += survival_reward + near_miss_penalty

                # Collision ends the episode with a large penalty
                if hits:
                    self.reward = -5.0
                    self.done = True
                    running = False
                    print("Collision! Total reward: ", self.total_reward + self.reward)

            # Accumulate total reward after scenario logic
            self.total_reward += self.reward

            # Optional max-steps termination
            if self.max_steps is not None and steps >= self.max_steps:
                self.done = True
                running = False
                print(f"Step cap reached ({self.max_steps}). Total reward: ", self.total_reward)

            # Only store experiences and train if TRAIN is True
            if self.TRAIN:
                # storage
                self.agent.remember(state, action, self.reward, next_state, self.done)
                # training
                self.agent.replay(batch_size)

            # update state
            state = next_state

            # epsilon is decayed once per episode in main.py

            # draw / render(show)
            if self.headless:
                # For pixel observations, keep rendering onto the surface for capture
                if self.obs_type == "pixels":
                    screen.fill(GREEN)
                    self.all_sprite.draw(screen)
            else:
                screen.fill(GREEN)
                self.all_sprite.draw(screen)
                pygame.display.flip()

            # print("Reward: ", self.reward)

        pygame.quit()
