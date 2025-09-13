# pygame template - skeleton for a new pygame project
import pygame
from Player import Player
from Enemy import Enemy
from DQLAgent import DQLAgent

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
    def __init__(self, TRAIN=False, scenario="eat", num_enemies=3, enemy_speed=3):
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
        self.agent = DQLAgent(self.num_enemies)

        self.TRAIN = TRAIN  # Set to True if you want to train, otherwise False to just run the game
        if not self.TRAIN:
            self.agent.load_model("model/model.h5")
            self.agent.epsilon = 0  # Set exploration rate to 0 to always choose the best action

    def findDistance(self, a, b):
        d = a - b
        return d

    def step(self, action):
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

        player_coords = self.player.getCoordinates()
        for enemy in self.enemies:
            enemy_coords = enemy.getCoordinates()
            # distance [(playerx-m1x),(playery-m1y),(playerx-m2x),(playery-m2y), (playerx-m3x),(playery-m3y)]
            state_list.append(self.findDistance(player_coords[0], enemy_coords[0]))
            state_list.append(self.findDistance(player_coords[1], enemy_coords[1]))

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

        state_list = []

        player_coords = self.player.getCoordinates()
        for enemy in self.enemies:
            enemy_coords = enemy.getCoordinates()
            state_list.append(self.findDistance(player_coords[0], enemy_coords[0]))
            state_list.append(self.findDistance(player_coords[1], enemy_coords[1]))

        return [state_list]

    # euclidean distance (used by 'avoid' scenario)
    def euclideanDistance(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def run(self):
        # initialize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("RL-Game")
        clock = pygame.time.Clock()

        # game loop
        state = self.initialStates()
        running = True
        batch_size = 64  # How many experiences to use for each training step. 24 is the first value.

        # Initialize variables for 'avoid' scenario
        time_based_reward = 0.0
        time_increment = 0.01
        near_miss_threshold = 50
        max_penalty = -50

        while running:
            # Default reward per step depends on scenario
            if self.scenario == "eat":
                self.reward = -0.1  # small step cost to encourage catching
            else:
                # In 'avoid', compute time-based shaping; near-miss penalty computed below
                time_based_reward += time_increment
                self.reward = 0.0
            # keep loop running at the right speed
            clock.tick(FPS)
            # process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            # update
            action = self.agent.act(state)
            next_state = self.step(action)
            # Scenario-specific rewards and terminations
            hits = pygame.sprite.spritecollide(self.player, self.enemy_group, False, pygame.sprite.collide_circle)

            if self.scenario == "eat":
                # Reward for catching enemies; continue episode
                if hits:
                    self.reward = 100
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
                    self.reward = -100
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
                # Apply shaping
                self.reward += time_based_reward + near_miss_penalty

                # Collision ends the episode with a large penalty
                if hits:
                    self.reward = -500
                    self.done = True
                    running = False
                    print("Collision! Total reward: ", self.total_reward + self.reward)

            # Accumulate total reward after scenario logic
            self.total_reward += self.reward

            # Only store experiences and train if TRAIN is True
            if self.TRAIN:
                # storage
                self.agent.remember(state, action, self.reward, next_state, self.done)
                # training
                self.agent.replay(batch_size)

            # update state
            state = next_state

            # epsilon greedy
            self.agent.adaptiveEGreedy()

            # draw / render(show)
            screen.fill(GREEN)
            self.all_sprite.draw(screen)
            # after drawing flip display
            pygame.display.flip()

            # print("Reward: ", self.reward)

        pygame.quit()
