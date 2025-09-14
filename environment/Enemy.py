import pygame
import random

class Enemy(pygame.sprite.Sprite):
    def __init__(self, WIDTH, HEIGHT, RED, WHITE, speed=3):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.RED = RED
        self.WHITE = WHITE
        self.base_speedy = speed

        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10, 10))
        self.image.fill(self.RED)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image, self.WHITE, self.rect.center, self.radius)
        self.rect.x = random.randrange(0, self.WIDTH - self.rect.width)
        self.rect.y = random.randrange(2, 6)

        self.speedx = 0
        self.speedy = self.base_speedy

    def update(self):
        self.rect.x += self.speedx
        self.rect.y += self.speedy

        if self.rect.top > self.HEIGHT + 10:
            self.rect.x = random.randrange(0, self.WIDTH - self.rect.width)
            self.rect.y = random.randrange(2, 6)
            self.speedy = self.base_speedy
            return True  # Signify that the enemy has passed the player
        return False

    def getCoordinates(self):
        return (self.rect.x, self.rect.y)
