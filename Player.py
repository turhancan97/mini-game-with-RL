import pygame

class Player(pygame.sprite.Sprite):
    # sprite for the player
    def __init__(self, WIDTH, HEIGHT, BLUE, RED):
        pygame.sprite.Sprite.__init__(self)
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.BLUE = BLUE
        self.RED = RED

        self.image = pygame.Surface((20, 20))
        self.image.fill(self.BLUE)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, self.RED, self.rect.center, self.radius)
        self.rect.centerx = self.WIDTH / 2
        self.rect.bottom = self.HEIGHT - 1
        self.speedx = 0

    def update(self, action):
        self.speedx = 0
        keystate = pygame.key.get_pressed()

        if keystate[pygame.K_LEFT] or action == 0:
            self.speedx = -4
        elif keystate[pygame.K_RIGHT] or action == 1:
            self.speedx = 4
        else:
            self.speedx = 0

        self.rect.x += self.speedx

        if self.rect.right > self.WIDTH:
            self.rect.right = self.WIDTH
        if self.rect.left < 0:
            self.rect.left = 0

    def getCoordinates(self):
        return (self.rect.x, self.rect.y)
