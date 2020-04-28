import pygame
import random
import math

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CRAZY CANNON")
# background
bg = pygame.image.load('sky2.png')
# Score
score_val = 0

font = pygame.font.Font('freesansbold.ttf', 32)
over_font = pygame.font.Font('freesansbold.ttf', 64)


def game_over_text():
    over_text = over_font.render("GAME OVER!", True, (255, 255, 255))
    screen.blit(over_text, (200, 250))


def show_score(x, y):
    # true to say it should be shown on the screen
    # render then blit
    score = font.render("Score :" + str(score_val), True, (255, 255, 255))
    screen.blit(score, (x, y))


# Enemy
enemyX1 = 20
enemyX2 = 780
enemyY1 = 20
enemyY2 = 10
enemy_change1 = 0
enemy_change2 = 0

velx1 = 30
vely1 = 0
velx2 = -0.255
vely2 = 0
t1 = 0
t2 = 0
a1  = 1.5
a2 = 1.75
# def initialiseRad():
rad1 = 70
rad2 = 70


def enemy(x, y, rad):
    pygame.draw.circle(screen, (255, 255, 0), (x, y), rad)


# Player
playerX = 370
playerY = 480
playerImg = pygame.image.load('nuke.png')
player_change = 0


def player(x, y):
    screen.blit(playerImg, (x, y))


# Bullet
bulletX = 0
bulletY = 480
bullet_change = 0
bullet_state = "ready"

bulletImg = pygame.image.load('bullet.png')


def fire_bullet(x, y):
    screen.blit(bulletImg, (x, y))
    # pygame.display.update()


# Collision
def iscollision(ex, ey, bx, by):
    dis = math.sqrt((math.pow(ex - bx, 2)) + (math.pow(ey - by, 2)))
    if dis <= 35:
        return True
    else:
        return False


# Redraw the screen
def redraw(x, y):
    screen.fill((0, 0, 0))
    screen.blit(bg, (0, 0))
    screen.blit(playerImg, (x, y))
    pygame.display.update()


# Main Game Loop
flag = True
while flag:
    redraw(playerX, playerY)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            flag = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                player_change = -7
            if event.key == pygame.K_RIGHT:
                player_change = 7
            if event.key == pygame.K_SPACE:
                bullet_state = "fire"
                bullet_change = 3
                bulletX = playerX
                fire_bullet(bulletX, bulletY)

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                player_change = 0

    # Player movement
    playerX += player_change
    if playerX > 738:
        playerX = 738
    if playerX <= 16:
        playerX = 16
    # Enemy movement
    # E1
    if enemyY1 < 600 - rad1:
        t1+= 0.05
        enemyX1 = velx1 * t1
        enemyY1 = 0.5 * (a1 * math.pow(t1, 2))
    # Gameover condition
    if enemyY1 > 500:
        enemyY1 = 2000
        enemyY2 = 2000
        game_over_text()
    # E2
    if enemyY2 < 600 - rad2:
        t2 += .077
        enemyX2 = enemyX2 + velx2*t2
        enemyY2 = (a2 *math.pow(t2 ,2))*0.5
    if enemyY2 > 500:
        enemyY2 = 2000
        enemyY1 = 2000
        game_over_text()



    # Bullet movement
    if bullet_state == "fire":
        bulletY -= bullet_change
        fire_bullet(bulletX, bulletY)
    if bulletY <= 0:
        bulletY = 480
        bullet_state = "ready"

    # Collision
    if iscollision(enemyX1, enemyY1, bulletX, bulletY):
        score_val += 1
        # Bullet
        bullet_state = "ready"
        bulletY = 480
        # Enemy radius changes
        rad1 = rad1 - 20
        if rad1 <= 30:
            rad1 = 70
            enemyY1 = 20
            enemyX1 = 20
            t1 = 0

    if iscollision(enemyX2, enemyY2, bulletX, bulletY):
        score_val += 1
        # Bullet
        bullet_state = "ready"
        bulletY = 480
        # Enemy radius changes
        rad2 = rad2 - 20
        if rad2 <= 30:
            rad2 = 70
            enemyY2 = 20
            enemyX2 = 780
            t2 = 0
    enemy(int(enemyX1), int(enemyY1), rad1)
    enemy(int(enemyX2), int(enemyY2), rad2)
    player(playerX, playerY)
    show_score(10, 10)
    pygame.display.update()
