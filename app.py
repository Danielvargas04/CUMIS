import pygame
import sys

# Inicializar Pygame
pygame.init()

window_height = 1440
window_width = 720
# Configurar pantalla
screen = pygame.display.set_mode((window_height, window_width))
pygame.display.set_caption('Seismic Detection')


# Definir colores
blanco = (255, 255, 255)
negro = (0, 0, 0)

# Cargar una imagen
fondo = pygame.image.load('./img/earth_image.jpg')

fondo = pygame.transform.scale(fondo, (window_height, window_width))

# Posición inicial del jugador
x, y = 320, 240

# Bucle principal
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    # Dibujar en la pantalla
    screen.fill(negro)
    screen.blit(fondo,(0,0))

    #mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()
    print(mouse_x, mouse_y)

    # Actualizar pantalla
    pygame.display.flip()

    # Control de FPS
    pygame.time.Clock().tick(60)

