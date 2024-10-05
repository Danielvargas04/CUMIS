import pygame
import sys
import numpy as np

# Inicializar Pygame
pygame.init()
window_w = 1440
window_h = 720

#global
Name = ""

# Configurar pantalla
screen = pygame.display.set_mode((window_w, window_h))
pygame.display.set_caption('Seismic Detection')


# Definir colores
blanco = (255, 255, 255)
negro = (0, 0, 0)
color_fondo  = (0, 51, 51)
color_font = (255, 255, 153)
color_mars = (234,79,45)
color_earth = (45,80,234)
color_moon = (242,253,108)

# MENU
fondo = pygame.image.load('./img/main_image.jpg')

fondo = pygame.transform.scale(fondo, (window_w, window_h))

moon_button_image = pygame.image.load('./img/moon_button.png')
moon_button_image =pygame.transform.scale(moon_button_image, (360, 360))

mars_button_image = pygame.image.load('./img/mars_button.png')
mars_button_image =pygame.transform.scale(mars_button_image, (300, 300))

earth_button_image = pygame.image.load('./img/earth_button.png')
earth_button_image =pygame.transform.scale(earth_button_image, (340, 340))

font = pygame.font.SysFont('suravaram', 71)
text = font.render('Seismic Detection', True, color_fondo)  # Blanco
text_rect = text.get_rect()

font2 = pygame.font.SysFont('suravaram', 70)
text_contour = font2.render('Seismic Detection', True, color_font)  # Blanco
text_rect2 = text.get_rect()

# Centrar el rectángulo en la pantalla
text_rect.center = (window_w // 2, window_h // 6)
text_rect2.center = (window_w // 2, window_h // 6)


def display_main_menu(screen):
    screen.fill(negro)
    screen.blit(fondo,(0,0))
    screen.blit(text,text_rect)
    screen.blit(text_contour,text_rect2)

# Función para crear un botón
def crear_boton_circular(texto, x, y, radio, color_inactivo, color_activo, icon, size, accion=None):
    mouse = pygame.mouse.get_pos()  # Obtener la posición del mouse
    click = pygame.mouse.get_pressed()  # Obtener el estado de los clics del mouse

    # Calcular la distancia entre el mouse y el centro del círculo
    distancia = np.sqrt((mouse[0] - x) ** 2 + (mouse[1] - y) ** 2)

    # Verificar si el mouse está dentro del círculo
    if distancia < radio:
        pygame.draw.circle(screen, color_activo, (x, y), radio)  # Cambia el color al pasar el cursor
        screen.blit(icon,(x-(size//2),y-(size//2)))
        if click[0] == 1 and accion is not None:  # Si se hace clic, realiza la acción
            accion()
    else:
        pygame.draw.circle(screen, color_inactivo, (x, y), radio)  # Color normal del botón
        screen.blit(icon,(x-(size//2),y-(size//2)))
    # Renderizar el texto del botón
    text = font.render(texto, True, color_font)
    text_rect = text.get_rect(center=(x, y))
    screen.blit(text, text_rect)  # Dibujar el texto en el botón


# Ejemplo de función de acción
def mensaje1():
    Name = "Moon"
    print("¡Botón presionado!")

def mensaje2():
    print("Boton 2")

def mensaje3():
    print("Boton3")


# Bucle principal
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    # Dibujar en la pantalla
    display_main_menu(screen)
    #mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()

   # print(mouse_x, mouse_y)
    crear_boton_circular('Moon', window_w//4, 500, 170,color_fondo, color_moon, moon_button_image, 360,mensaje1)
    crear_boton_circular('Mars', 2*window_w//4, 500, 170,color_fondo, color_mars, mars_button_image,300, mensaje2)
    crear_boton_circular('Earth', 3*window_w//4, 500, 170,color_fondo, color_earth, earth_button_image,340, mensaje3)
    # Actualizar pantalla
    pygame.display.flip()

    # Control de FPS
    pygame.time.Clock().tick(60)

