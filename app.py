import pygame
import sys
from functions import*
from principal import*
import pygame
from datetime import datetime
import calendar
import os
import re
import numpy as np
from collections import defaultdict




#Directorios
lunar = './data/lunar/test/data/S12_GradeB'
lunar_2 = './data/lunar/test/data/S15_GradeA'
lunar_3 = './data/lunar/test/data/S15_GradeB'
lunar_4 = './data/lunar/test/data/S16_GradeA'
lunar_5 = './data/lunar/test/data/S16_GradeB'
mars = './data/mars/test/data'
earth = './data/EARTH'

dir_moon = [lunar, lunar_2, lunar_3, lunar_4, lunar_5]
dir_mars = [mars]
dir_earth = [earth]

dates_lunar = {}
dates_lunar2 = {}
dates_lunar3 = {}
dates_lunar4 = {}
dates_lunar5 = {}
dates_mars = {}
dates_earth = {}

dates_moon_list = [dates_lunar, dates_lunar2, dates_lunar3, dates_lunar4, dates_lunar5]
dates_mars_list = [dates_mars]
dates_earth_list = [dates_earth]


# Inicializar Pygame
pygame.init()
window_w = 1440
window_h = 720


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
FONT = pygame.font.SysFont('suravaram', 36)
WHITE = (255, 255, 255)
BLACK = (27, 79, 114)
BLUE = (174, 214, 241)
RED = (52, 152, 219)
GREEN = (93, 173, 226)



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

font3 = pygame.font.SysFont('suravaram', 41)
text3 = font3.render('Choose your favorite object', True, color_font)  # Blanco
text_rect3 = text.get_rect()



Fondo_tierra = pygame.image.load('./img/earth_image.jpg')
Fondo_marte = pygame.image.load('./img/mars_image.png')
Fondo_Luna = pygame.image.load('./img/moon.png')



# Centrar el rectángulo en la pantalla
text_rect.center = (window_w // 2, window_h // 6)
text_rect2.center = (window_w // 2, window_h // 6)
text_rect3.center = ((window_w // 2)+ 30, (window_h // 3) +40)





def display_main_menu(screen):
    screen.fill(negro)
    screen.blit(fondo,(0,0))
    screen.blit(text,text_rect)
    screen.blit(text_contour,text_rect2)
    screen.blit(text3,text_rect3)
    

# Función para crear un botón
def crear_boton_circular(texto, x, y, radio, color_inactivo, color_activo, icon, size, accion=None, fondo_ventana = None, Dir = None, date = None):
    mouse = pygame.mouse.get_pos()  # Obtener la posición del mouse
    click = pygame.mouse.get_pressed()  # Obtener el estado de los clics del mouse

    # Calcular la distancia entre el mouse y el centro del círculo
    distancia = np.sqrt((mouse[0] - x) ** 2 + (mouse[1] - y) ** 2)

    # Verificar si el mouse está dentro del círculo
    if distancia < radio:
        pygame.draw.circle(screen, color_activo, (x, y), radio)  # Cambia el color al pasar el cursor
        screen.blit(icon,(x-(size//2),y-(size//2)))
        if click[0] == 1 and accion is not None:  # Si se hace clic, realiza la acción
            accion(fondo_ventana,Dir,date)
    else:
        pygame.draw.circle(screen, color_inactivo, (x, y), radio)  # Color normal del botón
        screen.blit(icon,(x-(size//2),y-(size//2)))
    # Renderizar el texto del botón
    text = font.render(texto, True, color_font)
    text_rect = text.get_rect(center=(x, y))
    screen.blit(text, text_rect)  # Dibujar el texto en el botón




# Función para extraer los años
def get_unique_years(eventos):
    years = set()
    for date_str in eventos.keys():
        year = date_str.split('-')[0]
        years.add(year)
    return sorted(list(years))

# Función para obtener los meses del año seleccionado
def get_months_for_year(eventos, year):
    months = set()
    for date_str in eventos.keys():
        if date_str.startswith(year):
            month = int(date_str.split('-')[1])
            months.add(month)
    return sorted(list(months))

# Función para obtener los días del mes seleccionado
def get_days_for_month(eventos, year, month):
    days = []
    for date_str, filepaths in eventos.items():
        y, m, d = map(int, date_str.split('-'))
        if y == int(year) and m == int(month):
            days.append((d, filepaths))
    return sorted(days)

# Dibujar botones con esquinas redondeadas y texto centrado
def draw_buttons(screen, options, pos_x, pos_y, width, height, columns=1):
    buttons = []
    for i, option in enumerate(options):
        # Cálculo de la posición en base a las columnas
        column = i % columns
        row = i // columns
        x_offset = pos_x + column * (width + 10)
        y_offset = pos_y + row * (height + 10)
        
        # Definir el rectángulo del botón
        rect = pygame.Rect(x_offset, y_offset, width, height)
        
        # Dibujar rectángulo con esquinas redondeadas
        pygame.draw.rect(screen, BLUE, rect, border_radius=15)
        
        # Obtener el tamaño del texto y centrarlo
        text_surface = FONT.render(str(option), True, WHITE)
        text_rect = text_surface.get_rect(center=rect.center)
        
        # Dibujar el texto centrado en el botón
        screen.blit(text_surface, text_rect)
        
        buttons.append((rect, option))  # Devolvemos la tupla (rect, opción)
    return buttons

# Dibujar flecha de "Volver"
def draw_back_button(screen):
    back_button_rect = pygame.Rect(50, 50, 50, 50)
    pygame.draw.rect(screen, GREEN, back_button_rect, border_radius=15)
    
    # Definir los puntos para dibujar una flecha que apunta a la izquierda
    arrow_color = WHITE
    arrow_points = [(back_button_rect.x + 10, back_button_rect.y + 25),  # Punta
                    (back_button_rect.x + 30, back_button_rect.y + 10),  # Arriba
                    (back_button_rect.x + 30, back_button_rect.y + 40)]  # Abajo
    
    pygame.draw.polygon(screen, arrow_color, arrow_points)  # Dibujar la flecha
    
    return back_button_rect

def draw_event_info(screen, date, filepath, image):
    # Cuadro de información a la derecha del calendario
    info_rect = pygame.Rect(680, 100, 750, 550)  # Ajustar el tamaño del cuadro de información
    pygame.draw.rect(screen, BLACK, info_rect, border_radius=15)
    
    # Mostrar la fecha
    text_surface = FONT.render(f"Fecha: {date}", True, WHITE)
    screen.blit(text_surface, (info_rect.x + 10, info_rect.y + 10))
    
    # Mostrar el filepath (primer archivo)
    filepath_surface = FONT.render(f"Archivo:", True, WHITE)
    screen.blit(filepath_surface, (info_rect.x + 10, info_rect.y + 50))
    
    # Dividir el filepath si es muy largo
    first_file = filepath[0]  # Corregido: seleccionar el primer archivo
    lines = first_file.split('/')  # Convertir en lista de strings por cada subdirectorio
    for i, line in enumerate(lines):
        filepath_line_surface = FONT.render(line, True, WHITE)
        screen.blit(filepath_line_surface, (info_rect.x + 10, info_rect.y + 80 + i * 30))
    
    # Mostrar la imagen (escalar si es necesario)
    if image:
        image = pygame.transform.scale(image, (200, 200))  # Redimensionar imagen
        screen.blit(image, (info_rect.x + 10, info_rect.y + 300))  # Mostrar la imagen

# Función para cargar una imagen desde un archivo
def load_image(filepath):
    try:
        image = pygame.image.load(filepath)
        print(f"Imagen cargada correctamente desde: {filepath}")
        return image
    except pygame.error as e:
        print(f"No se pudo cargar la imagen: {filepath}. Error: {e}")
        return None
# Función para correr el evento
def run_event(filepath):
    print(f"Ejecutando función para el archivo: {filepath}")
    
    
def window(fondo=0,Dir=0, date = 0):

    eventos = dictionary_name(dir_name = Dir, dates_name_list = date)
    SCREEN_WIDTH = 1440
    SCREEN_HEIGHT = 720
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    current_year = None
    current_month = None
    selected_event = None  # Inicializamos selected_event como None
    running = True

    fondo = pygame.transform.scale(fondo, (SCREEN_WIDTH, SCREEN_HEIGHT))


    while running:
        screen.blit(fondo,(0,0))
        back_button_rect = draw_back_button(screen)
        if current_year is None:
            # Mostrar botones con los años
            years = get_unique_years(eventos)
            # Centrar los botones y limitar a 2 columnas si hay más de 8 años
            if (len(years)>8) and (len(years)<18):
                columns=2
            elif len(years)>=18:
                columns=3
            else:
                columns=1
            year_buttons = draw_buttons(screen, years, (SCREEN_WIDTH - 200 * columns) // 2, 50, 200, 50, columns)
        elif current_month is None:
            text_surface = FONT.render(f"{current_year}", True, WHITE)
            screen.blit(text_surface, (SCREEN_WIDTH // 2 - text_surface.get_width() // 2, 20))
            # Mostrar botón de "Volver"
            back_button_rect = draw_back_button(screen)
            
            # Mostrar botones con los meses para el año seleccionado
            months = get_months_for_year(eventos, current_year)
            month_names = [calendar.month_name[month] for month in months]  # Mapeo de números a nombres de mes
            columns = 2 if len(month_names) > 8 else 1
            month_buttons = draw_buttons(screen, month_names, (SCREEN_WIDTH - 200 * columns) // 2, 100, 200, 50, columns)
        else:
            text_surface = FONT.render(f"{calendar.month_name[current_month]}", True, WHITE)
            screen.blit(text_surface, (SCREEN_WIDTH // 4 - text_surface.get_width() // 2, 20))
            # Mostrar botón de "Volver"
            back_button_rect = draw_back_button(screen)
            
            # Mostrar calendario con días de eventos
            days = get_days_for_month(eventos, current_year, current_month)
            cal = calendar.monthcalendar(int(current_year), int(current_month))
            
            # Dibujar calendario
            for week_index, week in enumerate(cal):
                for day_index, day in enumerate(week):
                    if day != 0:
                        rect = pygame.Rect(100 + day_index * 80, 120 + week_index * 80, 70, 70)
                        # Verificar si el día tiene eventos
                        if any(day == d for d, _ in days):
                            pygame.draw.rect(screen, RED, rect, border_radius=15)  # Días con eventos en rojo
                        else:
                            pygame.draw.rect(screen, BLACK, rect, border_radius=15)  # Días sin eventos en negro
                        text_surface = FONT.render(str(day), True, WHITE)
                        text_rect = text_surface.get_rect(center=rect.center)
                        screen.blit(text_surface, text_rect)
            
            # Mostrar cuadro de evento si hay uno seleccionado
        if selected_event:
            draw_event_info(screen, selected_event['date'], selected_event['filepath'], selected_event['image'])
        
            # Manejo de eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if back_button_rect.collidepoint(event.pos) and current_year is None and current_month is None:
                    # Si el botón "Back" es presionado, salir de la función para volver al menú de planetas
                    return  # Esto te devuelve al bucle principal, mostrando el menú de selección de planetas
                if current_year is None:
                    for rect, year in year_buttons:
                        if rect.collidepoint(event.pos):
                            current_year = year
                            break
                elif current_month is None:
                    if back_button_rect.collidepoint(event.pos):
                        selected_event = None
                        current_year = None  # Volver a la selección de años
                    else:
                        for rect, month_name in month_buttons:
                            if rect.collidepoint(event.pos):
                                current_month = months[month_names.index(month_name)]
                                break
                else:
                    if back_button_rect.collidepoint(event.pos):
                        current_month = None  # Volver a la selección de meses
                        selected_event = None
                    else:
                        for day, filepaths in days:
                            for week_index, week in enumerate(cal):
                                for day_index, cal_day in enumerate(week):
                                    if cal_day == day:  # Solo para días con eventos
                                        rect = pygame.Rect(100 + day_index * 80, 150 + week_index * 80, 70, 70)
                                        if rect.collidepoint(event.pos):
                                            ##############MAIIIIIIIIIIIIIIIIIIIIIIIIIIN############
                                            selected_event = {
                                                'date': f"{current_year}-{current_month:02}-{day:02}",
                                                'filepath': filepaths[0],  # Mostrar solo el primer archivo
                                                'image': load_image('./img/moon.png')  # Reemplaza con la ruta a tu imagen
                                            }
                                            break
            
        pygame.display.flip()
    return True
    

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
    crear_boton_circular('Moon', window_w//4, 500, 170,color_fondo, color_moon, moon_button_image, 360, window,Fondo_Luna,dir_moon,dates_moon_list)
    crear_boton_circular('Mars',  2*window_w//4, 500, 170,color_fondo, color_mars, mars_button_image,300, window,Fondo_marte,dir_mars,dates_mars_list)
    crear_boton_circular('Earth', 3*window_w//4, 500, 170,color_fondo, color_earth, earth_button_image,340, window,Fondo_tierra,dir_earth,dates_earth_list)
    # Actualizar pantalla
    pygame.display.flip()

    # Control de FPS
    pygame.time.Clock().tick(60)
