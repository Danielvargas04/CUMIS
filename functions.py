from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from obspy import read
from datetime import datetime
import os
from obspy.signal.trigger import classic_sta_lta
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks
from scipy import signal, ndimage
import re
from collections import defaultdict

def data_extrac(N=27, data_directory = './data/lunar/training/data/S12_GradeA/'):
    '''
    Extrae los datos de un archivo N del catalogo y filtra por bandpass
    ademas mas cositas (proximamente)
    '''
    cat_file = './data/lunar/training/catalogs/' + 'apollo12_catalog_GradeA_final.csv'
    cat = pd.read_csv(cat_file)

    #Obtener le ubicacion de los datos de una señal
    row = cat.iloc[N]          #señal N (27)

    test_filename = row.filename
    arrival_time_rel = row['time_rel(sec)']

    #Obtener datos de la señal
    mseed_file = f'{data_directory}{test_filename}.mseed'
    st = read(mseed_file)       #stream file (contains the trace(s))

    tr = st.traces[0].copy()    #Datos de la señal
    tr_times = tr.times()       #en segundos
    tr_data = tr.data

    #Filtro de band pass para frecuencias
    st_filt = st.copy()
    st_filt.filter('bandpass',freqmin=0.5,freqmax=1.0)
    tr_filt = st_filt.traces[0].copy()
    df = tr_filt.stats.sampling_rate
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    #Filtro de normalizacion de velocidades
    tr_data_filt_norm = tr_data_filt.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    tr_data_filt_norm = scaler.fit_transform(tr_data_filt_norm)
    #Filtro de normalizacion de tiempos
    tr_times_norm = tr_times.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    tr_times_norm = scaler.fit_transform(tr_times_norm)

    starttime = tr.stats.starttime.datetime
    arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
    arrival = (arrival_time - starttime).total_seconds()

    return tr_times, tr_data_filt_norm, arrival, df

#---------------------------------------------------

def data_extrac_test(N=1, data_directory = './data/lunar/test/data/S12_GradeB'):
    '''
    Extrae los datos de un archivo N del catalogo y filtra por bandpass
    ademas mas cositas (proximamente)
    '''
    # Lista para almacenar los nombres de los archivos
    path_data = []

    # Recorrer todos los archivos en el directorio
    for nombre_archivo in os.listdir(data_directory):
        # Comprobar si es un archivo (y no una carpeta)
        if os.path.isfile(os.path.join(data_directory, nombre_archivo)) and nombre_archivo.endswith('.mseed'):
            path_data.append(nombre_archivo)

    #print(f"Cantidad de datos en el directorio {data_directory} : {len(path_data)}")
    #Obtener datos de la señal
    mseed_file = data_directory+'/'+path_data[N]
    st = read(mseed_file)       #stream file (contains the trace(s))

    tr = st.traces[0].copy()    #Datos de la señal
    tr_times = tr.times()       #en segundos
    tr_data = tr.data

    #Filtro de band pass para frecuencias
    st_filt = st.copy()
    st_filt.filter('bandpass',freqmin=0.5,freqmax=1.0)
    tr_filt = st_filt.traces[0].copy()
    df = tr_filt.stats.sampling_rate
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    #Filtro de normalizacion de velocidades
    tr_data_filt_norm = tr_data_filt.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    tr_data_filt_norm = scaler.fit_transform(tr_data_filt_norm)

    return tr_times, tr_data_filt_norm, df, tr_data, path_data[N]

def data_path(path):
    #Obtener datos de la señal
    mseed_file = path
    st = read(mseed_file)       #stream file (contains the trace(s))

    tr = st.traces[0].copy()    #Datos de la señal
    tr_times = tr.times()       #en segundos
    tr_data = tr.data

    #Filtro de band pass para frecuencias
    st_filt = st.copy()
    st_filt.filter('bandpass',freqmin=0.5,freqmax=1.0)
    tr_filt = st_filt.traces[0].copy()
    df = tr_filt.stats.sampling_rate
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    #Filtro de normalizacion de velocidades
    tr_data_filt_norm = tr_data_filt.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    tr_data_filt_norm = scaler.fit_transform(tr_data_filt_norm)

    return tr_times, tr_data_filt_norm, df, tr_data_filt, path

def CFT(tr_data_filt_norm, df, sta_len = 60, lta_len = 900):

    #STA/LTA
    # How long should the short-term and long-term window be, in seconds?\
    # sta_len = 120
    # lta_len = 600
    # Run Obspy's STA/LTA to obtain a characteristic function
    # This function basically calculates the ratio of amplitude between the short-term 
    # and long-term windows, moving consecutively in time across the data
    #cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))
    tr_data_filt_norm = tr_data_filt_norm.reshape(-1)
    cft = classic_sta_lta(tr_data_filt_norm, int(sta_len * df), int(lta_len * df))
    return cft

def fourier_filter(cft, cutoff_freq = 0.0001/2):
    # Número de puntos en el arreglo
    n = len(cft)

    # Frecuencia de muestreo (puedes ajustarla dependiendo del contexto)
    sampling_rate = 1 #df  # Ajusta según corresponda

    # Realizar la FFT del arreglo
    cft_fft = fft(cft)

    # Crear las frecuencias correspondientes
    frequencies = fftfreq(n, d=sampling_rate)

    # Definir la frecuencia de corte para el filtro
    #cutoff_freq = 0.0001/2 # Ajusta según lo que necesites

    # Aplicar el filtro pasa bajos: eliminar las frecuencias más altas que la frecuencia de corte
    cft_fft[np.abs(frequencies) > cutoff_freq] = 0

    # Realizar la inversa de la FFT para regresar al dominio temporal
    filtered_cft = ifft(cft_fft)

    # La señal filtrada está en dominio temporal (real)
    filtered_cft_real = np.real(filtered_cft)
    return filtered_cft_real

def peaks_plot(tr_times, filter_cft, prominence=0.3, distance=10000*6.6, wlen_value=8000*6.6, height= 1.3 ):
    # Use find_peaks with prominence, distance, and wlen
    # wlen_value: Set the window length for prominence calculation (adjust as needed)
    peaks, properties = find_peaks(filter_cft, height= height, prominence=prominence, distance=distance, wlen=wlen_value)

    # Plot the signal and the detected peaks
    fig, ax = plt.subplots(1, 1, figsize=(12,3))
    ax.plot(tr_times, filter_cft, label="filter_cft")
    ax.plot(tr_times[peaks], filter_cft[peaks], "x", label="Peaks")

    # Plot the prominence as vertical lines
    for peak, prominence, left_base, right_base in zip(peaks, properties['prominences'], properties['left_bases'], properties['right_bases']):
        # Plot vertical lines from the peak to its prominence baseline
        ax.vlines(x=tr_times[peak], ymin=filter_cft[peak] - prominence, ymax=filter_cft[peak], color="C1", linestyles="--")
        
        # Plot horizontal lines showing the left and right bases of each prominence
        ax.hlines(y=filter_cft[peak] - prominence, xmin=tr_times[left_base], xmax=tr_times[right_base], color="C1", linestyles="--")
    ax.hlines(1, 0, tr_times[-1], color="gray", linestyles="--")
    # Customize the plot
    ax.set_title(f"Peak Detection with Prominence: ({prominence}) (wlen={wlen_value}) Visualization")
    ax.legend()

    # Show the detected peaks

    #print(f"Picos encontrados en los tiempos: {tr_times[peaks]}")
    #print(f"Amplitud del pico (CFT): {filter_cft[peaks]}")

    if len(peaks)!=0:
        # Calcular la confianza en función de la prominencia y la altura
        confidence = properties['prominences'] / np.max(properties['prominences'])  # Normalizar la prominencia
        confidence *= filter_cft[peaks] / np.max(filter_cft[peaks])  # Ajustar por la altura de los picos

        # Mostrar los picos con su confianza
        #for i, peak in enumerate(peaks):
            #print(f"Pico en tiempo {tr_times[peak]} tiene una confianza de [{confidence[i]:.2f}]")
    return peaks, properties

def peaks_from_data(tr_times, filter_cft, prominence=0.3, distance=10000*6.6, wlen_value=8000*6.6, height= 1.3 ):
    # Use find_peaks with prominence, distance, and wlen
    # wlen_value: Set the window length for prominence calculation (adjust as needed)
    peaks, properties = find_peaks(filter_cft, height= height, prominence=prominence, distance=distance, wlen=wlen_value)
    print(f"Picos encontrados en los tiempos: {tr_times[peaks]}")
    print(f"Amplitud del pico (CFT): {filter_cft[peaks]}")
    if len(peaks)!=0:
        # Calcular la confianza en función de la prominencia y la altura
        confidence = properties['prominences'] / np.max(properties['prominences'])  # Normalizar la prominencia
        confidence *= filter_cft[peaks] / np.max(filter_cft[peaks])  # Ajustar por la altura de los picos

        # Mostrar los picos con su confianza
        for i, peak in enumerate(peaks):
            print(f"Pico en tiempo {tr_times[peak]} tiene una confianza de [{confidence[i]:.2f}]")
    return peaks
def get_index_from_f(arrf,fval):
    tol = 1.5e-3
    aux = abs(arrf - fval)
    index = np.where(aux<tol)
    return int(index[0])

def hist_convolve_spectrogram(sxx, Nconvs):
    conv = sxx/sxx.max()
    kernel = np.ones((10, 10))
    for i in range(Nconvs):
        conv = ndimage.convolve(conv, kernel, mode='nearest')
        conv = conv/conv.max()
        
    hist = np.sum(conv/conv.max(), axis=0)
    hist /= hist.max()
    return hist , conv

def Seismic_Plot(tr_times, tr_data_filt_norm, filename, arrivals, confidence):
    # Buscar una fecha en formato YYYY-MM-DD usando una expresión regular
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    
    # Si se encuentra una coincidencia, extraer la fecha
    if date_match:
        date = date_match.group(0)  # Obtener la fecha encontrada
    
    # Encontrar el nombre del conjunto de datos ('mars' en este caso) buscando después de '/data/'
    dataset_match = re.search(r'/data/(\w+)/', filename)
    
    # Si se encuentra una coincidencia, extraer el nombre del conjunto de datos
    if dataset_match:
        dataset_name = dataset_match.group(1)  # Obtener el nombre del conjunto de datos
    # Comprobar si arrivals está vacío
    if len(arrivals) == 0:
        print("No arrivals provided. Plotting the data without vertical lines.")
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(tr_times, tr_data_filt_norm)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Normalized signal')
    ax.set_title(f'{f"{date} from {dataset_name}"}')
    
    if len(arrivals) > 0:
        # Ordenar arrivals y confidence en base a los valores de confidence
        sorted_indices = np.argsort(confidence)[::-1]  # Índices que ordenan confidence de mayor a menor
        sorted_arrivals = np.array(arrivals)[sorted_indices]  # Ordenar arrivals
        sorted_confidence = np.array(confidence)[sorted_indices]  # Ordenar confidence
        
        # Asignar colores de acuerdo a la longitud de arrivals
        if len(arrivals) == 1:
            colors = ['r']  # Solo rojo si hay un valor
        elif len(arrivals) == 2:
            colors = ['r', 'g']  # Rojo y verde si hay dos valores
        else:
            colors = ['r', 'g'] + list(plt.cm.viridis(np.linspace(0, 1, len(arrivals)-2)))  # Rojo, verde, y luego otros colores
        
        # Agregar líneas verticales
        for i in range(len(sorted_arrivals)):
            ax.axvline(x=sorted_arrivals[i], color=colors[i], alpha=sorted_confidence[i], 
                       label=f'Confidence: {sorted_confidence[i]:.2f}')
        ax.legend()
        ax.grid(True)
    
    # Save the plot with a transparent background
    plt.savefig("./img/peak_plots.png", dpi=300)
"""
def Seismic_Plot(tr_times, tr_data, filename, arrivals, confidence):
    # Buscar una fecha en formato YYYY-MM-DD usando una expresión regular
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    
    # Si se encuentra una coincidencia, extraer la fecha
    if date_match:
        date = date_match.group(0)  # Obtener la fecha encontrada
    
    # Encontrar el nombre del conjunto de datos ('mars' en este caso) buscando después de '/data/'
    dataset_match = re.search(r'/data/(\w+)/', filename)
    
    # Si se encuentra una coincidencia, extraer el nombre del conjunto de datos
    if dataset_match:
        dataset_name = dataset_match.group(1)  # Obtener el nombre del conjunto de datos
    
    # Comprobar si arrivals está vacío
    if len(arrivals) == 0:
        print("No arrivals provided. Plotting the data without vertical lines.")
    
    # Crear la figura y el eje, con un tamaño ajustado para que sea más larga en el eje X
    fig, ax = plt.subplots(1, 1, figsize=(4, 8))  # Más largo en el eje X

    # Graficar los datos de la señal
    ax.plot(tr_times, tr_data, color='navy', linewidth=1.5)
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Velocity [m/s]', fontsize=12)
    ax.set_title(f'{f"{date} from {dataset_name}"}', fontsize=14, weight='bold')

    # Ajustar la cuadrícula y mejorar la apariencia general
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f9f9f9')  # Color de fondo suave
    
    if len(arrivals) > 0:
        # Ordenar arrivals y confidence en base a los valores de confidence
        sorted_indices = np.argsort(confidence)[::-1]  # Índices que ordenan confidence de mayor a menor
        sorted_arrivals = np.array(arrivals)[sorted_indices]  # Ordenar arrivals
        sorted_confidence = np.array(confidence)[sorted_indices]  # Ordenar confidence
        
        # Asignar colores de acuerdo a la longitud de arrivals
        if len(arrivals) == 1:
            colors = ['r']  # Solo rojo si hay un valor
        elif len(arrivals) == 2:
            colors = ['r', 'g']  # Rojo y verde si hay dos valores
        else:
            colors = ['r', 'g'] + list(plt.cm.viridis(np.linspace(0, 1, len(arrivals)-2)))  # Rojo, verde, y luego otros colores
        
        # Agregar líneas verticales de llegada
        for i in range(len(sorted_arrivals)):
            ax.axvline(x=sorted_arrivals[i], color=colors[i], alpha=sorted_confidence[i], linestyle='--', 
                       label=f'Confidence: {sorted_confidence[i]:.2f}', linewidth=1.5)
        
        ax.legend(loc='upper right', fontsize=10)
    
    # Ajustar los márgenes para mejor presentación
    plt.tight_layout()

    # Guardar la imagen con alta resolución y fondo blanco
    plt.savefig('./img/peak_plots.pdf', dpi=300, bbox_inches='tight', transparent=False)
"""
def Confidence(tr_times, tr_data, df, peaks, properties, tolerance=0.15, f_min=0.5, f_max=1.0, ):

    time_left_bases = tr_times[properties['left_bases']] ##Left base of the time 
    time_right_bases = tr_times[properties['right_bases']] ##Rigght base of the time

    frequencies, times, sxx = signal.spectrogram(tr_data, df, nperseg=2000, noverlap=1800) ## do spectogram
    low_index = get_index_from_f(frequencies, f_min) ## filter for spectogram
    up_index = get_index_from_f(frequencies, f_max)

    sxx = sxx[low_index:up_index, :] ## spectogram filtered
    frequencies = frequencies[low_index:up_index] ## frequencies filtered

    hist, conv = hist_convolve_spectrogram(sxx, 10) ## histogram of the frequencies

    range_left = near_index(times, time_left_bases) ## left base
    range_right = near_index(times, time_right_bases) ##right base
    amplitude_confidence= np.zeros(len(time_left_bases))

    for i in range(len(time_left_bases)):

        local = hist[range_left[i]:range_right[i]] ## hist values in a local group

        amplitude_local = np.max(local) - np.mean(hist) ## local amplitud 
        amplitude_confidence[i] = amplitude_local 
    
    normalize = np.max(amplitude_confidence)
    amplitude_confidence = amplitude_confidence/normalize ##normalize confidence

    #Filter to tolerance in confidence    
    greater_tolerance = np.where(amplitude_confidence>tolerance)
    Definitive_times = tr_times[peaks[greater_tolerance]]
    Definitive_confidences = amplitude_confidence[amplitude_confidence>0.15]

    return Definitive_times,Definitive_confidences

def near_index(tiempos, valores):
    # Convertimos el array de tiempos a un array de numpy (si no lo es)
    tiempos = np.array(tiempos)
    
    # Inicializamos una lista para almacenar los índices
    indices = []
    
    # Iteramos sobre cada valor en el array de valores
    for valor in valores:
        # Calculamos las diferencias absolutas
        diferencias = np.abs(tiempos - valor)
        # Encontramos el índice del tiempo más cercano para cada valor
        indice = np.argmin(diferencias)
        # Guardamos el índice
        indices.append(indice)
    
    return indices

def work_flow(path, set = 'Moon'):
    tr_times, tr_data_filt_norm, df, tr_data, path = data_path(path)
    #Diccionario de argumentos clave para cada conjunto de datos
    #   0prominence      1distance    2wlen_value    3height       4sta     5lta      6fourier  
    dic_aux = {'Moon': [0.25, 3000, 4000, 1, 60, 900, 5e-05],
                'Earth': [0.2, 50, 80, 1, 5, 40, 0.00022],
                'Mars': [0.3, 50, 500, 1, 15, 100, 0.00022]}
    args = dic_aux[set]
    cft = CFT(tr_data_filt_norm, df, args[4], args[5])
    filtered_cft_real = fourier_filter(cft,args[6] )
    peaks, properties = find_peaks(filtered_cft_real, prominence=args[0], distance=args[1]*df, wlen=args[2]*df, height=args[3])
    times_peaks, confidence = Confidence(tr_times, tr_data , df, peaks, properties, f_min=0.5, f_max=1.0)

    Seismic_Plot(tr_times, tr_data_filt_norm, path, times_peaks, confidence)

def extract_date_from_filename(filename):
    # Expresión regular para buscar una fecha en el formato YYYY-MM-DD
    match = re.search(r'\d{4}-\d{2}-\d{2}', filename)
    if match:
        return match.group(0)
    return None


def find_mseed_files_with_dates(root_dir):
    # Recorrer todas las carpetas y archivos dentro del directorio raíz
    dates_dict = {}
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mseed'):
                filepath = os.path.join(foldername, filename)
                # Extraer la fecha del nombre del archivo
                date = extract_date_from_filename(filename)
                if date in dates_dict:
                        dates_dict[date].append(filepath)
                else:
                    dates_dict[date] = [filepath]

    return dates_dict

def dictionary_name(dir_name, dates_name_list):

    for i, j in enumerate(dir_name):
        root_directory = j
        dates_name_list[i] = find_mseed_files_with_dates(root_directory)
        

    dict_name = defaultdict(list)

    for d in dates_name_list:
        for key, value in d.items():
            dict_name[key].append(value)

    dict_name = dict(dict_name)

    dict_name = {k: dict_name[k] for k in sorted(dict_name.keys())}

    # years = {fecha.split('-')[0] for fecha in dict_name.keys()}
    # years = sorted(list(years))

    return dict_name