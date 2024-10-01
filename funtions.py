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

    print(f"Cantidad de datos en el directorio {data_directory} : {len(path_data)}")
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
    
    #Filtro de normalizacion de tiempos
    tr_times_norm = tr_times.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    tr_times_norm = scaler.fit_transform(tr_times_norm)

    return tr_times, tr_data_filt_norm, df, tr_data

def CFT(tr_data_filt_norm, df, sta_len = 120, lta_len = 600):

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

def fourier_filter(cft):
    # Número de puntos en el arreglo
    n = len(cft)

    # Frecuencia de muestreo (puedes ajustarla dependiendo del contexto)
    sampling_rate = 1 #df  # Ajusta según corresponda

    # Realizar la FFT del arreglo
    cft_fft = fft(cft)

    # Crear las frecuencias correspondientes
    frequencies = fftfreq(n, d=sampling_rate)

    # Definir la frecuencia de corte para el filtro
    cutoff_freq = 0.0001/2 # Ajusta según lo que necesites

    # Aplicar el filtro pasa bajos: eliminar las frecuencias más altas que la frecuencia de corte
    cft_fft[np.abs(frequencies) > cutoff_freq] = 0

    # Realizar la inversa de la FFT para regresar al dominio temporal
    filtered_cft = ifft(cft_fft)

    # La señal filtrada está en dominio temporal (real)
    filtered_cft_real = np.real(filtered_cft)
    return filtered_cft_real

def peaks_plot(tr_times, filter_cft, prominence=0.3, distance=10000*6.6, wlen_value=8000*6.6 ):
    # Use find_peaks with prominence, distance, and wlen
    # wlen_value: Set the window length for prominence calculation (adjust as needed)
    peaks, properties = find_peaks(filter_cft, prominence=prominence, distance=distance, wlen=wlen_value)

    # Plot the signal and the detected peaks
    fig, ax = plt.subplots(1, 1, figsize=(16, 3))
    ax.plot(tr_times, filter_cft, label="filter_cft")
    ax.plot(tr_times[peaks], filter_cft[peaks], "x", label="Peaks")

    # Plot the prominence as vertical lines
    for peak, prominence, left_base, right_base in zip(peaks, properties['prominences'], properties['left_bases'], properties['right_bases']):
        # Plot vertical lines from the peak to its prominence baseline
        ax.vlines(x=tr_times[peak], ymin=filter_cft[peak] - prominence, ymax=filter_cft[peak], color="C1", linestyles="--")
        
        # Plot horizontal lines showing the left and right bases of each prominence
        ax.hlines(y=filter_cft[peak] - prominence, xmin=tr_times[left_base], xmax=tr_times[right_base], color="C1", linestyles="--")

    # Customize the plot
    ax.set_title(f"Peak Detection with Prominence (wlen={wlen_value}) Visualization")
    ax.legend()

    # Show the indices of the detected peaks
    print(f"Picos encontrados en los tiempos: {tr_times[peaks]}")

    plt.show()

    
