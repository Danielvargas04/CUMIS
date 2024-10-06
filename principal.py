from functions import*

def parameters(name):
    if name == 'Moon':
        sta_len = 60
        lta_len = 900
        cutoff_freq = 0.0001/2
        prominence = 0.25
        distance = 5000 ##*df
        wlen_value = 4000 ##*df
        height = 1.0
        directory = 'data/lunar/test/data/S12_GradeB'

    if name == 'Earth':
        sta_len = 10
        lta_len = 50
        cutoff_freq = 0.0001/2 ##MISSING
        prominence = 0.25  ##MISSING
        distance = 5000 ##*df   ##MISSING
        wlen_value = 4000 ##*df  ##MISSING
        height = 1.0  ##MISSING
        directory = 'data/EARTH'

    if name == 'Mars':
        sta_len = 10
        lta_len = 50
        cutoff_freq = 0.0001/2 ##MISSING
        prominence = 0.25  ##MISSING
        distance = 5000 ##*df   ##MISSING
        wlen_value = 4000 ##*df  ##MISSING
        height = 1.0  ##MISSING
        directory = 'data/mars/test/data'

    return sta_len, lta_len, cutoff_freq, prominence, distance, wlen_value, height, directory

def main(name, filename):

    sta_len, lta_len, cutoff_freq, prominence, distance, wlen_value, height, data_directory = parameters(name)

    # Extract Data
    tr_times, tr_data_filt_norm, df, tr_data = data_extrac_test(filename = filename)

    #STA/LTA Algorithm. Get characteristic function
    cft = CFT(tr_data_filt_norm, df, sta_len = sta_len, lta_len = lta_len)

    #Characteristic function filtered with fourier transform
    filtered_cft_real = fourier_filter(cft, cutoff_freq = cutoff_freq)

    #Find peaks in characteristic function filtered
    peaks, properties = peaks_plot(tr_times, filtered_cft_real, prominence = prominence, distance = distance*df, wlen_value = wlen_value*df, height = height)

    #Spectrogram to determine which peaks are real events  
    frequencies, times, sxx = signal.spectrogram(tr_data, df, nperseg = 2000, noverlap = 1800)

    #Find index of the cutoff frequency in the spectrogram
    low_index = get_index_from_f(frequencies, 0.5) 
    up_index = get_index_from_f(frequencies, 1)

    #Spectrogram in the cutoff frequencies
    sxx = sxx[low_index:up_index, :]
    frequencies = frequencies[low_index:up_index]

    #Spectrogram convolution and its histogram
    hist, conv = hist_convolve_spectrogram(sxx, 10, frequencies)

    # plt.show()
    # plt.pcolormesh(times,frequencies,conv)
    # plt.show()

    #plt.pcolormesh(times,frequencies_filter, conv_filter)
    #plt.title('histograma filtrado')
    #plt.show()timr

    times_left_bases = tr_times[properties['left_bases']]
    times_right_bases = tr_times[properties['right_bases']]


    plt.plot(times,hist ,label='histograma')
    plt.vlines(times_left_bases, ymin=min(hist), ymax=max(hist), color='blue' )
    plt.vlines(times_right_bases, ymin=min(hist), ymax=max(hist), color='red')


    #plt.plot(hist_filter, label='histograma_filter')

    plt.legend()

    plt.show()

    # range_left = indice_mas_cercano(times,times_left_bases)
    # range_right = indice_mas_cercano(times,times_right_bases)
    # conn = np.zeros(len(times_left_bases))
    # for i in range(len(times_left_bases)):
    #     conn[i]=np.max(hist[range_left[i]:range_right[i]])

    # conn




    #up_index_filter = get_index_from_f(frequencies, 0.4)

    #filter test
    # sxx_filter = sxx[:up_index_filter, :]
    # frequencies_filter = frequencies[:up_index_filter]
    #hist_filter, conv_filter = hist_convolve_spectrogram(sxx_filter, 10, frequencies_filter)


    #Proximamente valores de confianza
    confianza = np.zeros(len(peaks))

    # Plot raw data
    # fig,ax = plt.subplots(1,1,figsize=(12,3))
    # ax.plot(tr_times, tr_data_filt_norm, label=f'{id} test ')
    # ax.plot(tr_times, filtered_cft_real, label=f'CFT ')
    # ax.set_xlim([min(tr_times),max(tr_times)])

    #ax.axvline(x = arrival, color='red',label='Rel. Arrival')


    #ax.axvline(x = max_pos, color='green')#,label='Rel. Arrival')
    #ax.set_xlim([72000,76000])
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel(f'Signal {id} from {data_directory} function')
    # ax.legend()

    plt.plot(tr_times, tr_data)
    plt.show()

#main('Moon', './data/lunar/test/data/S12_GradeB/xa.s12.00.mhz.1969-12-16HR00_evid00006.mseed')    

