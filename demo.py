# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from functions import *

import numpy as np
import matplotlib.pyplot as plt
   
#sospechosos = [8,22,36,42,49,56,68,74]

id = 40#np.random.randint(0,63)

tr_times, tr_data_filt_norm, df, tr_data= data_extrac_test(id, data_directory='./data/EARTH/EARTH/')
#tr_times, tr_data_filt_norm, arrival, df= data_extrac(id)

cft = CFT(tr_data_filt_norm, df, 10, 50)
filtered_cft_real = fourier_filter(cft, 0.00022)
max_pos = tr_times[np.where(filtered_cft_real == filtered_cft_real.max())]
#error_relativo = np.abs((arrival - max_pos) / arrival)
#error_abs = np.abs(arrival - max_pos)

#print('Error relativo: ', error_relativo)
#print('Error absoluto: ', error_abs)
# Plot characteristic function
fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(tr_times,filtered_cft_real)
ax.plot(tr_times, tr_data_filt_norm, label=f'{id} test')
ax.set_xlim([min(tr_times),max(tr_times)])
#ax.axvline(x = arrival, color='red',label='Rel. Arrival')


ax.axvline(x = max_pos, color='green')#,label='Rel. Arrival')
#ax.set_xlim([72000,76000])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Characteristic function')
ax.legend()


peaks_plot(tr_times, filtered_cft_real, prominence=0.4, distance=5*100, wlen_value=100*100, height=1.3)
