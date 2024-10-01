# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from funtions import *

import numpy as np
import matplotlib.pyplot as plt

# Inicialización de los valores de STA y LTA
STA_values = np.arange(20, 2500, 50)
LTA_values = np.arange(200, 25000, 500)

# Matrices para los errores relativos y absolutos
errores_rel = np.zeros((len(STA_values), len(LTA_values)))
errores_abs = np.zeros((len(STA_values), len(LTA_values)))

# Loop sobre los valores de STA y LTA
for i, STA in enumerate(STA_values):
    print('Paso: ', i)
    for j, LTA in enumerate(LTA_values):
        # Solo procesar si STA es menor que LTA
        if STA < LTA:
            # Extraer los datos y calcular el CFT
            tr_times, tr_data_filt_norm, arrival, df = data_extrac(54)
            cft = CFT(tr_data_filt_norm, df, STA, LTA)
            filtered_cft_real = fourier_filter(cft)

            # Obtener la posición máxima
            max_pos = tr_times[np.where(filtered_cft_real == filtered_cft_real.max())]
            
            # Calcular los errores relativos y absolutos
            error_relativo = np.abs((arrival - max_pos) / arrival)
            error_abs = np.abs(arrival - max_pos)
            
            # Almacenar los errores en las matrices
            #print(error_relativo.shape, errores_rel.shape)
            errores_rel[i, j] = error_relativo[0]
            errores_abs[i, j] = error_abs[0]
        else:
            # Si STA >= LTA, asignamos NaN (Not a Number) para evitar confusión
            errores_rel[i, j] = np.nan
            errores_abs[i, j] = np.nan

print(errores_rel.shape)
print(errores_abs.shape)
print(errores_rel[:2,:2])
print(errores_abs[:2,:2])

# Guardar errores_rel como un archivo .npy
np.save('errores_rel.npy', errores_rel)

# Guardar errores_abs como un archivo .npy
np.save('errores_abs.npy', errores_abs)

# Crear el mapa de calor de errores relativos
plt.figure(figsize=(10, 8))
plt.imshow(errores_rel, extent=[LTA_values.min(), LTA_values.max(), STA_values.min(), STA_values.max()],
           aspect='auto', origin='lower', cmap='inferno')
plt.colorbar(label='Error Relativo')
plt.xlabel('LTA')
plt.ylabel('STA')
plt.title('Mapa de Calor de Errores Relativos')


# Crear el mapa de calor de errores absolutos
plt.figure(figsize=(10, 8))
plt.imshow(errores_abs, extent=[LTA_values.min(), LTA_values.max(), STA_values.min(), STA_values.max()],
           aspect='auto', origin='lower', cmap='inferno')
plt.colorbar(label='Error Absoluto')
plt.xlabel('LTA')
plt.ylabel('STA')
plt.title('Mapa de Calor de Errores Absolutos')
plt.show()

    

'''tr_times, tr_data_filt_norm, arrival, df= data_extrac(47)
cft = CFT(tr_data_filt_norm, df)
filtered_cft_real = fourier_filter(cft)


# Plot characteristic function
fig,ax = plt.subplots(1,1,figsize=(12,3))
ax.plot(tr_times,filtered_cft_real)
ax.plot(tr_times, tr_data_filt_norm)
ax.set_xlim([min(tr_times),max(tr_times)])
print(arrival)
ax.axvline(x = arrival, color='red',label='Rel. Arrival')


ax.axvline(x = max_pos, color='green',label='Rel. Arrival')
#ax.set_xlim([72000,76000])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Characteristic function')


peaks_plot(tr_times, filtered_cft_real, prominence=0.8)'''
