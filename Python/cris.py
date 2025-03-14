# -*- coding: utf-8 -*-

import signal_utilities as su
import emg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ..............................................................................
# Funciones para calcular las métricas

# 1. RMS
def calculate_rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

# 2. MAV
def calculate_mav(signal):
    return np.mean(np.abs(signal))

# 3. ARV
def calculate_arv(signal):
    return np.mean(np.abs(signal))

# 4. ZC (Zero Crossing)
def calculate_zc(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings)

# 5. MNF y MDF
def calculate_mnf_mdf(signal, fs):
    # Realizamos la FFT de la señal
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_signal = np.fft.fft(signal)

    # Calculamos el espectro de potencia
    power_spectrum = np.abs(fft_signal)**2

    # Filtramos las frecuencias negativas
    # Create a mask for positive frequencies with the original length
    mask = freqs > 0

    # Apply the mask to both frequencies and power spectrum
    freqs = freqs[mask]
    power_spectrum = power_spectrum[mask]

    # Calculamos MNF y MDF
    mnf = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
    mdf = freqs[np.cumsum(power_spectrum) >= np.sum(power_spectrum) / 2][0]

    return mnf, mdf

# 6. PKF (Peak Frequency)
def calculate_pkf(signal, fs):
    # Realizamos la FFT de la señal
    n = len(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    fft_signal = np.fft.fft(signal)

    # Calculamos el espectro de potencia
    power_spectrum = np.abs(fft_signal)**2

    # Filtramos las frecuencias negativas
    # Create a mask for positive frequencies with the original length
    mask = freqs > 0

    # Apply the mask to both frequencies and power spectrum
    freqs = freqs[mask]
    power_spectrum = power_spectrum[mask]

    # Encontramos la frecuencia de pico
    pkf = freqs[np.argmax(power_spectrum)]

    return pkf
# ..............................................................................

#1.- Carga de datos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
data_path = '../Sample Signals/emgvalues_2.csv'
data_path = '../Sample Signals/emgvalues_3.csv'
fs = 1000  # Hz, frecuencia de muestreo del EMG que usamos

data_path = '../Sample Signals/emgvalues_1.csv'
emg_data = pd.read_csv(data_path, header=None).squeeze()  # Convierte el DataFrame en Serie

# Sobre el formato de la serie:
# El indice son números enteros a la frecuencia de muestreo (1KHz en los samples)
# El valor  son números enteros en mili voltios


#2.- Filtrado . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
basic_filter = emg.EMG_filter_basic(sample_frequency=fs, range_=0.1, reference_available=False)
emg_filter = emg.EMG_filter(sample_frequency=fs, range_=0.5, min_EMG_frequency=25, max_EMG_frequency=300, reference_available=False)
# Filtrar la señal usando el filtro avanzado
filtered_signal = [emg_filter.filter(value) for value in emg_data]

# Calcular todas las métricas
rms = calculate_rms(filtered_signal)
mav = calculate_mav(filtered_signal)
arv = calculate_arv(filtered_signal)
zc = calculate_zc(filtered_signal)
mnf, mdf = calculate_mnf_mdf(filtered_signal, fs)
pkf = calculate_pkf(filtered_signal, fs)

# Mostrar los resultados
print(f'RMS: {rms}')
print(f'MAV: {mav}')
print(f'ARV: {arv}')
print(f'Zero Crossings: {zc}')
print(f'MNF: {mnf} Hz')
print(f'MDF: {mdf} Hz')
print(f'Peak Frequency (PKF): {pkf} Hz')

#3.- Visualizacion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

# Eje temporal
time = [i / fs for i in range(len(emg_data))]

# Graficar las métricas junto con la señal
plt.figure(figsize=(12, 6))

# Graficar señal filtrada
plt.subplot(2, 1, 1)
plt.plot(time, emg_data, label='Señal Original', alpha=0.5)
plt.plot(time, filtered_signal, label='Señal Filtrada', linewidth=2)
plt.title("Filtrado de Señal EMG")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mV)")
plt.legend()
plt.grid(True)

# Graficar espectro de frecuencias
plt.subplot(2, 1, 2)
n = len(filtered_signal)
frequencies = np.fft.fftfreq(n, 1/fs)
fft_vals = np.fft.fft(filtered_signal)
plt.plot(frequencies[:n//2], np.abs(fft_vals)[:n//2])
plt.title("Espectro de Frecuencia de la Señal")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()
