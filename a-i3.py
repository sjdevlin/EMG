import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

directory_path = './data/a-i/'

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, header=None)
        average_value = df[0].mean()
        absolute_difference = (df[0] - average_value).abs()
        analytical_signal = hilbert(absolute_difference)
        envelope = np.abs(analytical_signal)
        window_size = 3
        smoothed_envelope = moving_average(envelope, window_size)

        plt.figure(figsize=(8, 6))

        # Plot original data
        plt.plot(df.index / 500, df[0], color='grey', alpha=0.3, label=f'Original {filename}', linewidth=0.5)
        
        # Plot the smoothed envelope
        plt.plot(df.index[int(window_size/2):-int(window_size/2)] / 500, smoothed_envelope, 'b', label='Smoothed Envelope', linewidth=0.5)
        
        # Mean and standard deviation for the smoothed envelope
        mean_value, std_value = smoothed_envelope.mean(), smoothed_envelope.std()
        plt.axhline(y=mean_value, color='r', linestyle='-', linewidth=0.5, label='Mean')
        plt.axhline(y=mean_value - std_value, color='r', linestyle=':', linewidth=0.5, label='Mean - Std Dev')
        plt.axhline(y=mean_value + std_value, color='r', linestyle=':', linewidth=0.5, label='Mean + Std Dev')
        
        plt.ylim(-50, 350)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        plt.title(f'CSV Data Plot for {filename}')
        plt.legend()

        output_file_path = os.path.join(directory_path, f'{os.path.splitext(filename)[0]}.png')
        plt.savefig(output_file_path)
        plt.close()
