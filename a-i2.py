import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

directory_path = './data/a-i/'
#os.makedirs(output_directory, exist_ok=True)

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
        window_size = 5
        smoothed_envelope = moving_average(envelope, window_size)
        midpoint = (smoothed_envelope.max() - smoothed_envelope.min()) / 2
        
        # Divide data based on midpoint
        below_midpoint = smoothed_envelope[smoothed_envelope < midpoint]
        above_midpoint = smoothed_envelope[smoothed_envelope >= midpoint]

        plt.figure(figsize=(8, 6))

        # Plot original data
        plt.plot(df.index, df[0], color='grey', alpha=0.3, label=f'Original {filename}')
        
        # Plot the two groups in different colors
        plt.plot(df.index[int(window_size/2):-int(window_size/2)][smoothed_envelope < midpoint], below_midpoint, 'b', label='Below Midpoint')
        plt.plot(df.index[int(window_size/2):-int(window_size/2)][smoothed_envelope >= midpoint], above_midpoint, 'g', label='Above Midpoint')
        
        # Mean and standard deviation for each set
        mean_below, std_below = below_midpoint.mean(), below_midpoint.std()
        mean_above, std_above = above_midpoint.mean(), above_midpoint.std()
        plt.errorbar(len(df) * 0.1, mean_below, yerr=std_below, color='b', fmt='o')
        plt.errorbar(len(df) * 0.2, mean_above, yerr=std_above, color='g', fmt='o')
        
        plt.ylim(-350, 350)
        plt.xlabel('Line Number')
        plt.ylabel('Value')
        plt.title(f'CSV Data Plot for {filename}')
        plt.legend()

        output_file_path = os.path.join(directory_path, f'{os.path.splitext(filename)[0]}.png')
        plt.savefig(output_file_path)
        plt.close()
