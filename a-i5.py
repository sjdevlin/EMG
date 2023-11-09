import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

directory_path = './data/a-i/'
output_directory = './data/a-i/'

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, header=None)
        average_value = df[0].mean()
        absolute_difference = (df[0] - average_value).abs()
        envelope = np.abs(hilbert(absolute_difference))

        plt.figure(figsize=(8, 6))

        # Plot original data and the envelope
        plt.plot(df.index / 500, df[0], color='grey', alpha=0.3, label=f'Original {filename}', linewidth=0.5)
        plt.plot(df.index / 500, envelope, color='purple', alpha=0.6, label='Envelope', linewidth=0.5)

        # Filtered envelope values based on specified criteria
        envelope_above_30 = envelope[envelope >= 30]
        envelope_below_30 = envelope[envelope < 30]

        # Mean and standard deviation for envelope values above 30
        mean_above, std_above = envelope_above_30.mean(), envelope_above_30.std()
        plt.axhline(y=mean_above, color='b', linestyle='-', linewidth=0.5, label='Mean (Env > 30)')
        plt.axhline(y=mean_above - std_above, color='b', linestyle=':', linewidth=0.5)
        plt.axhline(y=mean_above + std_above, color='b', linestyle=':', linewidth=0.5)

        # Mean and standard deviation for envelope values below 10
        mean_below, std_below = envelope_below_10.mean(), envelope_below_10.std()
        plt.axhline(y=mean_below, color='r', linestyle='-', linewidth=0.5, label='Mean (Env < 20)')
        plt.axhline(y=mean_below - std_below, color='r', linestyle=':', linewidth=0.5)
        plt.axhline(y=mean_below + std_below, color='r', linestyle=':', linewidth=0.5)

        plt.ylim(0, 375)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        plt.title(f'CSV Data Plot for {filename}')
        #plt.legend()

        output_file_path = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}.png')
        plt.savefig(output_file_path)
        plt.close()
