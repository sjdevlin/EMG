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

        # Calculate moving average of the envelope
        envelope_series = pd.Series(envelope) 
        # Convert envelope to a Pandas Series
        moving_avg = envelope_series.rolling(window=5).mean()

        plt.figure(figsize=(8, 6))

        # Plot original data, envelope, and moving average of the envelope
        plt.plot(df.index / 500, df[0], color='grey', alpha=0.3, linewidth=0.5)
        plt.plot(df.index / 500, envelope, color='purple', alpha=0.6, linewidth=0.5)
        plt.plot(df.index / 500, moving_avg, color='green', alpha=0.6, linewidth=0.5)

        # Filtered envelope values based on moving average
        envelope_above_25 = envelope[moving_avg >= 20]
        envelope_below_25 = envelope[moving_avg < 20]

        # Mean and standard deviation for envelope values where moving average >= 25
        mean_above, std_above = envelope_above_25.mean(), envelope_above_25.std()
        plt.axhline(y=mean_above, color='b', linestyle='-', linewidth=0.5)
        plt.axhline(y=mean_above - std_above, color='b', linestyle=':', linewidth=0.5)
        plt.axhline(y=mean_above + std_above, color='b', linestyle=':', linewidth=0.5)

        # Mean and standard deviation for envelope values where moving average < 25
        mean_below, std_below = envelope_below_25.mean(), envelope_below_25.std()
        plt.axhline(y=mean_below, color='r', linestyle='-', linewidth=0.5)
        plt.axhline(y=mean_below - std_below, color='r', linestyle=':', linewidth=0.5)
        plt.axhline(y=mean_below + std_below, color='r', linestyle=':', linewidth=0.5)

        plt.ylim(0, 375)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        plt.title(f'CSV Data Plot for {filename}')

        # Add text box with mean and std dev values
        text_str_above = f'Signal Mean: {mean_above:.2f}\nStd Dev: {std_above:.2f}'
        text_str_below = f'Noise Mean: {mean_below:.2f}\nStd Dev: {std_below:.2f}'

        plt.text(0.75, 250, text_str_above, color='b', bbox=dict(facecolor='none', edgecolor='b', boxstyle='round,pad=0.5'))
        plt.text(0.75, 100, text_str_below, color='r', bbox=dict(facecolor='none', edgecolor='r', boxstyle='round,pad=0.5'))

        output_file_path = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}.png')
        plt.savefig(output_file_path)
        plt.close()
