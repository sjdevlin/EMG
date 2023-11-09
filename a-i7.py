import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

directory_path = './data/a-i/'
output_directory = './data/a-i/'

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, header=None)
        average_value = df[0].mean()
        rectified_signal = (df[0] - average_value).abs()

        rectified_series = pd.Series(rectified_signal)  # Convert rectified signal to a Pandas Series
        moving_avg = rectified_series.rolling(window=21).mean()

        plt.figure(figsize=(10, 8))

        # Plot original data and moving average of the rectified signal
        plt.plot(df.index / 500, df[0], color='grey', alpha=0.3, linewidth=0.5)
        plt.plot(df.index / 500, rectified_series, color='blue', alpha=0.3, linewidth=0.5)
        plt.plot(df.index / 500, moving_avg, color='green', alpha=0.6, linewidth=0.5)  # Plot moving average

        # Filtered rectified signal values based on moving average
        signal_above_25 = rectified_signal[moving_avg >= 10]
        signal_below_25 = rectified_signal[moving_avg < 10]

        mean_above = signal_above_25.mean()
        std_above = signal_above_25.std()
        mean_below = signal_below_25.mean()
        std_below = signal_below_25.std()

        plt.axhline(mean_above, color='blue', linewidth=0.5)
        plt.axhline(mean_below, color='red', linewidth=0.5)
        plt.axhline(mean_above + std_above, color='blue', linestyle='--', linewidth=0.5)
        plt.axhline(mean_below - std_below, color='red', linestyle='--', linewidth=0.5)
        plt.axhline(mean_above - std_above, color='blue', linestyle='--', linewidth=0.5)
        plt.axhline(mean_below + std_below, color='red', linestyle='--', linewidth=0.5)

        # Add mean and standard deviation values as text
        plt.text(0.01, 170, f'Mean (>=25): {mean_above:.2f}', color='blue')
        plt.text(0.01, 145, f'Standard Deviation (>=25): {std_above:.2f}', color='blue')
        plt.text(200, 170, f'Mean (<25): {mean_below:.2f}', color='red')
        plt.text(200, 145, f'Standard Deviation (<25): {std_below:.2f}', color='red')

        plt.ylim(0, 375)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        plt.tight_layout()

        output_file_path = os.path.join(output_directory, filename.replace('.csv', '.png'))
        plt.savefig(output_file_path)
        plt.close()
