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

        # Plot 1: Raw Data with Mean
        plt.figure(figsize=(8, 6))
        plt.plot(df.index / 500, df[0], color='grey', alpha=0.3, linewidth=0.5)
        plt.axhline(df[0].mean(), color='red', linewidth=0.5)
        plt.ylim(0, 375)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        plt.tight_layout()
        output_file_path_1 = os.path.join(output_directory, filename.replace('.csv', '_raw.png'))
        plt.savefig(output_file_path_1)
        plt.close()

        # Rectified Signal and its Moving Average
        average_value = df[0].mean()
        rectified_signal = (df[0] - average_value).abs()
        moving_avg = rectified_signal.rolling(window=25).mean()

        # Plot 2: Rectified Signal with Moving Average and Sets Based on Value 15
        plt.figure(figsize=(8, 6))
        plt.plot(df.index / 500, rectified_signal, color='blue', alpha=0.5, linewidth=0.5)
        plt.plot(df.index / 500, moving_avg, color='red', alpha=0.75, linewidth=1)  # Plot moving average

        # Filtered rectified signal values based on moving average
        signal_above_15 = rectified_signal[moving_avg >= 20]
        signal_below_15 = rectified_signal[moving_avg < 20]
        av_signal_above_15 = moving_avg[moving_avg >= 20]
        av_signal_below_15 = moving_avg[moving_avg < 20]

        av_mean_above = av_signal_above_15.mean()
        av_std_above = av_signal_above_15.std()
        av_mean_below = av_signal_below_15.mean()
        av_std_below = av_signal_below_15.std()

        text_str_above = f'Smoothed Signal: {av_mean_above:.2f}  Std Dev: {av_std_above:.2f}'
        text_str_below = f'Smoothed Noise: {av_mean_below:.2f}  Std Dev: {av_std_below:.2f}'

        plt.text(1, 130, text_str_above, color='g')
        plt.text(1, 115, text_str_below, color='g')

        plt.ylim(0, 150)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        plt.tight_layout()

        output_file_path_2 = os.path.join(output_directory, filename.replace('.csv', '_rectified.png'))
        plt.savefig(output_file_path_2)
        plt.close()
