import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

directory_path = './data/a-iii/'
output_directory = './data/a-iii/'

# Lists to store means and standard deviations for all files
means_above, means_below, stds_above, stds_below, file_names = [], [], [], [], []

for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path, header=None)

        # Default values (will be overridden if actual values are computed)
        av_mean_above, av_mean_below, av_std_above, av_std_below = 0, 0, 0, 0
        easy_filename = filename[:-4]
        file_names.append(easy_filename)

        # Plot 1: Raw Data with Mean
        plt.figure(figsize=(8, 6))
        plt.plot(df.index / 500, df[0], color='grey', alpha=0.5, linewidth=0.5)
        plt.axhline(df[0].mean(), color='red', linewidth=0.5)
        plt.ylim(150, 275)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        #plt.title(f'Raw data for {easy_filename}')
        plt.tight_layout()
        output_file_path_1 = os.path.join(output_directory, filename.replace('.csv', '_raw.png'))
        plt.savefig(output_file_path_1)
        plt.close()

        # Rectified Signal and its Moving Average
        average_value = df[0].mean()
        rectified_signal = (df[0] - average_value).abs()
        moving_avg = rectified_signal.pow(2).rolling(window=49).apply (lambda x: np.sqrt(x.mean()))
        average_moving_value = moving_avg.mean() * 0.9

        # Plot 2: Rectified Signal with Moving Average and Sets Based on Value 15
        plt.figure(figsize=(8, 6))
        plt.plot(df.index / 500, rectified_signal, color='blue', alpha=0.4, linewidth=0.5)
        plt.plot(df.index / 500, moving_avg, color='red', alpha=0.75, linewidth=0.8)  # Plot moving average

        plt.axhline(average_moving_value, color='green', linewidth=1)
        # Filtered rectified signal values based on moving average
        signal_above_15 = moving_avg[moving_avg >= average_moving_value]
        signal_below_15 = moving_avg[moving_avg <  average_moving_value]

        av_mean_above = signal_above_15.mean()
        av_std_above = signal_above_15.std()

        av_mean_below = signal_below_15.mean()
        av_std_below = signal_below_15.std()

        text_str_above = f'Smoothed Signal: {av_mean_above:.2f}  Std Dev: {av_std_above:.2f}'
        text_str_below = f'Smoothed Noise: {av_mean_below:.2f}  Std Dev: {av_std_below:.2f}'

        plt.text(1, 130, text_str_above, color='g')
        plt.text(1, 115, text_str_below, color='g')

        #plt.title(f'Rectified and Smoothed data for {easy_filename}')
        plt.ylim(0, 150)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        plt.tight_layout()

        output_file_path_2 = os.path.join(output_directory, filename.replace('.csv', '_rectified.png'))
        plt.savefig(output_file_path_2)
        plt.close()

        means_above.append(av_mean_above)
        means_below.append(av_mean_below)
        stds_above.append(av_std_above)
        stds_below.append(av_std_below)

# Plotting means with standard deviations for all files
plt.figure(figsize=(6, 6))
positions = np.arange(len(file_names))
plt.errorbar(positions, means_above, lw=2, yerr=stds_above, color='blue', alpha=0.75, linestyle = 'none',capsize =8, capthick =2, label='Smoothed Signal')
plt.errorbar(positions, means_below, lw=2, yerr=stds_below, color='red', alpha=1, linestyle = 'none',capsize =8, capthick =2, label='Smoothed Noise')
plt.xticks(positions, file_names, rotation=45, ha="right")
plt.tight_layout()
plt.legend(loc='upper right')
output_file_path_final = os.path.join(output_directory, 'summary_plot.png')
plt.savefig(output_file_path_final)
plt.close()
