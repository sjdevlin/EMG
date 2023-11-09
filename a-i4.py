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

        # Plot original data
        plt.plot(df.index / 500, df[0], color='grey', alpha=0.3, label=f'Original {filename}', linewidth=0.5)

        # Filtered data based on envelope criteria
        above_15 = df[0][envelope > 15]
        below_10 = df[0][envelope < 10]

        # Mean and standard deviation for data where envelope is above 15
        mean_above, std_above = above_15.mean(), above_15.std()
        plt.axhline(y=mean_above, color='b', linestyle='-', linewidth=0.5, label='Mean (Env > 15)')
        plt.axhline(y=mean_above - std_above, color='b', linestyle=':', linewidth=0.5)
        plt.axhline(y=mean_above + std_above, color='b', linestyle=':', linewidth=0.5)

        # Mean and standard deviation for data where envelope is below 10
        mean_below, std_below = below_10.mean(), below_10.std()
        plt.axhline(y=mean_below, color='r', linestyle='-', linewidth=0.5, label='Mean (Env < 10)')
        plt.axhline(y=mean_below - std_below, color='r', linestyle=':', linewidth=0.5)
        plt.axhline(y=mean_below + std_below, color='r', linestyle=':', linewidth=0.5)

        plt.ylim(-10, 375)
        plt.xlabel('Seconds')
        plt.ylabel('EMG (au)')
        plt.title(f'CSV Data Plot for {filename}')
        plt.legend()

        output_file_path = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}.png')
        plt.savefig(output_file_path)
        plt.close()
