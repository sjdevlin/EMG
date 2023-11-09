import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

# Path to the directory containing your CSV files
directory_path = './data/'

# Function to compute the moving average for smoothing
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')



# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        # Construct full file path
        file_path = os.path.join(directory_path, filename)
       
        plt.figure(figsize=(12,8))

        # Read the CSV file
        df = pd.read_csv(file_path, header=None)  # Assuming no header in the CSV

	# calculate the average
        average_value = df[0].mean()
        absolute_difference = (df[0] - average_value).abs()
        analytical_signal = hilbert(absolute_difference)
        envelope = np.abs(analytical_signal)
        
        # Smooth the envelope using a moving average (you can adjust the window size as needed)
        window_size = 11
        smoothed_envelope = moving_average(envelope, window_size)
        smoothed_envelope += average_value

        # create a new plot for each file
        # Plot the data
        # X-axis: Line numbers
        # Y-axis: Values from the file
        plt.plot(df.index/500, df[0], label=filename, linewidth='0.5', color = 'gray')  # Assuming single column CSV
        plt.plot(df.index[int(window_size/2):-int(window_size/2)]/500,smoothed_envelope,linewidth='1', color = 'red')


        # Display the plots
        plt.xlabel('Time / Seconds')
        plt.ylabel('EMG Signal Value')
        plt.title('CSV Data Plot')
        #plt.legend()
        
        output_file = os.path.join(directory_path,f'{os.path.splitext(filename)[0]}.png')
        plt.savefig(output_file)
        plt.close()


