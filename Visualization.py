import matplotlib.pyplot as plt
import pandas as pd

# List of CSV files
csv_files = ['ShivanshData\ShivWalking.csv', 'NicoData\\NicoWalking.csv','AidanData\AidanWalking.csv',
             'ShivanshData\ShivJumping.csv', 'NicoData\\NicoJumping.csv','AidanData\AidanJumping.csv']

# Create subplots
fig, axs = plt.subplots(len(csv_files), 5, figsize=(12, len(csv_files)*5))

# Iterate over each CSV file
for i, file in enumerate(csv_files):
    # Read the CSV file
    data = pd.read_csv(file)

    # Extract data for x and y axes for each subplot
    x3_values = data['Linear Acceleration z (m/s^2)']
    y4_values = data['Absolute acceleration (m/s^2)']

    # Plot each graph
    axs[i, 0].plot(data['Time (s)'], data['Linear Acceleration x (m/s^2)'])
    axs[i, 0].set_title('X Acceleration')

    axs[i, 1].plot(data['Time (s)'], data['Linear Acceleration y (m/s^2)'])
    axs[i, 1].set_title('Y Acceleration')

    axs[i, 2].plot(data['Time (s)'], data['Linear Acceleration z (m/s^2)'])
    axs[i, 2].set_title('Z Acceleration')

    axs[i, 3].plot(data['Time (s)'], data['Absolute acceleration (m/s^2)'])
    axs[i, 3].set_title('Absolute Acceleration')

    axs[i, 4].plot(y4_values, x3_values)
    axs[i, 4].set_title('Absolute vs Z Acceleration')
    
    # Set figure title to the name of the CSV file without .CSV extension
    axs[i, 0].set_ylabel(file.split('\\')[-1].split('.')[0], rotation=0, ha='right')

# Adjust layout
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()
