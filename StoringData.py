import pandas as pd
import numpy as np
import h5py

from sklearn.model_selection import train_test_split

with h5py.File('Data.h5', 'w') as hdf:
    #Creating the folders in the HDF5 
    Dataset1 = hdf.create_group('/Shivansh')
    Dataset2 = hdf.create_group('/Aidan')
    Dataset3 = hdf.create_group('/Nico')
    Dataset41 = hdf.create_group('/dataset/Train')
    Dataset42 = hdf.create_group('/dataset/Test')


    #Reading the CSV files and storing them into the HDF5
    csv_path = 'ShivanshData/ShivWalking.csv'
    df = pd.read_csv(csv_path)
    Dataset1.create_dataset('ShivWalking', data=df)

    csv_path = 'ShivanshData\ShivJumping.csv'
    df = pd.read_csv(csv_path)
    Dataset1.create_dataset('ShivJumping', data=df)
    
    csv_path = 'AidanData\AidanWalking.csv'
    df = pd.read_csv(csv_path)
    Dataset2.create_dataset('AidanWalking', data=df)

    csv_path = 'AidanData\AidanJumping.csv'
    df = pd.read_csv(csv_path)
    Dataset2.create_dataset('AidanJumping', data=df)

    csv_path = 'NicoData\\NicoWalking.csv'
    df = pd.read_csv(csv_path)
    Dataset3.create_dataset('NicoWalking', data=df)

    csv_path = 'NicoData\\NicoJumping.csv'
    df = pd.read_csv(csv_path)
    Dataset3.create_dataset('NicoJumping', data=df)

    jumpData = ['ShivanshData\ShivJumping.csv', 'AidanData\AidanJumping.csv', 'NicoData\\NicoJumping.csv']
    walkData = ['ShivanshData/ShivWalking.csv', 'AidanData\AidanWalking.csv', 'NicoData\\NicoWalking.csv']

    jumping_dfs = [pd.read_csv(file) for file in jumpData]
    walking_dfs = [pd.read_csv(file) for file in walkData]

    jumping_df = pd.concat(jumping_dfs, axis=0, ignore_index=True)
    walk_df = pd.concat(walking_dfs, axis=0, ignore_index=True)


    # Divide signal into windows
    windowSize = 500  # 500 points corresponds to 5 seconds
    numWindows1 = int(len(jumping_df) / windowSize)
    numWindows2 = int(len(walk_df) / windowSize)

    jumpingWindowSize = [jumping_df.iloc[i:i+windowSize] for i in range(0, len(jumping_df), windowSize) if
                      len(jumping_df.iloc[i:i + windowSize] == windowSize)]

    walkingWindowSize = [walk_df.iloc[i:i + windowSize] for i in range(0, len(walk_df), windowSize) if
                      len(walk_df.iloc[i:i + windowSize] == windowSize)]

    jumpTrain, jumpTest = train_test_split(jumpingWindowSize, test_size=0.1, random_state=42)
    walkTrain, walkTest = train_test_split(walkingWindowSize, test_size=0.1, random_state=42)

    for i, dataset in enumerate(jumpTrain):
        Dataset41.create_dataset(f'jumpTrain_{i}', data=dataset)

    for i, dataset in enumerate(walkTrain):
        Dataset41.create_dataset(f'walkTrain_{i}', data=dataset)

    for i, dataset in enumerate(jumpTest):
        Dataset42.create_dataset(f'jumpTest_{i}', data=dataset)

    for i, dataset in enumerate(walkTrain):
        Dataset42.create_dataset(f'walkTest_{i}', data=dataset)

    print('done')