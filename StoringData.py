import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split


with h5py.File('Data.h5', 'w') as hdf:
    #Creating the folders in the HDF5 
    G1 = hdf.create_group('/Shivansh')
    G2 = hdf.create_group('/Aidan')
    G3 = hdf.create_group('/Nico')
    G4_1 = hdf.create_group('/dataset/Train')
    G4_2 = hdf.create_group('/dataset/Test')


    #Reading the CSV files and storing them into the HDF5
    csv_path = 'ShivanshData/ShivWalking.csv'
    df = pd.read_csv(csv_path)
    G1.create_dataset('ShivWalking', data=df)

    csv_path = 'ShivanshData\ShivJumping.csv'
    df = pd.read_csv(csv_path)
    G1.create_dataset('ShivJumping', data=df)
    
    csv_path = 'AidanData\AidanWalking.csv'
    df = pd.read_csv(csv_path)
    G2.create_dataset('AidanWalking', data=df)

    csv_path = 'AidanData\AidanJumping.csv'
    df = pd.read_csv(csv_path)
    G2.create_dataset('AidanJumping', data=df)

    csv_path = 'NicoData\\NicoWalking.csv'
    df = pd.read_csv(csv_path)
    G3.create_dataset('NicoWalking', data=df)

    csv_path = 'NicoData\\NicoJumping.csv'
    df = pd.read_csv(csv_path)
    G3.create_dataset('NicoJumping', data=df)

    jumpData = ['ShivanshData\ShivJumping.csv', 'AidanData\AidanJumping.csv', 'NicoData\\NicoJumping.csv']
    walkData = ['ShivanshData/ShivWalking.csv', 'AidanData\AidanWalking.csv', 'NicoData\\NicoWalking.csv']

    jumping_dfs = [pd.read_csv(file) for file in jumpData]
    walking_dfs = [pd.read_csv(file) for file in walkData]

    jumping_df = pd.concat(jumping_dfs, axis=0, ignore_index=True)
    walk_df = pd.concat(walking_dfs, axis=0, ignore_index=True)



    # Divide signal into windows
    windowSize = 500  # 500 becuase 500 corresponds to 5 seconds
    numWindows1 = int(len(jumping_df) / windowSize)
    numWindows2 = int(len(walk_df) / windowSize)

    jumpingWindows = [jumping_df.iloc[i:i+windowSize] for i in range(0, len(jumping_df), windowSize) if
                      len(jumping_df.iloc[i:i + windowSize] == windowSize)]

    walkingWindows = [walk_df.iloc[i:i + windowSize] for i in range(0, len(walk_df), windowSize) if
                      len(walk_df.iloc[i:i + windowSize] == windowSize)]

    np.random.shuffle(jumpingWindows)
    np.random.shuffle(walkingWindows)

    jumpTrain, jumpTest = train_test_split(jumpingWindows, test_size=0.1, random_state=42)
    walkTrain, walkTest = train_test_split(walkingWindows, test_size=0.1, random_state=42)

    for i, dataset in enumerate(jumpTrain):
        G4_1.create_dataset(f'jumpTrain_{i}', data=dataset)

    for i, dataset in enumerate(walkTrain):
        G4_1.create_dataset(f'walkTrain_{i}', data=dataset)

    for i, dataset in enumerate(jumpTest):
        G4_2.create_dataset(f'jumpTest_{i}', data=dataset)

    for i, dataset in enumerate(walkTrain):
        G4_2.create_dataset(f'walkTest_{i}', data=dataset)

    print('done')