import h5py
import numpy as np
import pandas as pd
#---------------------------------------------------------------------------------------------------------------------
def movingAvgFilter(data):
    window_size=5
    df = pd.DataFrame(data)
    filteredData = df.rolling(window_size).mean()
    filteredData = filteredData.fillna(value=0)
    return filteredData
#---------------------------------------------------------------------------------------------------------------------
def normalize(data):
    normalized_data =()
    newData = np.array(data)
    normalized_data = (newData - newData.min()) / (newData.max() - newData.min())
    print(normalized_data, "\n")
    return normalized_data
#----------------------------------------------------------------------------------------------------------------------
with h5py.File('Data.h5', 'r') as f:
    with h5py.File('Processed.h5', 'w') as output_f:
        #----------------------------------------------------------
        ShivJData = f['Shivansh/ShivJumping']
        processed_ShivJData = movingAvgFilter(ShivJData)
        normalized_ShivJData = normalize(processed_ShivJData)
        ShivWData = f['Shivansh/ShivWalking']
        processed_ShivWData = movingAvgFilter(ShivWData)
        normalized_ShivWData = normalize(processed_ShivWData)
        G1 = output_f.create_group('/Shivansh')
        G1.create_dataset('ShivWalking', data=normalized_ShivWData)
        G1.create_dataset('ShivJumping', data=normalized_ShivJData)
        #----------------------------------------------------------
        NicoWData = f['Nico/NicoWalking']
        processed_NicoWData = movingAvgFilter(NicoWData)
        normalized_NicoWData = normalize(processed_NicoWData)
        NicoJData = f['Nico/NicoJumping']
        processed_NicoJData = movingAvgFilter(NicoJData)
        normalized_NicoJData = normalize(processed_NicoJData)
        G2 = output_f.create_group('/Aidan')
        G2.create_dataset('NicoWalking', data=normalized_NicoWData)
        G2.create_dataset('NicoJumping', data=normalized_NicoJData)
        #----------------------------------------------------------
        AidanJData = f['Aidan/AidanJumping']
        processed_AidanJData = movingAvgFilter(AidanJData)
        normalized_AidanJData = normalize(processed_AidanJData)
        AidanWData = f['Aidan/AidanWalking']
        processed_AidanWData = movingAvgFilter(AidanWData)
        normalized_AidanWData = normalize(processed_AidanWData)
        G3 = output_f.create_group('/Nico')
        G3.create_dataset('AidanWalking', data=normalized_AidanWData)
        G3.create_dataset('AidanJumping', data=normalized_AidanJData)
        #----------------------------------------------------------
        dataset_group = f['dataset']
        training_group = dataset_group['Train']
        processed_training_data = []
        normalized_training_data = []
        testing_group = dataset_group['Test']
        processed_testing_data = []
        normalized_testing_data = []
        G4_1 = output_f.create_group('/dataset/Train')
        G4_2 = output_f.create_group('/dataset/Test')
        
        for x in training_group:
            trainingSetLocation = ("dataset/Train/" + x)
            trainingTrainData = f[trainingSetLocation]

            processed_TrainingData = movingAvgFilter(trainingTrainData)
            normalized_TrainingData = normalize(processed_TrainingData)

            G4_1.create_dataset(x,data=normalized_TrainingData)
    
        for x in testing_group:        
            testSetLocation = ("dataset/Test/" + x)
            trainingTestData = f[trainingSetLocation]

            processed_TestData = movingAvgFilter(trainingTestData)
            normalized_TestData = normalize(processed_TestData)

            G4_2.create_dataset(x,data=normalized_TestData)
        #----------------------------------------------------------
print('done')