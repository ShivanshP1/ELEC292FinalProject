import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing 

def movingAvgFilter(data):
    window_size=31
    df = pd.DataFrame(data)
    df = df.iloc[:, 1:-1]
    filteredData = df.rolling(window_size).mean()
    return filteredData

# def normalize(data):
#     sc = preprocessing.StandardScaler()
#     df = pd.DataFrame(data)
#     normalized_data = sc.fit_transform(df.T).T
#     return normalized_data

def normalize(features):
    """
    Normalizes the features using min-max scaling.

    Args:
        features (list): List of features to be normalized.

    Returns:
        numpy.ndarray: Normalized features.
    """
    if len(features) == 0:
        return np.array([])  # Return an empty array if no features are provided

    normalized_features = []
    for feature in features:
        # Reshape the feature to have a consistent shape
        if len(feature.shape) == 0:  # Scalar value, reshape it to (1, 1)
            feature = np.array([[feature]])
        elif len(feature.shape) == 1:  # 1D array, reshape it to (n, 1)
            feature = feature.reshape(-1, 1)

        min_val = np.min(feature)
        max_val = np.max(feature)
        normalized_feature = (feature - min_val) / (max_val - min_val)
        normalized_features.append(normalized_feature)
    return np.concatenate(normalized_features, axis=1)

processed_data = {}
with h5py.File('HDF5_Data.h5', 'r') as f:
    ShivData = f['Shivansh']
    NicoData = f['Nico']
    AidanData = f['Aidan']
    dataset_group = f['dataset']
    training_group = dataset_group['Train']
    testing_group = dataset_group['Test']
    print('done 0')
    processed_ShivData = movingAvgFilter(ShivData)
    processed_NicoData = movingAvgFilter(NicoData)
    processed_AidanData = movingAvgFilter(AidanData)
    processed_training_data = movingAvgFilter(training_group)
    processed_testing_data = movingAvgFilter(testing_group)
    print('done 1')
    normalized_ShivData = normalize(processed_ShivData)
    normalized_NicoData = normalize(processed_NicoData)
    normalized_AidanData = normalize(processed_AidanData)
    normalized_training_data = normalize(processed_training_data)
    normalized_testing_data = normalize(processed_testing_data)
    print('done 2')
    processed_data['Shivansh'] = normalized_ShivData
    processed_data['Nico'] = normalized_NicoData
    processed_data['Aidan'] = normalized_AidanData
    processed_data['Training'] = normalized_training_data
    processed_data['Testing'] = normalized_testing_data
    print('done 3')
# Now you can do whatever you want with the processed_data dictionary
# For example, you could save it to another HDF5 file:
with h5py.File('output_file.h5', 'w') as output_f:
    for key, value in processed_data.items():
        output_f[key] = value

print('done')
