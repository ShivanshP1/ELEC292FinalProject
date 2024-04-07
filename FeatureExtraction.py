import h5py
import pandas as pd
import numpy as np

def check_hdf5_datasets(file_path):
    group_path = 'dataset/Train' 
    datasets = {}
    with h5py.File(file_path, 'r') as f:
        def add_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[:]
        group = f[group_path]
        group.visititems(add_dataset)
    return datasets

def FeatureExtraction(ds):   
        features = {}
        features['max'] = np.max(ds)
        features['min'] = np.min(ds)
        features['range'] = np.max(ds) - np.min(ds)
        features['mean'] = np.mean(ds)
        features['median'] = np.median(ds)
        features['variance'] = np.var(ds)
        return features

# Path to HDF5 file
file_path = 'Data.h5'
datasets = check_hdf5_datasets(file_path)
# print("Datasets found in the HDF5 file:")
# print(datasets)

Tfeatures = []
for datasetKey, datasetValue in datasets.items():
    Tfeatures.append(FeatureExtraction(datasetValue))

features_DF= pd.DataFrame(Tfeatures)
feature_normalized = (features_DF - features_DF.min()) / (features_DF.max()-features_DF.min())

print("Processed features:", Tfeatures)
print(feature_normalized , '\n' , "done")