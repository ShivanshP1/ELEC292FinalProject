import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


#Feature Extraction
#--------------------------------------------------------------------------------------------------------------
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
        print(ds)
        features = {}
        features['max'] = np.max(ds[4])
        features['min'] = np.min(ds[4])
        features['range'] = np.max(ds[4]) - np.min(ds[4])
        features['mean'] = np.mean(ds[4])
        features['median'] = np.median(ds[4])
        features['variance'] = np.var(ds[4])
        return features

# Path to HDF5 file
file_path = 'Data.h5'
datasets = check_hdf5_datasets(file_path)
# print("Datasets found in the HDF5 file:")
# print(datasets)

Trainingfeatures = []
for datasetKey, datasetValue in datasets.items():
    Trainingfeatures.append(FeatureExtraction(datasetValue))

features_DF= pd.DataFrame(Trainingfeatures)
feature_normalized = (features_DF - features_DF.min()) / (features_DF.max()-features_DF.min())

# print("Processed features:", Trainingfeatures)
print(feature_normalized , '\n' , "done")

with h5py.File('Features.h5', 'w') as output_f:
    G1 = output_f.create_dataset("Features", data=feature_normalized)