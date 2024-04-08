import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import h5py
import joblib

# read dataset
TrainingData = None
with h5py.File('Features.h5', 'r') as f:
    TrainingData = f['Features']
    dataset = pd.DataFrame(TrainingData)
    print(dataset)
median = 4

dataset.loc[dataset[median] <= .38, median] = 0
dataset.loc[dataset[median] >= .38, median] = 1
X_train,X_test, y_train,y_test = train_test_split(dataset,dataset[4],test_size=0.1, shuffle=True, random_state=0)
scaler = StandardScaler()
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_clf_prob = clf.predict_proba(X_test)

# accuracy
acc = accuracy_score(y_test, y_pred)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# plot ROC
fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# find AUC
auc = roc_auc_score(y_test, y_clf_prob[:, 1])

print("The AUC is: ", auc)
print('The Accuracy is: ', acc)

# Save the model to disk
joblib.dump(clf, 'LRM.pkl')