import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, \
    RocCurveDisplay, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

# read dataset
dataset_name = 'winequalityN-lab6.csv'
dataset = pd.read_csv(dataset_name)
dataset.loc[dataset['quality'] <= 5, 'quality'] = 0
dataset.loc[dataset['quality'] >= 6, 'quality'] = 1
data = dataset.iloc[:, 1:-1]
labels = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=0)

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

# F1 score
f1 = f1_score(y_test, y_pred)

# plot ROC
fpr, tpr, _ = roc_curve(y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# find AUC
auc = roc_auc_score(y_test, y_clf_prob[:, 1])

# finally, print all elements
print("Part 1")
print("The AUC is: ", auc)
print('The Accuracy is: ', acc)
print("The F1 Score is: ", f1)

# part 2

# repeat of steps 1-5
dataset_name = 'winequalityN-lab6.csv'
dataset = pd.read_csv(dataset_name)
dataset.loc[dataset['quality'] <= 5, 'quality'] = 0
dataset.loc[dataset['quality'] >= 6, 'quality'] = 1
data = dataset.iloc[:, 1:-1]
labels = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=0)

l_reg = LogisticRegression(max_iter=10000)
pca = PCA(n_components=2)

# step 2
pca_pipe = make_pipeline(StandardScaler(), pca)

# step 3
X_train_pca = pca_pipe.fit_transform(X_train, y_train)
X_test_pca = pca_pipe.fit_transform(X_test, y_test)

# step 4
clf = make_pipeline(l_reg)

# step 5
clf.fit(X_train_pca, y_train)

y_pred_pca = clf.predict(X_test_pca)

# step 7
# decision boundary
disp = DecisionBoundaryDisplay.from_estimator(
clf, X_train_pca, response_method="predict",
xlabel='X1', ylabel='X2',
alpha=0.5,
)

# step 8
disp.ax_.scatter(X_train_pca[:, 0],X_train_pca[:, 1], c=y_train)
plt.show()

# get accuracy
acc = accuracy_score(y_test, y_pred_pca)

print("Part 2")
print('The Accuracy is: ', acc)
