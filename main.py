import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#col_names = ['toi','tid','tfopwg_disp','pl_tranmid','pl_tranmiderrtot','pl_orbper','pl_orbpererrtot','pl_trandurh','pl_trandurherrtot','pl_trandep','pl_trandeperrtot','pl_rade','pl_radeerrtot','pl_insol','pl_insolerrtot','pl_eqt','pl_eqterrtot','st_tmag','st_tmagerrtot','st_dist','st_disterrtot','st_teff','st_tefferrtot','st_logg','st_loggerrtot','st_rad','st_raderrtot']
df= pd.read_csv("input.csv")

x = df.drop(['tfopwg_disp'], axis=1)
y = df['tfopwg_disp']
print(f'x shape: {x.shape} | y shape: {y.shape}')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()