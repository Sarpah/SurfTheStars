import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

koi_df_raw = pd.read_csv("cumulative_2025.10.03_23.25.17.csv")

parameter_keys = ['koi_period', 'koi_prad', 'koi_duration', 'koi_depth', 'koi_prad_err1', 'koi_prad_err2', 'koi_duration_err1', 'koi_duration_err2', 'koi_depth_err1', 'koi_depth_err2']

# Filter out entries with missing data
# idx_drop = np.where(pd.isnull(koi_df_raw['koi_period']))
idx_drop_prad = np.where(pd.isnull(koi_df_raw['koi_prad']))
idx_drop_depth = np.where(koi_df_raw['koi_depth'] == 0.0)
idx_drop_duration_err = np.where(pd.isnull(koi_df_raw['koi_duration_err1']))
# idx_drop_ingress = np.where(pd.isnull(koi_df_raw['koi_ingress']))
# idx_drop = np.where(pd.isnull(koi_df_raw['koi_duration']))
# for idx, row in koi_df_raw.iterrows():
#     # Check for missing values in 'koi_prad'
#     if row['koi_prad'] == "":
#         idx_drop.append(idx)

print(f"Planet Count: {len(koi_df_raw)}" )
print(f"Dropping {len(np.unique(idx_drop_prad))} entries (koi_prad == null)")
print(f"Dropping {len(np.unique(idx_drop_depth))} entries (koi_depth == 0.0)")
print(f"Dropping {len(np.unique(idx_drop_duration_err))} entries (idx_drop_depth_err == null)")
# print(f"Dropping {len(np.unique(idx_drop_ingress))} entries (koi_ingress == null)")
koi_df = koi_df_raw.drop(np.unique(np.hstack([idx_drop_prad, idx_drop_depth, idx_drop_duration_err])))

fig, (ax_1, ax_2, ax_3, ax_4) = plt.subplots(1, 4)

sns.histplot(data=koi_df['koi_period'].to_numpy(), log_scale=True, ax=ax_1)
ax_1.set_xlabel("Orbital Period [days]")

sns.histplot(data=koi_df['koi_duration'].to_numpy(), log_scale=True, ax=ax_2)
ax_2.set_xlabel("Transit Duration [hrs]")

sns.histplot(data=koi_df['koi_prad'].to_numpy(), log_scale=True, ax=ax_3)
ax_3.set_xlabel("Planetary Radius [Earth radii]")

sns.histplot(data=koi_df['koi_depth'].to_numpy(), log_scale=True, ax=ax_4)
ax_4.set_xlabel("Transit Depth [ppm]")

# plt.show()

# Apply scaling to data
X = np.vstack([koi_df[parameter_keys[0]].to_numpy()])
for i in range(1, len(parameter_keys)):
    X = np.vstack([X, koi_df[parameter_keys[i]].to_numpy()])

state_map = {'CONFIRMED': True, 'FALSE POSITIVE': False, 'CANDIDATE': False, 'NOT DISPOSITIONED': False}
y = koi_df['koi_disposition'].map(state_map).to_numpy()

adaboost_classifier = AdaBoostClassifier(random_state=38)
X_train, X_test, y_train, y_test = train_test_split(X.transpose(), y, test_size=0.3, random_state=38)

adaboost_classifier.fit(X_train, y_train)

# Make predictions
predictions = adaboost_classifier.predict(X_test)
print(f"Test accuracy: {100 * np.sum(predictions == y_test) / len(koi_df)} %")