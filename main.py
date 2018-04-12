import numpy as np
import pandas as pd
from pathlib import Path
from Source import write_result

from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression


# Define paths:
d = Path().resolve()
data_path = str(d) + "/Data"
predictions_path = str(d) + "/Predictions"
sample_submission_path = data_path + "/DengAI_Predicting_Disease_Spread_-_Submission_Format.csv"


# Read data:
df_train_features = pd.read_csv(data_path + "/dengue_features_train.csv").fillna(0)
df_train_labels = pd.read_csv(data_path + "/dengue_labels_train.csv")
df_test_features = pd.read_csv(data_path + "/dengue_features_test.csv").fillna(0)
features_list = ["station_max_temp_c", "station_min_temp_c", "station_precip_mm", "station_diur_temp_rng_c", "precipitation_amt_mm", "reanalysis_dew_point_temp_k", "reanalysis_air_temp_k", "reanalysis_relative_humidity_percent", "reanalysis_specific_humidity_g_per_kg", "reanalysis_precip_amt_kg_per_m2", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_tdtr_k", "ndvi_se", "ndvi_sw", "ndvi_ne", "ndvi_nw"]
n_features = len(features_list)

X_train = df_train_features[features_list].values
X_test = df_test_features[features_list].values

y_train = df_train_labels["total_cases"].values

# Preprocess data:
lb = LabelBinarizer()
y_train_enc = lb.fit_transform(y_train)
print(y_train_enc.shape)


# Baseline Model:
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Write Result:
write_result(predictions, "/baseline.csv", sample_source = sample_submission_path, write_source = predictions_path)