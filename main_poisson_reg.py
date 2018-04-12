import numpy as np
import pandas as pd
from pathlib import Path
from Source import write_result

from statsmodels.discrete.discrete_model import Poisson

# Define paths:
d = Path().resolve()
data_path = str(d) + "/Data"
predictions_path = str(d) + "/Predictions"
sample_submission_path = data_path + "/DengAI_Predicting_Disease_Spread_-_Submission_Format.csv"


# Read data:
df_train_features = pd.read_csv(data_path + "/dengue_features_train.csv").fillna(0)
df_train_labels = pd.read_csv(data_path + "/dengue_labels_train.csv")
df_test_features = pd.read_csv(data_path + "/dengue_features_test.csv").fillna(0)
features_list = ["ndvi_sw", "station_max_temp_c", "station_min_temp_c", "station_precip_mm", "station_diur_temp_rng_c", "precipitation_amt_mm", "reanalysis_dew_point_temp_k", "reanalysis_air_temp_k", "reanalysis_relative_humidity_percent", "reanalysis_specific_humidity_g_per_kg", "reanalysis_precip_amt_kg_per_m2", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_tdtr_k"]
n_features = len(features_list)

X_train = df_train_features[features_list].values
X_test = df_test_features[features_list].values

y_train = df_train_labels["total_cases"].values

# Model:
poisson_mod = Poisson(endog = y_train, exog = X_train).fit(maxiter = 61)

print(poisson_mod.summary())

predictions = poisson_mod.predict(X_test)
predictions_rounded = np.rint(predictions).astype(np.int64)
print(predictions_rounded)

write_result(predictions_rounded, "/poisson.csv", sample_source = sample_submission_path, write_source = predictions_path)