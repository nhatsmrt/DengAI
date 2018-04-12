import numpy as np
import pandas as pd

def write_result(predictions, csv_name, sample_source, write_source):
    sample_csv = pd.read_csv(sample_source)
    sample_csv["total_cases"] = predictions
    sample_csv.to_csv(write_source + csv_name, sep=',', encoding='utf-8', index=False)
