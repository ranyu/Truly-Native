import pandas as pd


xgboost = pd.read_csv("xgboost.csv")
sample = pd.read_csv("sampleSubmission.csv")
sample = sample.drop("sponsored", axis=1)

xgb_map = dict()

for i, row in xgboost.iterrows():
    row["file"] = str(int(row["id"])) + "_raw_html.txt"
    xgb_map[row["file"]] = row["prediction"]

for i, row in sample.iterrows():
    sponsored = xgb_map[row["file"]]
    sample.set_value(i, "sponsored", sponsored)

sample.to_csv("xgboostChecked.csv", index=False)
