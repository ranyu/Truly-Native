import pandas as pd


xgboost1 = pd.read_csv("xgboostChecked1.csv")
xgboost2 = pd.read_csv("xgboostChecked2.csv")

idx = xgboost1.file.values
prediction = (xgboost1.sponsored.values + xgboost2.sponsored.values) / 2
submission = pd.DataFrame({"file": idx, "sponsored": prediction})
submission.to_csv("ensemble.csv", index=False)
