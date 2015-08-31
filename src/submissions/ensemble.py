import pandas as pd


logistic = pd.read_csv("logisticChecked.csv")
xgboost = pd.read_csv("xgboostChecked.csv")

idx = xgboost.file.values
prediction = logistic.sponsored.values*0.3 + xgboost.sponsored.values*0.7
submission = pd.DataFrame({"file": idx, "sponsored": prediction})
submission.to_csv("ensemble.csv", index=False)
