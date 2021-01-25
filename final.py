from mod1bb.datas import preproceesingdata,show_scores
import pandas as pd
import pickle
#load
loaded_model = pickle.load(open(r"C:\coding_stuff\project2\trained_bul.pkl", 'rb'))
test=preproceesingdata(r"C:\coding_stuff\project2\bluebook-for-bulldozers\Test.csv")
test_preds = loaded_model.predict(test)
df_preds = pd.DataFrame()
df_preds["SalesID"] = test["SalesID"]
df_preds["SalesPrice"] = test_preds
df_preds.to_csv(r"C:\coding_stuff\project2_predictions.csv", index=False)
print(df_preds)