#imporiting librarie
from mod1bb.datas import preproceesingdata,show_scores,rmsle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestRegressor
#Data importing
df=preproceesingdata(r"C:\coding_stuff\project2\bluebook-for-bulldozers\Train\Train.csv")
#splitting
x=df.drop(["SalePrice"],axis=1)
x=x.drop(["SalePricemissing"],axis=1)
y=df["SalePrice"]
#####val#############
j=preproceesingdata("C:\\coding_stuff\\project2\\bluebook-for-bulldozers\\Valid.csv")
val_sol=pd.read_csv(r"C:\coding_stuff\project2\bluebook-for-bulldozers\ValidSolution.csv")
val_sol.drop(["Usage"],inplace=True,axis=1)
z=pd.merge(val_sol,j,"inner")
x_val=z.drop(["SalePrice"],axis=1)
y_val=z["SalePrice"]
#############################################
from  sklearn.model_selection import  RandomizedSearchCV
randf={"n_estimators":np.arange(10,200,10),
       "max_depth": np.arange(2,10,2),
       "min_samples_split": np.arange(2,20,2),
"min_samples_leaf": np.arange(1, 20, 2),
       "max_features":["auto","sqrt", "log2"],
       "max_samples":[10000],
       "n_jobs":[-1]}
ranscv=RandomizedSearchCV(RandomForestRegressor(),
                          param_distributions=randf,
                          n_iter=100,
                          cv=5,n_jobs=-1,
                          verbose=1)
ranscv.fit(x,y)
print(ranscv.best_params_)
print(ranscv.best_score_)
sc=show_scores(ranscv,x,x_val,y_val,y)
print(sc)

import pickle
filename=r"C:\coding_stuff\project2\trained_bul.pkl"
pickle.dump(ranscv, open(filename, 'wb'))





