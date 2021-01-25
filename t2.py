import pandas as pd
from pandas import DataFrame
val=pd.read_csv(r"C:\coding_stuff\project2\bluebook-for-bulldozers\Valid.csv")
val_sol=pd.read_csv(r"C:\coding_stuff\project2\bluebook-for-bulldozers\ValidSolution.csv")
val_sol.drop(["Usage"],inplace=True,axis=1)
print(val)
val.to_csv(r"C:\coding_stuff\project2\val.csv")
print("."*50)
print(val_sol)
print("."*50)

z=pd.merge(val,val_sol)
z.to_csv(r"C:\coding_stuff\project2\z1.csv",index=False)
print(z)