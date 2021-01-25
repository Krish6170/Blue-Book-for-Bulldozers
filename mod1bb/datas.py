import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
def preproceesingdata(path):
    # Data importing
    vdf = pd.read_csv(path, low_memory=False,
                       parse_dates=["saledate"])


    vdf.sort_values(by=["saledate"], inplace=True, ascending=True, ignore_index=True)
    # filling numerical (first filling numerical as after turning into categorical the codes also behave as numerical values)
    for label, content in vdf.items():
        if pd.api.types.is_numeric_dtype(content):
            vdf[label + "missing"] = pd.isnull(content).astype("int")
            if pd.isnull(content).sum():

                vdf[label] = content.fillna(content.median())

    # turning strings to categorical and filling missing value
    for label, content in vdf.items():
        if pd.api.types.is_string_dtype(content):
            vdf[label + "missing"] = pd.isnull(content).astype("int")
            vdf[label] = content.astype("category").cat.as_ordered()
            vdf[label ] = vdf[label].cat.codes + 1
    vdf["saleYear"] = vdf.saledate.dt.year
    vdf["saleMonth"] = vdf.saledate.dt.month
    vdf["saleDay"] = vdf.saledate.dt.day
    vdf["saleDayOfWeek"] = vdf.saledate.dt.dayofweek
    vdf["saleDayOfYear"] = vdf.saledate.dt.dayofyear
    # Now we've enriched our DataFrame with date time features, we can remove 'saledate'
    vdf.drop("saledate", axis=1, inplace=True)
    x = vdf
    return x
def rmsle(y_test, y_preds):
    """
    Caculates root mean squared log error between predictions and
    true labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))
def show_scores(model,x,x_val,y_val,y):
    '''

    :param model:
    :param x:true label
    :param x_val:
    :param y_val:
    :param y:
    :return:
    '''
    train_preds = model.predict(x)
    val_preds = model.predict(x_val)
    scores = {"Training MAE": mean_absolute_error(y, train_preds),
              "Valid MAE": mean_absolute_error(y_val, val_preds),
              "Training RMSLE": rmsle(y, train_preds),
              "Valid RMSLE": rmsle(y_val, val_preds),
              "Training R^2": r2_score(y, train_preds),
              "Valid R^2": r2_score(y_val, val_preds)}
    return scores

if __name__=="main":
    preproceesingdata()
    show_scores()
    rmsle()