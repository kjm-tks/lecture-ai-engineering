# day5/演習3/tests/test_metrics.py
import pandas as pd

def test_metrics_file_exists():
    # metrics.csv が生成されていること
    import os
    assert os.path.isfile('metrics.csv')

def test_logistic_accuracy_threshold():
    df = pd.read_csv('metrics.csv')
    lr_acc = df.loc[df['model']=='LogisticRegression','accuracy'].iloc[0]
    assert lr_acc >= 0.75

def test_randomforest_beats_logistic():
    df = pd.read_csv('metrics.csv')
    rf_acc = df.loc[df['model']=='RandomForest','accuracy'].iloc[0]
    lr_acc = df.loc[df['model']=='LogisticRegression','accuracy'].iloc[0]
    assert rf_acc > lr_acc