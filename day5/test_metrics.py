# test_metrics.py
import pandas as pd
import pytest

METRIC_FILE = 'metrics.csv'

@pytest.fixture(scope='module')
def metrics():
    return pd.read_csv(METRIC_FILE, index_col=0)

def test_accuracy_above_threshold(metrics):
    # LogisticRegression の accuracy が 0.75 以上あるか
    assert metrics.loc['LogisticRegression','accuracy'] >= 0.75

def test_random_forest_beats_logreg(metrics):
    # RandomForest の accuracy が LogisticRegression を上回っているか
    rf_acc = metrics.loc['RandomForest','accuracy']
    lr_acc = metrics.loc['LogisticRegression','accuracy']
    assert rf_acc > lr_acc