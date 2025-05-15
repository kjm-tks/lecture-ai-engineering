# day5/演習3/evaluate_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, f1_score, roc_auc_score

def load_data():
    # Titanic CSV はリポジトリの data/ 以下に置く想定
    df = pd.read_csv('data/titanic.csv')
    # 必要に応じて欠損値処理やカテゴリ変換を行ってください
    X = df[['Pclass','Age','SibSp','Parch','Fare']].fillna(0)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate():
    X_train, X_test, y_train, y_test = load_data()
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest'      : RandomForestClassifier(n_estimators=100),
    }
    records = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        records.append({
            'model'   : name,
            'accuracy': accuracy_score(y_test, preds),
            'f1'      : f1_score(y_test, preds),
            'auc'     : roc_auc_score(y_test, m.predict_proba(X_test)[:,1]),
        })
    df = pd.DataFrame(records)
    df.to_csv('metrics.csv', index=False)
    print(df)

if __name__ == '__main__':
    evaluate()