# evaluate_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, f1_score, roc_auc_score

def load_data():
    df = pd.read_csv('data/titanic.csv')
    X = df.drop(['Survived','Name','Ticket','Cabin'], axis=1)
    y = df['Survived']
    # 前処理は省略
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate():
    X_train, X_test, y_train, y_test = load_data()
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest':       RandomForestClassifier(n_estimators=100),
        # ほかに試したいモデルを追加
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, preds),
            'f1'      : f1_score(y_test, preds),
            # 必要なら AUC も
            'auc'     : roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
        }
    # JSON や CSV ファイルとして保存
    pd.DataFrame(results).T.to_csv('metrics.csv', index=True)
    print(pd.DataFrame(results).T)

if __name__ == '__main__':
    evaluate()