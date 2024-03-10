from sklearn.metrics import ndcg_score, classification_report, roc_auc_score


def get_metrics(model, X_test, y_test, verbose=True):
    """Возвращает метрики рассчитаные на основе тестовых выборок"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("nDCG score:", ndcg_score([y_test], [y_pred_proba]))
    if verbose:
        print("ROC AUC score:", roc_auc_score(y_test, y_pred_proba))
        print("Classification report:\n", classification_report(y_true=y_test, y_pred=y_pred))


def get_training_samples(df, features_to_drop=None):
    """Возвращает выборки из датафрейма"""
    not_features = ["search_id", "target"]
    if isinstance(features_to_drop, list):
        not_features = not_features + features_to_drop
    
    X_data = df.loc[:, [x not in not_features for x in df.columns]]
    y_data = df["target"].to_numpy()

    return X_data, y_data