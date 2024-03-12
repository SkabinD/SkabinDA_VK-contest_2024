from imblearn.over_sampling import SMOTE
from sklearn.metrics import ndcg_score, classification_report, roc_auc_score


def get_metrics(model, X_test, y_test, verbose=True, return_values=False):
    """Возвращает метрики рассчитаные на основе тестовых выборок"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    score = ndcg_score([y_test], [y_pred_proba])

    print("nDCG score:", score)
    if verbose:
        print("ROC AUC score:", roc_auc_score(y_test, y_pred_proba))
        print("Classification report:\n", classification_report(y_true=y_test, y_pred=y_pred))
    
    if return_values:
        return y_pred, y_pred_proba, score


def get_training_samples(df, features_to_drop=None, synthetic=False):
    """Возвращает выборки из датафрейма"""
    not_features = ["search_id", "target"]
    if isinstance(features_to_drop, list):
        not_features = not_features + features_to_drop
    
    X_data = df.loc[:, [x not in not_features for x in df.columns]]
    y_data = df["target"].to_numpy()

    # Синтетическая балансировка классов
    if synthetic:
        return SMOTE().fit_resample(X_data, y_data)

    return X_data, y_data