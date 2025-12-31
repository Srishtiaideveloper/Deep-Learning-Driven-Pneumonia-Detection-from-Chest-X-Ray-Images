import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, test_gen):
    preds = model.predict(test_gen)
    y_pred = (preds > 0.5).astype(int)
    y_true = test_gen.classes

    print(classification_report(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, preds))

    return confusion_matrix(y_true, y_pred)
