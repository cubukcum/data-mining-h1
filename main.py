# KDD - Pima Diabetes (Python)
# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1) Data Selection
df = pd.read_csv('diabetes.csv')   # Kaggle'dan indirdiğin dosya
print("Şekil:", df.shape)
print(df.head())

# 2) Preprocessing (eksikleri/aykırıları düzeltme)
# Pima datasında Glucose, BloodPressure, SkinThickness, Insulin, BMI sütunlarında 0 bazen 'missing' demektir.
cols_zero_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols_zero_missing] = df[cols_zero_missing].replace(0, np.nan)
print(df.isna().sum())

# 3) Transformation (impute + scale)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 4) Data Mining (model eğitimi)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                    stratify=y, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 5) Evaluation — cross validation (f1 ortalaması)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    f1_cv = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
    print(f"{name} CV F1 (5-fold): mean={f1_cv.mean():.4f}, std={f1_cv.std():.4f}")

# 6) Knowledge Presentation — ROC örneği (RandomForest)
rf = models['RandomForest']
y_prob = rf.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'RF (AUC = {roc_auc:.3f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# (İstersen results dict'ini CSV'ye kaydet)
res_df = pd.DataFrame(results).T
res_df.to_csv('model_metrics_summary.csv')
print("\nMetrics summary saved to model_metrics_summary.csv")
