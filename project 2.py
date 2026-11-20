import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 1. Load the dataset
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# 2. Stratified 80/20 Train/Test Split (Zero Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================
# PIPELINE 1: Tuned Logistic Regression (StandardScaler -> SMOTE -> LR)
# ==========================================
print("Training Tuned Logistic Regression Pipeline...")
lr_best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(k_neighbors=5, random_state=42)),
    ('classifier', LogisticRegression(C=0.01, max_iter=1000, random_state=42))
])
lr_best_pipeline.fit(X_train, y_train)

lr_preds = lr_best_pipeline.predict(X_test)
lr_probs = lr_best_pipeline.predict_proba(X_test)[:, 1]

# ==========================================
# PIPELINE 2: Random Forest (SMOTE -> RF, No Scaler Needed)
# ==========================================
print("\nTuning Random Forest Pipeline (this will take a few minutes)...")
rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))
])

rf_param_grid = {
    'smote__k_neighbors': [3, 5, 7],
    'classifier__max_depth': [10, 20, None]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    param_grid=rf_param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)
rf_grid.fit(X_train, y_train)

rf_preds = rf_grid.predict(X_test)
rf_probs = rf_grid.predict_proba(X_test)[:, 1]

print("\n--- Random Forest Results ---")
print("Best Params:", rf_grid.best_params_)
print(classification_report(y_test, rf_preds, digits=4))

# ==========================================
# FINAL COMPARISON TABLE (Fraud Class = 1)
# ==========================================
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Precision (Fraud)': [precision_score(y_test, lr_preds), precision_score(y_test, rf_preds)],
    'Recall (Fraud)': [recall_score(y_test, lr_preds), recall_score(y_test, rf_preds)],
    'F1-Score (Fraud)': [f1_score(y_test, lr_preds), f1_score(y_test, rf_preds)],
    'ROC-AUC': [roc_auc_score(y_test, lr_probs), roc_auc_score(y_test, rf_probs)]
})

print("\n=== FINAL MODEL COMPARISON (UNTOUCHED TEST SET) ===")
print(comparison.to_string(index=False))

# ==========================================
# VISUALIZATION 1: Side-by-Side Confusion Matrices
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression\nConfusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Random Forest\nConfusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.close()

# ==========================================
# VISUALIZATION 2: ROC-AUC Curve Comparison
# ==========================================
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_probs):.4f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probs):.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC-AUC Curve: Fraud Detection Pipelines')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300)
plt.close()

print("\nCharts saved successfully: 'confusion_matrices.png' and 'roc_curves.png'")