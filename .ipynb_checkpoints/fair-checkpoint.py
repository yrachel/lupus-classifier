import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from functools import partial
from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold, GridSearchCV)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, roc_curve, mean_squared_error, r2_score, accuracy_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_DIR = '/gpfs/home/lv2255/scratch'
OUT      = '/gpfs/home/lv2255/scratch/sle_outputs_fin'
os.makedirs(OUT, exist_ok=True)
print(f"Saving all outputs to: {OUT}")

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_importances(model, gene_names):
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=gene_names).sort_values(ascending=False)
    elif hasattr(model, 'coef_'):
        return pd.Series(np.abs(model.coef_[0]), index=gene_names).sort_values(ascending=False)
    return None

def get_c_metrics(model, X, y_test, s, name=""):
    y_pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = y_pred.astype(float)
        auc = roc_auc_score(y_test, y_prob)
    dpd_race = demographic_parity_difference(y_test, y_pred, sensitive_features=s['race'])
    dpd_sex  = demographic_parity_difference(y_test, y_pred, sensitive_features=s['sex'])
    eod_race = equalized_odds_difference(y_test, y_pred, sensitive_features=s['race'])
    eod_sex  = equalized_odds_difference(y_test, y_pred, sensitive_features=s['sex'])
    mf_race = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=s['race'])
    mf_sex  = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, sensitive_features=s['sex'])
    print(f"\n[{name}] Test AUC: {auc:.4f}")
    print(f"[{name}] DPD Race: {dpd_race:.4f} | DPD Sex: {dpd_sex:.4f}")
    print(f"[{name}] EOD Race: {eod_race:.4f} | EOD Sex: {eod_sex:.4f}")
    print(f"[{name}] Selection rate by race:\n{mf_race.by_group}")
    print(f"[{name}] Selection rate by sex:\n{mf_sex.by_group}")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'SLE']))
    return {
        'model':      model,
        'y_pred':     y_pred,
        'y_prob':     y_prob,
        'test_auc':   auc,
        'dpd_race':   dpd_race,
        'dpd_sex':    dpd_sex,
        'eod_race':   eod_race,
        'eod_sex':    eod_sex,
        'sr_by_race': mf_race.by_group.to_dict(),
        'sr_by_sex':  mf_sex.by_group.to_dict(),
    }

def get_r_metrics(model, X, y_test, s, name=""):
    y_pred = model.predict(X)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    median_sledai = np.median(y_test)
    y_binary      = (y_test > median_sledai).astype(int)
    y_pred_binary = (y_pred > median_sledai).astype(int)
    dpd_race = demographic_parity_difference(y_binary, y_pred_binary, sensitive_features=s['race'])
    dpd_sex  = demographic_parity_difference(y_binary, y_pred_binary, sensitive_features=s['sex'])
    mf_race = MetricFrame(metrics=selection_rate, y_true=y_binary, y_pred=y_pred_binary, sensitive_features=s['race'])
    mf_sex  = MetricFrame(metrics=selection_rate, y_true=y_binary, y_pred=y_pred_binary, sensitive_features=s['sex'])
    print(f"\n[{name}] Test RMSE: {rmse:.4f} | R²: {r2:.4f}")
    print(f"[{name}] DPD Race: {dpd_race:.4f} | DPD Sex: {dpd_sex:.4f}")
    print(f"[{name}] High SLEDAI rate by race:\n{mf_race.by_group}")
    print(f"[{name}] High SLEDAI rate by sex:\n{mf_sex.by_group}")
    return {
        'model':      model,
        'y_pred':     y_pred,
        'rmse':       rmse,
        'r2':         r2,
        'dpd_race':   dpd_race,
        'dpd_sex':    dpd_sex,
        'sr_by_race': mf_race.by_group.to_dict(),
        'sr_by_sex':  mf_sex.by_group.to_dict(),
    }

def train_fair_model(base_model, X_train, y_train, s_train):
    fair = ExponentiatedGradient(
        estimator=base_model,
        constraints=DemographicParity(),
        eps=0.05,
        max_iter=50,
    )
    fair.fit(X_train, y_train, sensitive_features=s_train['race'])
    return fair

# ─────────────────────────────────────────────
# 0. LOAD DATA
# ─────────────────────────────────────────────
print("="*60)
print("LOADING DATA")
print("="*60)

expr = pd.read_csv(f'{DATA_DIR}/expr_clean.csv', index_col=0)
meta = pd.read_csv(f'{DATA_DIR}/meta_clean.csv', index_col=0)

shared = expr.index.intersection(meta.index)
expr = expr.loc[shared]
meta = meta.loc[shared]

print(f"Expression matrix: {expr.shape}")
print(f"Metadata: {meta.shape}")
print(f"Label distribution:\n{meta['label'].value_counts()}")

X          = expr.values
y_clf      = meta['label'].values
gene_names = expr.columns.tolist()

s = pd.DataFrame(index=meta.index)
s['race'] = np.where(meta['race'] == 'white', 0, 1)
s['sex']  = np.where(meta['sex'] == 'Female', 1, 0)

print(f"\nRace distribution (0=white, 1=non-white):\n{s['race'].value_counts()}")
print(f"Sex distribution (1=female, 0=male):\n{s['sex'].value_counts()}")

# ─────────────────────────────────────────────
# PART 1: CLASSIFICATION (Healthy vs SLE)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PART 1: CLASSIFICATION — Healthy vs SLE")
print("="*60)

X_train_c, X_test_c, y_train_c, y_test_c, s_train_c, s_test_c = train_test_split(
    X, y_clf, s, test_size=0.2, random_state=42, stratify=y_clf
)
s_train_c = s_train_c.reset_index(drop=True)
s_test_c  = s_test_c.reset_index(drop=True)

print(f"Train: {len(X_train_c)} | Test: {len(X_test_c)}")
print(f"Train dist: {pd.Series(y_train_c).value_counts().to_dict()}")
print(f"Test dist:  {pd.Series(y_test_c).value_counts().to_dict()}")

sm = SMOTE(random_state=42)
X_train_c_res, y_train_c_res = sm.fit_resample(X_train_c, y_train_c)

# SMOTE doubles the minority class — rebuild sensitive features to match
# duplicate the original s_train_c to cover SMOTE-expanded dataset
n_orig = len(s_train_c)
n_res  = len(y_train_c_res)
repeats = (n_res // n_orig) + 1
s_train_c_res = pd.concat([s_train_c] * repeats, ignore_index=True).iloc[:n_res].reset_index(drop=True)

print(f"After SMOTE: {pd.Series(y_train_c_res).value_counts().to_dict()}")

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c_res)
X_test_c_scaled  = scaler_c.transform(X_test_c)

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

clf_grids = {
    'Random Forest': (
        RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        {
            'n_estimators':      [200, 500],
            'max_depth':         [5, 10, 20],
            'min_samples_split': [5, 10, 20],
            'max_features':      ['sqrt', 'log2'],
            'max_samples':       [0.7, 0.8, 1.0],
        }
    ),
    'XGBoost': (
        xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
        {
            'n_estimators':     [200, 500],
            'max_depth':        [4, 6, 8],
            'learning_rate':    [0.01, 0.05, 0.1],
            'subsample':        [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'scale_pos_weight': [(y_train_c_res==0).sum() / (y_train_c_res==1).sum()],
        }
    ),
    'Logistic Regression': (
        LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
        {
            'C':       [0.001, 0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver':  ['saga'],
        }
    ),
}

clf_results = {}

for name, (model, param_grid) in clf_grids.items():
    print(f"\n{'='*40}\n--- {name} ---")

    X_tr = X_train_c_scaled if name == 'Logistic Regression' else X_train_c_res
    X_te = X_test_c_scaled  if name == 'Logistic Regression' else X_test_c

    gs = GridSearchCV(model, param_grid, cv=cv_strat, scoring='roc_auc',
                      n_jobs=-1, verbose=1, return_train_score=True)
    gs.fit(X_tr, y_train_c_res)

    print(f"Best params: {gs.best_params_}")
    print(f"Best CV AUC: {gs.best_score_:.4f}")

    pd.DataFrame(gs.cv_results_).sort_values('rank_test_score').to_csv(
        f'{OUT}/clf_{name.replace(" ","_")}_grid_results.csv', index=False
    )

    clf_results[name] = get_c_metrics(gs.best_estimator_, X_te, y_test_c, s_test_c, name=name)
    clf_results[name]['best_params'] = gs.best_params_
    clf_results[name]['best_cv_auc'] = gs.best_score_

    print(f"\nTraining fair version of {name}...")
    try:
        fair_model = train_fair_model(gs.best_estimator_, X_tr, y_train_c_res, s_train_c_res)
        clf_results[f'{name} (Fair)'] = get_c_metrics(fair_model, X_te, y_test_c, s_test_c, name=f'{name} Fair')
        clf_results[f'{name} (Fair)']['best_params'] = gs.best_params_
        clf_results[f'{name} (Fair)']['best_cv_auc'] = gs.best_score_
    except Exception as e:
        print(f"Fair model failed for {name}: {e}")

# ROC curves
plt.figure(figsize=(10, 6))
for name, res in clf_results.items():
    fpr, tpr, _ = roc_curve(y_test_c, res['y_prob'])
    ls = '--' if 'Fair' in name else '-'
    plt.plot(fpr, tpr, linestyle=ls, label=f"{name} (AUC={res['test_auc']:.4f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Classification (Original vs Fair)')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/clf_roc_curves.png', dpi=150)
plt.close()

# DPD comparison plot
orig_models = [k for k in clf_results if 'Fair' not in k]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, attr, title in zip(axes, ['dpd_race', 'dpd_sex'], ['DPD by Race', 'DPD by Sex']):
    names    = []
    dpd_orig = []
    dpd_fair = []
    for name in orig_models:
        names.append(name)
        dpd_orig.append(abs(clf_results[name][attr]))
        fair_key = f'{name} (Fair)'
        dpd_fair.append(abs(clf_results[fair_key][attr]) if fair_key in clf_results else np.nan)
    x = np.arange(len(names))
    ax.bar(x - 0.2, dpd_orig, 0.4, label='Original')
    ax.bar(x + 0.2, dpd_fair, 0.4, label='Fair')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('|DPD|')
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/clf_dpd_comparison.png', dpi=150)
plt.close()

# Best classifier importances
best_clf_name = max(orig_models, key=lambda k: clf_results[k]['test_auc'])
best_clf      = clf_results[best_clf_name]['model']
print(f"\nBest classifier: {best_clf_name} (AUC={clf_results[best_clf_name]['test_auc']:.4f})")

clf_importances = get_importances(best_clf, gene_names)
if clf_importances is not None:
    top_clf_genes = clf_importances.head(20)
    imp_label = 'Coefficient Magnitude' if hasattr(best_clf, 'coef_') else 'Feature Importance'
    plt.figure(figsize=(10, 6))
    top_clf_genes.plot(kind='barh')
    plt.title(f'Top 20 Genes — Classification ({best_clf_name})')
    plt.xlabel(imp_label)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUT}/clf_feature_importance.png', dpi=150)
    plt.close()
    clf_importances.to_csv(f'{OUT}/clf_all_gene_importances.csv', header=['importance'])
    print(f"\nTop 20 classification genes:\n{top_clf_genes}")

# ─────────────────────────────────────────────
# PART 2: SLEDAI REGRESSION (SLE only)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PART 2: SLEDAI REGRESSION — SLE patients only")
print("="*60)

combined_mask = (meta['label'] == 1) & meta['sledai_at_baseline'].notna()
X_sle    = expr.loc[combined_mask].values
y_sledai = pd.to_numeric(meta.loc[combined_mask, 'sledai_at_baseline'].values, errors='coerce')
s_sle    = s.loc[combined_mask].reset_index(drop=True)

print(f"SLE patients with SLEDAI: {len(X_sle)}")
print(f"SLEDAI range: {y_sledai.min():.0f} - {y_sledai.max():.0f}")
print(f"SLEDAI mean ± std: {y_sledai.mean():.2f} ± {y_sledai.std():.2f}")

X_train_r, X_test_r, y_train_r, y_test_r, s_train_r, s_test_r = train_test_split(
    X_sle, y_sledai, s_sle, test_size=0.2, random_state=42
)
s_train_r = s_train_r.reset_index(drop=True)
s_test_r  = s_test_r.reset_index(drop=True)

scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled  = scaler_r.transform(X_test_r)

cv_kf = KFold(n_splits=5, shuffle=True, random_state=42)

reg_grids = {
    'Random Forest': (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            'n_estimators':      [200, 500],
            'max_depth':         [5, 10, 20],
            'min_samples_split': [5, 10, 20],
            'max_features':      ['sqrt', 'log2'],
            'max_samples':       [0.7, 0.8, 1.0],
        }
    ),
    'XGBoost': (
        xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
        {
            'n_estimators':     [200, 500],
            'max_depth':        [4, 6, 8],
            'learning_rate':    [0.01, 0.05, 0.1],
            'subsample':        [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
        }
    ),
    'Ridge Regression': (
        Ridge(),
        {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        }
    ),
}

reg_results = {}

for name, (model, param_grid) in reg_grids.items():
    print(f"\n{'='*40}\n--- {name} ---")

    X_tr = X_train_r_scaled if name == 'Ridge Regression' else X_train_r
    X_te = X_test_r_scaled  if name == 'Ridge Regression' else X_test_r

    gs = GridSearchCV(model, param_grid, cv=cv_kf, scoring='r2',
                      n_jobs=-1, verbose=1, return_train_score=True)
    gs.fit(X_tr, y_train_r)

    print(f"Best params: {gs.best_params_}")
    print(f"Best CV R²: {gs.best_score_:.4f}")

    pd.DataFrame(gs.cv_results_).sort_values('rank_test_score').to_csv(
        f'{OUT}/reg_{name.replace(" ","_")}_grid_results.csv', index=False
    )

    reg_results[name] = get_r_metrics(gs.best_estimator_, X_te, y_test_r, s_test_r, name=name)
    reg_results[name]['best_params'] = gs.best_params_
    reg_results[name]['best_cv_r2']  = gs.best_score_

    print(f"\nTraining fair version of {name}...")
    try:
        fair_model = train_fair_model(gs.best_estimator_, X_tr, y_train_r, s_train_r)
        clf_results[f'{name} (Fair)'] = get_r_metrics(fair_model, X_te, y_test_r, s_test_r, name=f'{name} Fair')
        clf_results[f'{name} (Fair)']['best_params'] = gs.best_params_
        clf_results[f'{name} (Fair)']['best_cv_auc'] = gs.best_score_
    except Exception as e:
        print(f"Fair model failed for {name}: {e}")

# Predicted vs actual
orig_reg = [k for k in reg_results if 'Fair' not in k]
fig, axes = plt.subplots(1, len(orig_reg), figsize=(15, 5))
for ax, name in zip(axes, orig_reg):
    res = reg_results[name]
    ax.scatter(y_test_r, res['y_pred'], alpha=0.4, s=10)
    ax.plot([y_test_r.min(), y_test_r.max()],
            [y_test_r.min(), y_test_r.max()], 'r--')
    ax.set_xlabel('True SLEDAI')
    ax.set_ylabel('Predicted SLEDAI')
    ax.set_title(f"{name}\nRMSE={res['rmse']:.2f}, R²={res['r2']:.4f}")
plt.tight_layout()
plt.savefig(f'{OUT}/reg_predicted_vs_actual.png', dpi=150)
plt.close()

# DPD comparison plot
orig_models = [k for k in reg_results if 'Fair' not in k]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, attr, title in zip(axes, ['dpd_race', 'dpd_sex'], ['DPD by Race', 'DPD by Sex']):
    names    = []
    dpd_orig = []
    dpd_fair = []
    for name in orig_models:
        names.append(name)
        dpd_orig.append(abs(reg_results[name][attr]))
        fair_key = f'{name} (Fair)'
        dpd_fair.append(abs(reg_results[fair_key][attr]) if fair_key in reg_results else np.nan)
    x = np.arange(len(names))
    ax.bar(x - 0.2, dpd_orig, 0.4, label='Original')
    ax.bar(x + 0.2, dpd_fair, 0.4, label='Fair')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('|DPD|')
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/reg_dpd_comparison.png', dpi=150)
plt.close()

# Best regressor importances
best_reg_name = max(orig_reg, key=lambda k: reg_results[k]['r2'])
best_reg      = reg_results[best_reg_name]['model']
print(f"\nBest regressor: {best_reg_name} (R²={reg_results[best_reg_name]['r2']:.4f})")

reg_importances = get_importances(best_reg, gene_names)
if reg_importances is not None:
    top_reg_genes = reg_importances.head(20)
    plt.figure(figsize=(10, 6))
    top_reg_genes.plot(kind='barh')
    plt.title(f'Top 20 Genes — SLEDAI Regression ({best_reg_name})')
    plt.xlabel('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUT}/reg_feature_importance.png', dpi=150)
    plt.close()
    reg_importances.to_csv(f'{OUT}/reg_all_gene_importances.csv', header=['importance'])
    print(f"\nTop 20 SLEDAI regression genes:\n{top_reg_genes}")

# ─────────────────────────────────────────────
# PART 3: BIOLOGICAL VALIDATION
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PART 3: BIOLOGICAL VALIDATION")
print("="*60)

if clf_importances is not None and reg_importances is not None:
    comparison_df = pd.DataFrame({
        'rank':                range(1, 21),
        'classification_gene': top_clf_genes.index.tolist(),
        'clf_importance':      top_clf_genes.values,
        'regression_gene':     top_reg_genes.index.tolist(),
        'reg_importance':      top_reg_genes.values,
    })
    comparison_df.to_csv(f'{OUT}/top_genes_comparison.csv', index=False)
    print("\nTop 20 genes side by side:")
    print(comparison_df.to_string(index=False))

    top_clf_50 = set(clf_importances.head(50).index)
    top_reg_50 = set(reg_importances.head(50).index)
    overlap_50 = top_clf_50 & top_reg_50
    print(f"\nOverlap at top 50: {len(overlap_50)} genes — {sorted(overlap_50)}")

    if overlap_50:
        pd.DataFrame({
            'gene':           list(overlap_50),
            'clf_importance': [clf_importances[g] for g in overlap_50],
            'reg_importance': [reg_importances[g] for g in overlap_50],
        }).sort_values('clf_importance', ascending=False).to_csv(
            f'{OUT}/overlapping_genes_top50.csv', index=False
        )

print("\nSpearman correlation — top 10 regression genes vs SLEDAI:")
sle_expr = pd.DataFrame(X_sle, columns=gene_names)
spearman_results = []
for gene in top_reg_genes.index[:10]:
    r, p = stats.spearmanr(sle_expr[gene], y_sledai)
    spearman_results.append({'gene': gene, 'spearman_r': r, 'p_value': p})
    print(f"  {gene}: r={r:.4f}, p={p:.4e}")

pd.DataFrame(spearman_results).to_csv(f'{OUT}/spearman_correlations.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, gene in zip(axes.flatten(), top_reg_genes.index[:6]):
    r, p = stats.spearmanr(sle_expr[gene], y_sledai)
    ax.scatter(sle_expr[gene], y_sledai, alpha=0.3, s=8)
    ax.set_xlabel(f'{gene} expression')
    ax.set_ylabel('SLEDAI')
    ax.set_title(f'{gene}\nr={r:.3f}, p={p:.2e}')
plt.suptitle('Top Genes vs SLEDAI Score (Spearman)', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/gene_sledai_correlations.png', dpi=150)
plt.close()

# ─────────────────────────────────────────────
# EXPORT PREDICTIONS
# ─────────────────────────────────────────────
_, test_indices = train_test_split(
    range(len(y_clf)), test_size=0.2, random_state=42, stratify=y_clf
)
test_gsm_ids = [list(expr.index)[i] for i in test_indices]

predictions_df = pd.DataFrame({
    'gsm_id':          test_gsm_ids,
    'true_label':      y_test_c,
    'predicted_label': clf_results[best_clf_name]['y_pred'],
    'probability_sle': clf_results[best_clf_name]['y_prob'],
})
predictions_df = predictions_df.merge(
    meta[['race', 'sex']], left_on='gsm_id', right_index=True, how='left'
)
predictions_df.to_csv(f'{OUT}/test_predictions.csv', index=False)
print("Predictions saved.")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nClassification Results:")
for name, res in clf_results.items():
    print(f"  {name}: AUC={res['test_auc']:.4f} | DPD Race={res['dpd_race']:.4f} | DPD Sex={res['dpd_sex']:.4f}")

print("\nRegression Results:")
for name, res in reg_results.items():
    print(f"  {name}: R²={res['r2']:.4f} | RMSE={res['rmse']:.4f} | DPD Race={res['dpd_race']:.4f} | DPD Sex={res['dpd_sex']:.4f}")

summary_rows = []
for name, res in clf_results.items():
    summary_rows.append({
        'task': 'Classification', 'model': name,
        'best_params': str(res.get('best_params', '')),
        'cv_score':    res.get('best_cv_auc', ''),
        'test_metric': 'AUC', 'test_score': res['test_auc'],
        'dpd_race': res['dpd_race'], 'dpd_sex': res['dpd_sex'],
        'eod_race': res['eod_race'], 'eod_sex': res['eod_sex'],
    })
for name, res in reg_results.items():
    summary_rows.append({
        'task': 'Regression', 'model': name,
        'best_params': str(res.get('best_params', '')),
        'cv_score':    res.get('best_cv_r2', ''),
        'test_metric': 'R²', 'test_score': res['r2'],
        'dpd_race': res['dpd_race'], 'dpd_sex': res['dpd_sex'],
        'eod_race': '', 'eod_sex': '',
    })

pd.DataFrame(summary_rows).to_csv(f'{OUT}/full_pipeline_summary.csv', index=False)
print(f"\nAll outputs saved to {OUT}")