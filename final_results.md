

Saving all outputs to: /gpfs/home/lv2255/scratch/sle_outputs_fin
============================================================
LOADING DATA
============================================================
Expression matrix: (1816, 5000)
Metadata: (1816, 7)
Label distribution:
label
1    1756
0      60
Name: count, dtype: int64

Race distribution (0=white, 1=non-white):
race
0    1215
1     601
Name: count, dtype: int64
Sex distribution (1=female, 0=male):
sex
1    1656
0     160
Name: count, dtype: int64

============================================================
PART 1: CLASSIFICATION — Healthy vs SLE
============================================================
Train: 1452 | Test: 364
Train dist: {1: 1404, 0: 48}
Test dist:  {1: 352, 0: 12}
After SMOTE: {1: 1404, 0: 1404}

========================================
--- Random Forest ---
Fitting 5 folds for each of 108 candidates, totalling 540 fits
Best params: {'max_depth': 5, 'max_features': 'sqrt', 'max_samples': 0.8, 'min_samples_split': 5, 'n_estimators': 200}
Best CV AUC: 1.0000

[Random Forest] Test AUC: 0.9316
[Random Forest] DPD Race: 0.0180 | DPD Sex: 0.0073
[Random Forest] EOD Race: 0.5000 | EOD Sex: 0.5143
[Random Forest] Selection rate by race:
race
0    0.953975
1    0.936000
Name: selection_rate, dtype: float64
[Random Forest] Selection rate by sex:
sex
0    0.941176
1    0.948485
Name: selection_rate, dtype: float64
              precision    recall  f1-score   support

     Healthy       0.32      0.50      0.39        12
         SLE       0.98      0.96      0.97       352

    accuracy                           0.95       364
   macro avg       0.65      0.73      0.68       364
weighted avg       0.96      0.95      0.95       364


Training fair version of Random Forest...

[Random Forest Fair] Test AUC: 0.7590
[Random Forest Fair] DPD Race: 0.0159 | DPD Sex: 0.0260
[Random Forest Fair] EOD Race: 0.4167 | EOD Sex: 0.6571
[Random Forest Fair] Selection rate by race:
race
0    0.912134
1    0.928000
Name: selection_rate, dtype: float64
[Random Forest Fair] Selection rate by sex:
sex
0    0.941176
1    0.915152
Name: selection_rate, dtype: float64
              precision    recall  f1-score   support

     Healthy       0.23      0.58      0.33        12
         SLE       0.99      0.93      0.96       352

    accuracy                           0.92       364
   macro avg       0.61      0.76      0.65       364
weighted avg       0.96      0.92      0.94       364


========================================
--- XGBoost ---
Fitting 5 folds for each of 72 candidates, totalling 360 fits
Best params: {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 200, 'scale_pos_weight': np.float64(1.0), 'subsample': 0.7}
Best CV AUC: 1.0000

[XGBoost] Test AUC: 0.9920
[XGBoost] DPD Race: 0.0640 | DPD Sex: 0.0406
[XGBoost] EOD Race: 0.3333 | EOD Sex: 0.4571
[XGBoost] Selection rate by race:
race
0    1.000
1    0.936
Name: selection_rate, dtype: float64
[XGBoost] Selection rate by sex:
sex
0    0.941176
1    0.981818
Name: selection_rate, dtype: float64
              precision    recall  f1-score   support

     Healthy       1.00      0.67      0.80        12
         SLE       0.99      1.00      0.99       352

    accuracy                           0.99       364
   macro avg       0.99      0.83      0.90       364
weighted avg       0.99      0.99      0.99       364


Training fair version of XGBoost...

[XGBoost Fair] Test AUC: 0.7500
[XGBoost Fair] DPD Race: 0.0480 | DPD Sex: 0.0467
[XGBoost Fair] EOD Race: 0.5000 | EOD Sex: 0.1714
[XGBoost Fair] Selection rate by race:
race
0    1.000
1    0.952
Name: selection_rate, dtype: float64
[XGBoost Fair] Selection rate by sex:
sex
0    0.941176
1    0.987879
Name: selection_rate, dtype: float64
              precision    recall  f1-score   support

     Healthy       1.00      0.50      0.67        12
         SLE       0.98      1.00      0.99       352

    accuracy                           0.98       364
   macro avg       0.99      0.75      0.83       364
weighted avg       0.98      0.98      0.98       364


========================================
--- Logistic Regression ---
Fitting 5 folds for each of 10 candidates, totalling 50 fits
Best params: {'C': 0.001, 'penalty': 'l2', 'solver': 'saga'}
Best CV AUC: 1.0000

[Logistic Regression] Test AUC: 0.9946
[Logistic Regression] DPD Race: 0.0636 | DPD Sex: 0.0316
[Logistic Regression] EOD Race: 0.2500 | EOD Sex: 0.6000
[Logistic Regression] Selection rate by race:
race
0    0.991632
1    0.928000
Name: selection_rate, dtype: float64
[Logistic Regression] Selection rate by sex:
sex
0    0.941176
1    0.972727
Name: selection_rate, dtype: float64
              precision    recall  f1-score   support

     Healthy       0.82      0.75      0.78        12
         SLE       0.99      0.99      0.99       352

    accuracy                           0.99       364
   macro avg       0.90      0.87      0.89       364
weighted avg       0.99      0.99      0.99       364


Training fair version of Logistic Regression...

[Logistic Regression Fair] Test AUC: 0.8537
[Logistic Regression Fair] DPD Race: 0.0092 | DPD Sex: 0.0246
[Logistic Regression Fair] EOD Race: 0.2500 | EOD Sex: 0.6000
[Logistic Regression Fair] Selection rate by race:
race
0    0.937238
1    0.928000
Name: selection_rate, dtype: float64
[Logistic Regression Fair] Selection rate by sex:
sex
0    0.911765
1    0.936364
Name: selection_rate, dtype: float64
              precision    recall  f1-score   support

     Healthy       0.38      0.75      0.50        12
         SLE       0.99      0.96      0.97       352

    accuracy                           0.95       364
   macro avg       0.68      0.85      0.74       364
weighted avg       0.97      0.95      0.96       364


Best classifier: Logistic Regression (AUC=0.9946)

Top 20 classification genes:
SNORA23               0.043110
SNORA11               0.037678
RNU105A               0.035195
RNU5A-1               0.034540
SNORA31               0.034515
SNORD91B              0.031341
SNORA38B              0.029959
OTTHUMG00000037783    0.028081
SNORD42A              0.026915
SNORD9                0.026260
SNORA57               0.025361
SNORD61               0.024799
SLC4A10               0.024590
SNORA14B              0.024451
RNY4                  0.023625
SNORD51               0.023046
SNORD59A              0.021659
RPS19                 0.021596
EIF1AY                0.021358
SNORD78               0.021051
dtype: float64

============================================================
PART 2: SLEDAI REGRESSION — SLE patients only
============================================================
SLE patients with SLEDAI: 1756
SLEDAI range: 2 - 40
SLEDAI mean ± std: 10.37 ± 3.76

========================================
--- Random Forest ---
Fitting 5 folds for each of 108 candidates, totalling 540 fits
Best params: {'max_depth': 10, 'max_features': 'sqrt', 'max_samples': 0.7, 'min_samples_split': 5, 'n_estimators': 200}
Best CV R²: 0.0726

[Random Forest] Test RMSE: 3.4371 | R²: 0.1125
[Random Forest] DPD Race: 0.1096 | DPD Sex: 0.0931
[Random Forest] High SLEDAI rate by race:
race
0    0.598326
1    0.707965
Name: selection_rate, dtype: float64
[Random Forest] High SLEDAI rate by sex:
sex
0    0.720000
1    0.626911
Name: selection_rate, dtype: float64

========================================
--- XGBoost ---
Fitting 5 folds for each of 72 candidates, totalling 360 fits
Best params: {'colsample_bytree': 0.7, 'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.7}
Best CV R²: 0.0791

[XGBoost] Test RMSE: 3.4212 | R²: 0.1207
[XGBoost] DPD Race: 0.0834 | DPD Sex: 0.0495
[XGBoost] High SLEDAI rate by race:
race
0    0.527197
1    0.610619
Name: selection_rate, dtype: float64
[XGBoost] High SLEDAI rate by sex:
sex
0    0.600000
1    0.550459
Name: selection_rate, dtype: float64

========================================
--- Ridge Regression ---
Fitting 5 folds for each of 5 candidates, totalling 25 fits
Best params: {'alpha': 100.0}
Best CV R²: -0.2613

[Ridge Regression] Test RMSE: 4.3944 | R²: -0.4506
[Ridge Regression] DPD Race: 0.0940 | DPD Sex: 0.0093
[Ridge Regression] High SLEDAI rate by race:
race
0    0.481172
1    0.575221
Name: selection_rate, dtype: float64
[Ridge Regression] High SLEDAI rate by sex:
sex
0    0.520000
1    0.510703
Name: selection_rate, dtype: float64

Best regressor: XGBoost (R²=0.1207)

Top 20 SLEDAI regression genes:
DTL                   0.002701
LGALS3BP              0.002198
ZNF600                0.002167
VPREB3                0.002043
KIF3A                 0.002032
RUNX1-IT1             0.002014
OTTHUMG00000037911    0.001977
BUB1                  0.001959
HIST1H4D              0.001920
OTTHUMG00000035719    0.001831
MYO1D                 0.001805
AKR1E2                0.001794
OTTHUMG00000162173    0.001715
RNA5SP497             0.001684
GPR146                0.001678
OTTHUMG00000162919    0.001669
GNG11                 0.001663
LINC00341             0.001659
OTTHUMG00000003643    0.001656
ARHGAP5-AS1           0.001650
dtype: float32

============================================================
PART 3: BIOLOGICAL VALIDATION
============================================================

Top 20 genes side by side:
 rank classification_gene  clf_importance    regression_gene  reg_importance
    1             SNORA23        0.043110                DTL        0.002701
    2             SNORA11        0.037678           LGALS3BP        0.002198
    3             RNU105A        0.035195             ZNF600        0.002167
    4             RNU5A-1        0.034540             VPREB3        0.002043
    5             SNORA31        0.034515              KIF3A        0.002032
    6            SNORD91B        0.031341          RUNX1-IT1        0.002014
    7            SNORA38B        0.029959 OTTHUMG00000037911        0.001977
    8  OTTHUMG00000037783        0.028081               BUB1        0.001959
    9            SNORD42A        0.026915           HIST1H4D        0.001920
   10              SNORD9        0.026260 OTTHUMG00000035719        0.001831
   11             SNORA57        0.025361              MYO1D        0.001805
   12             SNORD61        0.024799             AKR1E2        0.001794
   13             SLC4A10        0.024590 OTTHUMG00000162173        0.001715
   14            SNORA14B        0.024451          RNA5SP497        0.001684
   15                RNY4        0.023625             GPR146        0.001678
   16             SNORD51        0.023046 OTTHUMG00000162919        0.001669
   17            SNORD59A        0.021659              GNG11        0.001663
   18               RPS19        0.021596          LINC00341        0.001659
   19              EIF1AY        0.021358 OTTHUMG00000003643        0.001656
   20             SNORD78        0.021051        ARHGAP5-AS1        0.001650

Overlap at top 50: 1 genes — ['RPS4Y1']

Spearman correlation — top 10 regression genes vs SLEDAI:
  DTL: r=0.2577, p=4.8987e-28
  LGALS3BP: r=0.2416, p=9.6988e-25
  ZNF600: r=-0.1031, p=1.4902e-05
  VPREB3: r=-0.0203, p=3.9440e-01
  KIF3A: r=-0.0443, p=6.3520e-02
  RUNX1-IT1: r=-0.0877, p=2.3392e-04
  OTTHUMG00000037911: r=0.0072, p=7.6455e-01
  BUB1: r=0.2693, p=1.5231e-30
  HIST1H4D: r=0.2346, p=2.2317e-23
  OTTHUMG00000035719: r=0.0040, p=8.6639e-01
Predictions saved.

============================================================
SUMMARY
============================================================

Classification Results:
  Random Forest: AUC=0.9316 | DPD Race=0.0180 | DPD Sex=0.0073
  Random Forest (Fair): AUC=0.7590 | DPD Race=0.0159 | DPD Sex=0.0260
  XGBoost: AUC=0.9920 | DPD Race=0.0640 | DPD Sex=0.0406
  XGBoost (Fair): AUC=0.7500 | DPD Race=0.0480 | DPD Sex=0.0467
  Logistic Regression: AUC=0.9946 | DPD Race=0.0636 | DPD Sex=0.0316
  Logistic Regression (Fair): AUC=0.8537 | DPD Race=0.0092 | DPD Sex=0.0246

Regression Results:
  Random Forest: R²=0.1125 | RMSE=3.4371 | DPD Race=0.1096 | DPD Sex=0.0931
  XGBoost: R²=0.1207 | RMSE=3.4212 | DPD Race=0.0834 | DPD Sex=0.0495
  Ridge Regression: R²=-0.4506 | RMSE=4.3944 | DPD Race=0.0940 | DPD Sex=0.0093

All outputs saved to /gpfs/home/lv2255/scratch/sle_outputs_fin
