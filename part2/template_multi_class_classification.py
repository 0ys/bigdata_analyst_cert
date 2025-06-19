# ============================================
# 1. 라이브러리 불러오기
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# ============================================
# 2. 데이터 불러오기
# ============================================
train = pd.read_csv("https://raw.githubusercontent.com/lovedlim/bigdata_analyst_cert/main/part2/ch5/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/lovedlim/bigdata_analyst_cert/main/part2/ch5/test.csv")

# ============================================
# 3. EDA
# ============================================
print("Train shape:", train.shape)
print(train.info())
print(train.isnull().sum())
print(train['Credit_Score'].value_counts())

# ============================================
# 4. 전처리 준비
# ============================================

target = train.pop('Credit_Score')
categorical_cols = train.select_dtypes(include='object').columns.tolist()

# ============================================
# 4-1. One-Hot Encoding 준비 (RandomForest용)
# ============================================
train_oh = train.copy()
test_oh = test.copy()

# 결측치 처리 (이 예제는 결측치 거의 없음)
for col in categorical_cols:
    train_oh[col] = train_oh[col].fillna('Unknown')
    test_oh[col] = test_oh[col].fillna('Unknown')

# One-Hot Encoding 적용
train_oh = pd.get_dummies(train_oh, columns=categorical_cols)
test_oh = pd.get_dummies(test_oh, columns=categorical_cols)

# train/test 동일하게 컬럼 맞추기
train_oh, test_oh = train_oh.align(test_oh, join='left', axis=1, fill_value=0)

# ============================================
# 4-2. Category Encoding 준비 (LightGBM용)
# ============================================
train_cat = train.copy()
test_cat = test.copy()

# category로 타입 변환 (LightGBM 자동 인식)
for col in categorical_cols:
    train_cat[col] = train_cat[col].astype('category')
    test_cat[col] = test_cat[col].astype('category')

# ============================================
# 5. 검증 데이터 분리
# ============================================
# RandomForest용 (OneHot)
X_train_oh, X_val_oh, y_train, y_val = train_test_split(train_oh, target, test_size=0.2, random_state=42)

# LightGBM용 (category)
X_train_cat, X_val_cat, _, _ = train_test_split(train_cat, target, test_size=0.2, random_state=42)

# ============================================
# 6. 평가 함수 정의
# ============================================
def print_metrics(y_true, y_pred, model_name="Model"):
    print(f"[{model_name}]")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("F1_macro :", f1_score(y_true, y_pred, average='macro'))
    print("-" * 40)

# ============================================
# 7-1. RandomForestClassifier (OneHot 사용)
# ============================================
print("▶▶ RandomForest (OneHot Encoding)")
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_oh, y_train)
pred_rf = rf.predict(X_val_oh)
print_metrics(y_val, pred_rf, "RandomForest")

# ============================================
# 7-2. LightGBMClassifier (Category 사용)
# ============================================
print("▶▶ LightGBM (Category Encoding)")
lgbmc = lgb.LGBMClassifier(random_state=42, verbose=-1)
lgbmc.fit(X_train_cat, y_train)
pred_lgb = lgbmc.predict(X_val_cat)
print_metrics(y_val, pred_lgb, "LightGBM")

# ============================================
# 8. 최종 예측 및 결과 저장
# ============================================
# LightGBM 최종 예측
final_pred = lgbmc.predict(test_cat)
submit = pd.DataFrame({'Credit_Score_Prediction': final_pred})
submit.to_csv("final_result.csv", index=False)
print(submit.head())
