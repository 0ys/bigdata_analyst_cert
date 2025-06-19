# ============================================
# 1. 라이브러리 및 데이터 불러오기
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# 예시용 데이터셋 (필요 시 수정)
train = pd.read_csv("https://raw.githubusercontent.com/lovedlim/bigdata_analyst_cert/main/part2/ch4/train.csv")
test = pd.read_csv("https://raw.githubusercontent.com/lovedlim/bigdata_analyst_cert/main/part2/ch4/test.csv")

# ============================================
# 2. EDA (탐색적 데이터 분석)
# ============================================
print("Train shape:", train.shape)
print("Test shape:", test.shape)

print(train.info())
print(train.describe())

# 범주형 변수 분포
print(train.describe(include='O'))

# 결측치 확인
print(train.isnull().sum())
print(test.isnull().sum())

# 목표변수 분포 시각화
train['Item_Outlet_Sales'].hist(bins=50)

# ============================================
# 3. 데이터 전처리
# ============================================
target = train.pop('Item_Outlet_Sales')

# 범주형 변수 인코딩
categorical_cols = train.select_dtypes(include='object').columns.tolist()

# train/test 합치기
df = pd.concat([train, test], axis=0)

# 레이블 인코딩
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# 결측치 처리
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].min())
df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])

# train/test 분리
train = df.iloc[:len(target)].copy()
test = df.iloc[len(target):].copy()

# 불필요한 식별자 제거
drop_cols = ['Item_Identifier']
for col in drop_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)

# ============================================
# 4. 검증 데이터 분리
# ============================================
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=42)

# ============================================
# 5. 평가 지표 함수 정의
# ============================================
def print_metrics(y_true, y_pred, model_name="Model"):
    print(f"[{model_name}]")
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("R2 :", r2_score(y_true, y_pred))
    print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
    print("-" * 40)

# ============================================
# 6. 모델 학습 및 평가
# ============================================

## ① 선형 회귀
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_val)
print_metrics(y_val, pred_lr, "Linear Regression")

## ② 랜덤 포레스트
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_val)
print_metrics(y_val, pred_rf, "Random Forest")

## ③ LightGBM
lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
pred_lgb = lgb_model.predict(X_val)
print_metrics(y_val, pred_lgb, "LightGBM")

# ============================================
# 7. 최종 예측 및 결과 저장
# ============================================
final_pred = lgb_model.predict(test)
submit = pd.DataFrame({'Predicted_Sales': final_pred})
submit.to_csv("final_result.csv", index=False)
print(submit.head())
