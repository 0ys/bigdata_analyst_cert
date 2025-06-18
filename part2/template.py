########################################
# 머신러닝 전체 프로세스 템플릿 코드
########################################

# 0. 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

########################################
# 1. 문제 정의 (이 단계는 주석으로 정리)
# 예: 이진 분류 문제, 목표: 고객 이탈 예측
########################################

########################################
# 2. 데이터 불러오기
########################################

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

########################################
# 3. 탐색적 데이터 분석 (EDA)
########################################

# 데이터 기본 정보
print(train.shape)
print(train.dtypes)
print(train.head())

# 결측치 확인
print(train.isnull().sum())

# 기초 통계량
print(train.describe())

# 타깃 변수 분포
print(train['target'].value_counts())

# 시각화 예시 (필요 시)
sns.histplot(train['feature1'])
plt.show()

########################################
# 4. 데이터 전처리
########################################

# 결측치 처리
train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)

# 범주형 인코딩
cat_cols = train.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 스케일링 (선택적)
num_cols = train.select_dtypes(include=np.number).drop('target', axis=1).columns

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

########################################
# 5. 검증용 데이터 분리
########################################

X = train.drop('target', axis=1)
y = train['target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

########################################
# 6. 모델 학습
########################################

# 예시: 분류 문제(Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# (회귀 문제라면 RandomForestRegressor 사용)
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)

########################################
# 7. 모델 평가
########################################

y_pred = model.predict(X_valid)

# 분류 평가
print("Accuracy:", accuracy_score(y_valid, y_pred))
print("F1 Score:", f1_score(y_valid, y_pred, average='macro'))

# 회귀 평가 예시
# mse = mean_squared_error(y_valid, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_valid, y_pred)
# print("RMSE:", rmse)
# print("R2:", r2)

########################################
# 8. 최종 예측 및 제출 파일 생성
########################################

test_pred = model.predict(test.drop('id', axis=1))
submission = pd.DataFrame({'id': test['id'], 'target': test_pred})
submission.to_csv('submission.csv', index=False)

########################################
# 9. (옵션) 하이퍼파라미터 튜닝 예시
########################################

# params = {'n_estimators': [100, 200], 'max_depth': [5, 10, 20]}
# grid = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=3)
# grid.fit(X_train, y_train)
# print("Best Params:", grid.best_params_)
