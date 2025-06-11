## 🌱 목차

- Intro. 시험 응시 전략, 시험 환경 소개, 코드 및 데이터 불러오기, 자주하는 질문 등
- PART1. 작업형1 (파이썬, 판다스, 연습문제)
- PART2. 작업형2 (이진분류, 다중분류, 회귀, 평가지표, 연습문제)
- PART3. 작업형3 (가설검정, 분산 분석, 카이제곱, 회귀, 로지스틱 회귀, 연습문제)
- PART4. 기출유형 (예시문제, 2회 ~ 8회까지)

## 시험 테스트

[시험 테스트](https://www.kaggle.com/code/agileteam/t3-ex-ver2025-py)

## 시험 환경

- 시험시간: 180분
- 시험환경: 클라우드 기반 CBT
- 언어: R, Python
- 제약사항
  - 코드 라인별 실행 불가
  - 그래프 기능, 단축키, 자동완성 미제공
  - 코드 실행 시간은 1분으로 제한, 시간 초과 시 강제 취소
  - 제공된 패키지만 사용가능
- 기타
  - 문제별 과락 및 시간제한 없음
  - 원하는 문제로 언제든 이동 가능

## 합격 기준

100점 만점에 60점 이상 (과목별 과락 없음)

## 시험 주요 내용

1. 작업형1(30점): 주어진 데이터를 정제, 변환, 연상하고 요구하는 조건에 맞는 값 제출
   1. 3문항(문항당 10점)
   2. 각 문항별로 정답 여부에 따라 배점 기준이 만점 또는 0점
   3. 작성 코드에 대한 부분 점수 없음, 제출한 답안만 채점함
   4. 지시된 제출 형식(소수점) 준수
2. 작업형2(40점): 머신러닝 모델을 만들고 예측 결과 제출(데이터 전처리, 평가 등 수행)
   1. 1문항
   2. 평가용 데이터를 이용한 예측 결과를 CSV 파일로 제출하며, 지시된 제출 형식 준수
   3. 평가지표에 따라 구간별 점수 부여
   4. 작성 코드에 대한 부분 점수 없음(생성한 csv 파일로만 평가)
   5. 평가지표에 따른 구간 점수를 획득해도 제출 형식을 위반하면 득점 점수에서 감점하며, 감점 유형이 중복되면 누적해 감점함
3. 작업형3(30점): 가설검정, 회귀 분석, 로지스틱 회귀, 카이제곱검정 등 통계 결괏값 제출
   1. 2문항(문항당 15점)
   2. 각 문항별 소문항의 순서대로 답안을 제출하며, 지시된 제출 형식 준수
   3. 각 문항의 소문항별로 정답 여부에 따라 배점
   4. 작성 코드에 대한 부분 점수 없음

## 레포지토리 구조

```text
.
├── README.md
├── part1 (작업형1)
│   ├── ch1
│   │   └── ch1_python.ipynb (코드)
│   ├── ch2
│   │   └── ch2_pandas.ipynb (코드)
│   └── ch3
│       ├── ch3_ex_type1.ipynb (코드)
│       ├── delivery_time.csv
│       ├── school_data.csv
│       ├── school_data_science.csv
│       ├── school_data_social.csv
│       ├── type1_data1.csv
│       └── type1_data2.csv
├── part2 (작업형2)
│   ├── ch2
│   │   ├── ch2_classification.ipynb (코드)
│   │   ├── test.csv
│   │   └── train.csv
│   ├── ch3
│   │   └── ch3_metrics.ipynb (코드)
│   ├── ch4
│   │   ├── ch4_regression.ipynb (코드)
│   │   ├── test.csv
│   │   └── train.csv
│   ├── ch5
│   │   ├── ch5_multi_class_classification.ipynb (코드)
│   │   ├── test.csv
│   │   └── train.csv
│   ├── ch6
│   │   ├── ch6_ex_classification.ipynb (코드)
│   │   ├── creditcard_test.csv
│   │   ├── creditcard_train.csv
│   │   ├── diabetes_test.csv
│   │   ├── diabetes_train.csv
│   │   ├── hr_test.csv
│   │   └── hr_train.csv
│   ├── ch7
│   │   ├── ch7_ex_multi_class_classification.ipynb (코드)
│   │   ├── drug_test.csv
│   │   ├── drug_train.csv
│   │   ├── glass_test.csv
│   │   ├── glass_train.csv
│   │   ├── score_test.csv
│   │   └── score_train.csv
│   └── ch8
│       ├── car_test.csv
│       ├── car_train.csv
│       ├── ch8_ex_regression.ipynb (코드)
│       ├── flight_test.csv
│       ├── flight_train.csv
│       ├── laptop_test.csv
│       └── laptop_train.csv
├── part3 (작업형3)
│   ├── ch1
│   │   └── ch1_hypothesis_testing.ipynb (코드)
│   ├── ch2
│   │   ├── ch2_anova.ipynb (코드)
│   │   ├── fertilizer.csv
│   │   └── tree.csv
│   ├── ch3
│   │   └── ch3_chi_square.ipynb (코드)
│   ├── ch4
│   │   ├── ch4_linear_regression.ipynb (코드)
│   │   └── study.csv
│   ├── ch5
│   │   ├── ch5_logistic_regression.ipynb (코드)
│   │   └── health_survey.csv
│   └── ch6
│       ├── ch6_ex_type3.ipynb (코드)
│       ├── math.csv
│       └── tomato2.csv
└── part4 (기출유형)
    ├── ch2
    │   ├── X_test.csv
    │   ├── X_train.csv
    │   ├── members.csv
    │   ├── p2_type1.ipynb (작업형1 코드)
    │   ├── p2_type2.ipynb (작업형2 코드)
    │   └── y_train.csv
    ├── ch3
    │   ├── members.csv
    │   ├── p3_type1.ipynb (작업형1 코드)
    │   ├── p3_type2.ipynb (작업형2 코드)
    │   ├── test.csv
    │   ├── train.csv
    │   └── year.csv
    ├── ch4
    │   ├── data4-1.csv
    │   ├── data4-2.csv
    │   ├── data4-3.csv
    │   ├── p4_type1.ipynb (작업형1 코드)
    │   ├── p4_type2.ipynb (작업형2 코드)
    │   ├── test.csv
    │   └── train.csv
    ├── ch5
    │   ├── data5-1.csv
    │   ├── data5-2.csv
    │   ├── data5-3.csv
    │   ├── p5_type1.ipynb (작업형1 코드)
    │   ├── p5_type2.ipynb (작업형2 코드)
    │   ├── test.csv
    │   └── train.csv
    ├── ch6
    │   ├── data6-1-1.csv
    │   ├── data6-1-2.csv
    │   ├── data6-1-3.csv
    │   ├── data6-3-2.csv
    │   ├── energy_test.csv
    │   ├── energy_train.csv
    │   ├── p6_type1.ipynb (작업형1 코드)
    │   ├── p6_type2.ipynb (작업형2 코드)
    │   └── p6_type3.ipynb (작업형3 코드)
    ├── ch7
    │   ├── air_quality.csv
    │   ├── clam.csv
    │   ├── mart_test.csv
    │   ├── mart_train.csv
    │   ├── p7_type1.ipynb (작업형1 코드)
    │   ├── p7_type2.ipynb (작업형2 코드)
    │   ├── p7_type3.ipynb (작업형3 코드)
    │   ├── stock_market.csv
    │   ├── student_assessment.csv
    │   └── system_cpu.csv
    └── ch8
        ├── chem.csv
        ├── churn.csv
        ├── churn_test.csv
        ├── churn_train.csv
        ├── customer_travel.csv
        ├── drinks.csv
        ├── p8_type1.ipynb (작업형1 코드)
        ├── p8_type2.ipynb (작업형2 코드)
        ├── p8_type3.ipynb (작업형3 코드)
        ├── piq.csv
        └── tourist.csv
```

이 레포지토리에 실린 모든 내용의 저작권은 저자에게 있으며, 저자의 허락 없이 이 코드의 일부 또는 전부를 복제, 배포할 수 없습니다.
