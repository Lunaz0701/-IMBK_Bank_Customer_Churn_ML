#  고객 이탈 분류 ML 및 인사이트분석
시기 : 2026.04.09-2026.04.10
---

## technique stacks :
  
  1. Data Preprocessing
    - pandas
    - numpy
    - sklearn.preprocessing (LabelEncoder, StandardScaler)
    - sklearn.compose (ColumnTransformer)
    
  2. Data Split
    - sklearn.model_selection (train_test_split, cross_val_score)
    
  3. EDA & Visualization
    - matplotlib.pyplot
    - seaborn
    
  4. AutoML (Model Selection)
    - pycaret.classification (setup, compare_models)
    
  5. ML Modeling
    - sklearn.ensemble (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier)
    - sklearn.linear_model (LogisticRegression)
    - sklearn.svm (SVC)
    - sklearn.neighbors (KNeighborsClassifier)
    - lightgbm (LGBMClassifier)
    - catboost (CatBoostClassifier)
    
  6. Hyperparameter Tuning
    - optuna
    
  7. Model Evaluation
    - sklearn.metrics (f1_score)
    
  8. Model Explainability
    - shap (TreeExplainer, summary_plot)
    
  9. Ensemble (Stacking)
    - sklearn.ensemble (StackingClassifier)

## 데이터 출처

  해당 분석에 사용된 데이터의 출처는 다음과 같음.
  - 캐글 Bank Customer Churn Dataset (row: 10000, col:12) [링크 : https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data]

## 데이터 전처리 및 변수 선택

  분석 시, 결측치는 없어 대체할 필요가 없이 그대로 진행한다.
  
  이상치에 대해서는, 나이의 경우는 노인의 경우, 스마트뱅킹보다는 창구거래를 일반적으로 많이 사용하며
  노화에 따라 건강이 악화되어 몸을 가누기 힘들어지면서 창구에 가는 경우가 적어지면서 이탈이 될 수도 있다는 가능성이 있어 이상치 대체는 하지 않는다.
  
  신용 점수의 경우, 낮다면 그 원인으로 인해서 고객 이탈 가능성이 높아질 수도 있으므로, 이상치 대체는 하지 않는다.

  변수 중 고객 이탈에 연관이 없어보이는 변수는 제외한다. (county, customer_id)
   - customer_id는 고객 고유 ID이므로 분석에는 연관이 없다.
   - country의 경우, 나라에 따른 이탈률은 상식적으로 연관이 없다.

  스케일링에 대해서는, 변수 별로 단위가 다르기 때문에 범주형 자료를 제외한 일반 수치형 컬럼은 스케일링을 시행한다.
  이 때, 이상치들은 정당한 이상치들이므로, 표준화 스케일링을 시도해도 괜찮을 것이라 판단하였다.

## EDA 및 해석

  1. 수치형 컬럼 분석

  <img width="800" height="354" alt="image" src="https://github.com/user-attachments/assets/bdf09545-fe3d-4df8-8d0a-982511bb6e54" />

  credit_score는 낮은 값의 이상치, age는 높은 값의 이상치가 보이며, 그 외의 컬럼에는 이상치가 없음을 확인하였으며, 해당 결과를 통해 위에서 언급했듯이 전처리를 진행한다.

  <img width="554" height="404" alt="image" src="https://github.com/user-attachments/assets/dc6577b3-7833-45df-a402-78abc96c9a48" />

  컬럼 간 상관관계가 없는 것으로 보아, 나이에 따른, 혹은 신용 점수에 따른 등의 연관을 짓는 의미가 없다. 즉, 서로 독립적인 지표를 보인다.
  
  이는 어느 하나의 수치가 변동되어도, 다른 지표 상에는 그것이 바뀌어도, 안바뀌어도 상관이 없다는 결론이다.

  2. 범주형 컬럼 분석

  <img width="800" height="354" alt="image" src="https://github.com/user-attachments/assets/ed3c1b40-b5d8-41d9-bd60-100c684b7ee9" />

  금융상품이 적은 고객이 훨씬 많고, 신용 카드 보유 고객이 약 2배 넘게 더 많으며, 활동적인 회원이 유사하지만 조금 더 많다.
  
  은행의 상품 수에 관해서는 수가 1개, 2개인 경우가 대부분이므로, 이에 관해서 결과 대한 편향이 있을 수 있음을 주의한다.

## 데이터 분석 진행

  분석에 대한 흐름은 다음과 같다.

  ### AutoML – Hyperparameter Tuning – Stacking Pipe – Shap value

  1. AutoML

  F1-score를 기준으로 하여, 상위 4개의 모델을 선정하기 위해, 먼저 AutoML을 실행하였으며, 해당하는 결과는 다음과 같다.

  <img width="977" height="565" alt="image" src="https://github.com/user-attachments/assets/edcd3885-b00e-4007-ae48-3521cfaa2a3b" />

  결과, 상위 4개 모델은 Adaboost, Light GBM, GBM classifier, Catboost 이므로, 해당 4개 모델을 사용한다.

  2. Hyperparameter Tuning

  각 모델에 대한 objective function을 작성하고, 이를 이용해 optuna로 최적의 hyperparameter를 튜닝한다.
  조건은 다음과 같다.
    1. 각 parameter 별로, n_estimators는 50~300, learning_rate는 0.01~0.5, max_depth는 3~15로 통일한다.
    2. 코드 실행 시 마다 수치가 달라지는 걸 방지하기 위해 random_state를 이용해 시드값 42로 고정한다.
    3. 본 코드에 작성된 verbose=-1, verbose=0 은 코드 실행 시 결과의 가독성을 위해 불필요한 구문을 생략하는 parameter다.

  
