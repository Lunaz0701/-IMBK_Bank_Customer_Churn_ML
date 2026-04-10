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

  ### AutoML – Hyperparameter Tuning – Stacking Pipe – SHAP value

  1. AutoML

  F1-score를 기준으로 하여, 상위 4개의 모델을 선정하기 위해, 먼저 AutoML을 실행하였으며, 해당하는 결과는 다음과 같다.

  <img width="800" height="435" alt="image" src="https://github.com/user-attachments/assets/edcd3885-b00e-4007-ae48-3521cfaa2a3b" />

  결과, 상위 4개 모델은 Adaboost, Light GBM, GBM classifier, Catboost 이므로, 해당 4개 모델을 사용한다.

  2. Hyperparameter Tuning

  각 모델에 대한 objective function을 작성하고, 이를 이용해 optuna로 최적의 hyperparameter를 튜닝한다.
  조건은 다음과 같다.
    1. 각 parameter 별로, n_estimators는 50 - 300, learning_rate는 0.01 - 0.5, max_depth는 3 - 15의 범위들로 통일한다.
    2. 코드 실행 시 마다 수치가 달라지는 걸 방지하기 위해 random_state를 이용해 시드값 42로 고정한다.
    3. 본 코드에 작성된 verbose=-1, verbose=0 은 코드 실행 시 결과의 가독성을 위해 불필요한 구문을 생략하는 parameter이며, 분석 결과에는 영향을 주지 않는다.

  튜닝을 통해 도출된 최적의 hyperparameter는 다음과 같다.
    1. AdaBoost : {'n_estimators': 157, 'learning_rate': 0.38612642143043374}
    2. Light_GBM : {'n_estimators': 271, 'learning_rate': 0.034879786777408876, 'max_depth': 6, 'num_leaves': 90, 'min_child_samples': 17}
    3. GBM Classifier : {'n_estimators': 169, 'learning_rate': 0.40183983523656674, 'max_depth': 14}
    4. CatBoost : {'iterations': 94, 'learning_rate': 0.10917295154082785, 'max_depth': 9}

  3. Stacking Pipe

  최적화된 hyperparameter를 이용하여, 최종적으로 stacking ML Pipeline을 구성한다.

  전방은 Adaboost, LGBM, GBM을 이용하며 후방은 CatBoost로 구성하였으며, 최적의 성능을 위해 CPU 사용의 제한을 무시한다.

  결과적으로 Stack 모델의 F1-score는 약 0.5634의 수치로 도출되었다.

  4. SHAP value

  Stack 모델에 사용된 모델 중, Light-GBM 모델을 이용하여 SHAP value를 도출한다.

  선정 이유는 SHAP value는 Tree기반이며, 이는 속도도 빠르고, 대용량 처리에 적합한 모델로 Light-GBM이 1순위이기 때문이다.

  이를 통해 시각화된 SHAP value 그래프는 다음과 같다.

  <img width="554" height="404" alt="image" src="https://github.com/user-attachments/assets/b3c8a10d-1de8-4cc5-8e27-fed523424c6d" />

  이에 대한 해석은 다음과 같다.
    1. 신용카드 보유 여부는 고객 이탈율을 알기 어렵다. 보유하고 있음에도, 이탈할 가능성이 높을 수 있다. 단, 보유하지 않았다면 이탈할 가능성이 높다.
    2. 성별의 경우, 남성이 여성보다 고객 이탈율이 더 높음을 알 수 있다. 이는 성별에 따라서 어떤 것을 중요시 하는가에 대한 성향으로 인한 결과일 것이다.
    3. 추정 연봉이 낮다면 고객 이탈 확률이 증가한다. 미래에 보유할 돈이 적어진다면 그만큼 거래의 활동성이 줄을 것이다.
    4. 은행을 오랫동안 이용한다면 이탈율이 증가할 것이다. 이는 해당 은행의 서비스에 너무 익숙해져 관심도가 갈수록 낮아지면서 생기는 현상이다.
    혹은 신규 가입 고객의 상품에 더 치중한다면 기존 고객의 중요도가 상대적으로 떨어지므로, 이로 인해 기존 고객이 이탈할 가능성이 증가할 것이다.
    5. 은행에서 이용하는 상품의 수가 적다면 이탈율이 증가한다. 해당 은행에 대한 관심 또는 서비스 이용 희망이 비교적 적다는 것이고, 이로인해 이탈할 가능성이 증가할 것이다.
    6. 계좌 잔액은 고객의 이탈율을 알기 어렵다. 잔액은 고객의 충성도가 아니며, 단독으로는 설명하기 어려운 변수이다. 혹은 비선형의 관계를 가질 수도 있음을 염두한다.
    7. 신용 점수는 고객의 이탈율을 알기 어렵다. 신용 점수는 신용 리스크를 설명하기 위한 변수이다. 다만, 점수가 낮다면 이탈할 가능성이 증가한다.
    8. 나이는 고객 이탈율을 알기 어렵다. 거동이 불편한 노인일수록 고객 이탈율이 더 높을 것이라 예상하였으나, 나이만으로는 설명하지 못하는 부분이 있기 때문일 것으로 추정된다.
    9. 활성 고객 여부는 고객 이탈율에 영향을 거의 주지 못한다. 다만, 비활성 고객이 미세하지만 이탈율 증가에 기여한다는 것을 알 수 있다.

  ## 인사이트 제안

  위의 결론을 종합하여 다음과 같은 인사이트를 제안한다.

  추정 연봉, 은행 이용 기간, 상품 수, 성별이 영향을 확실히 줄 수 있으며, 이에 대한 대책이 필요하다는 결론이다.
  
  이중 은행 이용 기간에 대한 결과에 대해 인사이트를 제안한다.
  
  신규 고객 유치도 필요하지만 너무 이에만 열중한다면, 기존 고객에 대한 중요도는 떨어져, 기존 고객 이탈이라는 역효과가 발생할 것이다.
  따라서 기존 고객들도 당행을 꾸준히 이용할 수 있게끔, 기존 고객만을 위한 상품 또한 필요하다.
  
  꾸준히 이용해온 고객들에게 멤버십 혜택, 혹은 마일리지 형식의 상품을 주는 것도 충분히 도움이 될 수 있는 아이디어라 생각한다.
  이는 은행 이용 기간에 대한 인사이트이지만, 이에만 해당하지 않고, 이는 상품 수에도 영향을 줄 수 있다.
  만약 고객이 이러한 상품에 더 가입한다면 해당 고객의 이용하는 상품 수가 증가한다는 것이기도 하며,
  따라서 이탈율을 줄이기 위한 효과적인 방법이라고 생각한다.

  이를 통해, 신규 고객의 유치 뿐만이 아닌, 기존 고객의 유지에도 신경을 씀으로서, 고객 이탈 방지를 보다 효과적으로 할 수 있을 것이라 기대된다.
