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

  <style type="text/css">
#T_0cdb3_row8_col1 {
  background-color: lightgreen;
}
</style>
<table id="T_0cdb3">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0cdb3_level0_col0" class="col_heading level0 col0" >Description</th>
      <th id="T_0cdb3_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0cdb3_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_0cdb3_row0_col0" class="data row0 col0" >Session id</td>
      <td id="T_0cdb3_row0_col1" class="data row0 col1" >42</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_0cdb3_row1_col0" class="data row1 col0" >Target</td>
      <td id="T_0cdb3_row1_col1" class="data row1 col1" >churn</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_0cdb3_row2_col0" class="data row2 col0" >Target type</td>
      <td id="T_0cdb3_row2_col1" class="data row2 col1" >Binary</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_0cdb3_row3_col0" class="data row3 col0" >Original data shape</td>
      <td id="T_0cdb3_row3_col1" class="data row3 col1" >(8000, 11)</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_0cdb3_row4_col0" class="data row4 col0" >Transformed data shape</td>
      <td id="T_0cdb3_row4_col1" class="data row4 col1" >(8000, 11)</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_0cdb3_row5_col0" class="data row5 col0" >Transformed train set shape</td>
      <td id="T_0cdb3_row5_col1" class="data row5 col1" >(5600, 11)</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_0cdb3_row6_col0" class="data row6 col0" >Transformed test set shape</td>
      <td id="T_0cdb3_row6_col1" class="data row6 col1" >(2400, 11)</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_0cdb3_row7_col0" class="data row7 col0" >Numeric features</td>
      <td id="T_0cdb3_row7_col1" class="data row7 col1" >10</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_0cdb3_row8_col0" class="data row8 col0" >Preprocess</td>
      <td id="T_0cdb3_row8_col1" class="data row8 col1" >True</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_0cdb3_row9_col0" class="data row9 col0" >Imputation type</td>
      <td id="T_0cdb3_row9_col1" class="data row9 col1" >simple</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_0cdb3_row10_col0" class="data row10 col0" >Numeric imputation</td>
      <td id="T_0cdb3_row10_col1" class="data row10 col1" >mean</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_0cdb3_row11_col0" class="data row11 col0" >Categorical imputation</td>
      <td id="T_0cdb3_row11_col1" class="data row11 col1" >mode</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_0cdb3_row12_col0" class="data row12 col0" >Fold Generator</td>
      <td id="T_0cdb3_row12_col1" class="data row12 col1" >StratifiedKFold</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_0cdb3_row13_col0" class="data row13 col0" >Fold Number</td>
      <td id="T_0cdb3_row13_col1" class="data row13 col1" >10</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_0cdb3_row14_col0" class="data row14 col0" >CPU Jobs</td>
      <td id="T_0cdb3_row14_col1" class="data row14 col1" >-1</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_0cdb3_row15_col0" class="data row15 col0" >Use GPU</td>
      <td id="T_0cdb3_row15_col1" class="data row15 col1" >False</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_0cdb3_row16_col0" class="data row16 col0" >Log Experiment</td>
      <td id="T_0cdb3_row16_col1" class="data row16 col1" >False</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_0cdb3_row17_col0" class="data row17 col0" >Experiment Name</td>
      <td id="T_0cdb3_row17_col1" class="data row17 col1" >clf-default-name</td>
    </tr>
    <tr>
      <th id="T_0cdb3_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_0cdb3_row18_col0" class="data row18 col0" >USI</td>
      <td id="T_0cdb3_row18_col1" class="data row18 col1" >1747</td>
    </tr>
  </tbody>
</table>

  결과, 상위 4개 모델은 Adaboost, Light GBM, GBM classifier, Catboost 이므로, 해당 4개 모델을 사용한다.

  2. Hyperparameter Tuning

  각 모델에 대한 objective function을 작성하고, 이를 이용해 optuna로 최적의 hyperparameter를 튜닝한다.
  조건은 다음과 같다.
    1. 각 parameter 별로, n_estimators는 50~300, learning_rate는 0.01~0.5, max_depth는 3~15로 통일한다.
    2. 코드 실행 시 마다 수치가 달라지는 걸 방지하기 위해 random_state를 이용해 시드값 42로 고정한다.
    3. 본 코드에 작성된 verbose=-1, verbose=0 은 코드 실행 시 결과의 가독성을 위해 불필요한 구문을 생략하는 parameter다.

  
