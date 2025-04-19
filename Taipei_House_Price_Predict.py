import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 特徵與目標變數
features = ['行政區', '土地面積', '建物總面積', '屋齡', '樓層', '總樓層', '房數', '廳數', '衛數', '電梯', '車位類別']
target = '總價'

X = df[features]
y = df[target]

# 類別型特徵
categorical_features = ['行政區', '車位類別']
numerical_features = list(set(features) - set(categorical_features))

# 前處理器：對類別型欄位做 One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ],
    remainder='passthrough'  # 其餘數值欄位保留
)

# 拆分訓練/測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========= 線性迴歸模型 =========
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lr_model.fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, y_lr_pred))
lr_r2 = r2_score(y_test, y_lr_pred)

# ========= 隨機森林模型 =========
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, y_rf_pred))
rf_r2 = r2_score(y_test, y_rf_pred)

# ========= 顯示結果 =========
print("模型比較：")
print(f"[線性迴歸]    RMSE: {lr_rmse:.2f} 萬, R²: {lr_r2:.4f}")
print(f"[隨機森林]    RMSE: {rf_rmse:.2f} 萬, R²: {rf_r2:.4f}")
