import pandas as pd
import numpy as np

# =====================
# 1. 讀取資料
# =====================

tourism = pd.read_csv("tourism(1).csv", encoding="utf-8-sig")
atype = pd.read_csv("attraction-type.csv", encoding="utf-8-sig")
weather = pd.read_excel("weather_clean_format.xlsx")

# =====================
# 2. 統一日期格式
# =====================

tourism["date"] = tourism["date"].astype(str).str.replace("_", "-", regex=False)
atype["date"] = atype["date"].astype(str).str.replace("_", "-", regex=False)
weather["date"] = weather["date"].astype(str)

# =====================
# 3. 統一城市名稱
# =====================

city_map = {
    "臺北市": "臺北",
    "台北市": "臺北",
    "新北市": "新北",
    "基隆市": "基隆",
    "桃園市": "新屋",
    "新竹市": "新竹",
    "新竹縣": "新竹",

    "苗栗縣": "後龍",
    "臺中市": "臺中",
    "台中市": "臺中",
    "彰化縣": "田中",
    "南投縣": "日月潭",
    "雲林縣": "古坑",

    "嘉義市": "嘉義",
    "嘉義縣": "嘉義",
    "臺南市": "臺南",
    "台南市": "臺南",
    "高雄市": "高雄",
    "屏東縣": "恆春"
}

tourism["city"] = tourism["city"].replace(city_map)
atype["city"] = atype["city"].replace(city_map)

# =====================
# 4. 合併資料
# =====================

df = tourism.merge(
    weather,
    on=["date", "city"],
    how="left"
)

df = df.merge(
    atype[["date", "attraction", "city", "type"]],
    on=["date", "attraction", "city"],
    how="left"
)

# =====================
# 5. 建立地區分類
# =====================

north = ["臺北", "新北", "基隆", "新屋", "新竹"]
central = ["後龍", "臺中", "田中", "日月潭", "古坑"]
south = ["嘉義", "臺南", "高雄", "恆春"]

def classify_region(city):
    if city in north:
        return "north"
    elif city in central:
        return "central"
    elif city in south:
        return "south"
    else:
        return "other"

df["region"] = df["city"].apply(classify_region)

# =====================
# 6. 清理資料
# =====================

df["tourists"] = pd.to_numeric(df["tourists"], errors="coerce")

for col in ["rainfall", "temperature", "rainy_days", "sunshine"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[
    "tourists",
    "rainfall",
    "temperature",
    "rainy_days",
    "sunshine",
    "type"
])

# =====================
# 7. 建立分析欄位
# =====================

df["ln_tourists"] = np.log1p(df["tourists"])

df = df.sort_values(["attraction", "date"])

df["rain_lag1"] = df.groupby("attraction")["rainfall"].shift(1)
df["rain_lag2"] = df.groupby("attraction")["rainfall"].shift(2)

threshold = df["rainfall"].quantile(0.75)
df["heavy_rain"] = (df["rainfall"] > threshold).astype(int)

df = df.dropna(subset=["rain_lag1", "rain_lag2"])

# =====================
# 8. 輸出結果
# =====================

df.to_excel("merged_data.xlsx", index=False)
df.to_csv("merged_data.csv", index=False, encoding="utf-8-sig")

print("合併完成！")
print("資料筆數：", len(df))
print("輸出檔案：merged_data.xlsx、merged_data.csv")
print(df.head())



import matplotlib.pyplot as plt
import os

# =====================
# 9. 描述性圖表
# =====================

os.makedirs("charts", exist_ok=True)

# 把 date 轉成真正日期格式
df["date_dt"] = pd.to_datetime(df["date"] + "-01")

# 1. 觀光人次長期趨勢圖
trend = (
    df.groupby("date_dt", as_index=False)["tourists"]
    .sum()
    .sort_values("date_dt")
)

plt.figure(figsize=(12, 6))
plt.plot(trend["date_dt"], trend["tourists"], marker="o", markersize=2)
plt.title("Total Tourists Over Time")
plt.xlabel("Date")
plt.ylabel("Tourists")
plt.xticks(trend["date_dt"][::6], rotation=45)
plt.tight_layout()
plt.savefig("charts/tourists_trend.png", dpi=300)
plt.show()

# 2. 北中南地區比較圖
region_trend = (
    df.groupby(["date_dt", "region"], as_index=False)["tourists"]
    .sum()
    .sort_values(["region", "date_dt"])
)

plt.figure(figsize=(12, 6))

for region in sorted(region_trend["region"].unique()):
    temp = region_trend[region_trend["region"] == region].sort_values("date_dt")
    plt.plot(
        temp["date_dt"],
        temp["tourists"],
        marker="o",
        markersize=2,
        label=region
    )

plt.title("Tourists by Region")
plt.xlabel("Date")
plt.ylabel("Tourists")
plt.legend()
dates = sorted(region_trend["date_dt"].unique())
plt.xticks(dates[::6], rotation=45)
plt.tight_layout()
plt.savefig("charts/tourists_by_region.png", dpi=300)
plt.show()

# 3. 室內 / 戶外景點在大雨時的人次比較
type_rain = (
    df.groupby(["type", "heavy_rain"], as_index=False)["tourists"]
    .mean()
)

type_rain["label"] = type_rain["type"] + "_rain" + type_rain["heavy_rain"].astype(str)

plt.figure(figsize=(8, 5))
plt.bar(type_rain["label"], type_rain["tourists"])
plt.title("Average Tourists: Indoor vs Outdoor under Heavy Rain")
plt.xlabel("Type and Heavy Rain")
plt.ylabel("Average Tourists")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("charts/indoor_outdoor_heavy_rain.png", dpi=300)
plt.show()

print("圖表已完成，存在 charts 資料夾")



# =====================
# 10. 迴歸分析
# =====================

import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# 保險起見：重新整理分析用資料
analysis_df = df.copy()

analysis_df["tourists"] = pd.to_numeric(analysis_df["tourists"], errors="coerce")
analysis_df["rainfall"] = pd.to_numeric(analysis_df["rainfall"], errors="coerce")
analysis_df["temperature"] = pd.to_numeric(analysis_df["temperature"], errors="coerce")
analysis_df["rainy_days"] = pd.to_numeric(analysis_df["rainy_days"], errors="coerce")
analysis_df["sunshine"] = pd.to_numeric(analysis_df["sunshine"], errors="coerce")
analysis_df["rain_lag1"] = pd.to_numeric(analysis_df["rain_lag1"], errors="coerce")
analysis_df["rain_lag2"] = pd.to_numeric(analysis_df["rain_lag2"], errors="coerce")

analysis_df = analysis_df.dropna(subset=[
    "ln_tourists",
    "rainfall",
    "temperature",
    "rainy_days",
    "sunshine",
    "rain_lag1",
    "rain_lag2",
    "heavy_rain",
    "region",
    "type"
])

# =====================
# 10-1. 基本多變量迴歸
# =====================

model1 = smf.ols(
    formula="""
    ln_tourists ~ rainfall + temperature + rainy_days + sunshine 
    + C(region) + C(type)
    """,
    data=analysis_df
).fit()

# =====================
# 10-2. 交互作用迴歸
# 雨量 × 地區、雨量 × 景點類型
# =====================

model2 = smf.ols(
    formula="""
    ln_tourists ~ rainfall + temperature + rainy_days + sunshine 
    + C(region) + C(type)
    + rainfall:C(region)
    + rainfall:C(type)
    """,
    data=analysis_df
).fit()

# =====================
# 10-3. Lag 延遲效果迴歸
# =====================

model3 = smf.ols(
    formula="""
    ln_tourists ~ rainfall + rain_lag1 + rain_lag2
    + temperature + rainy_days + sunshine
    + C(region) + C(type)
    """,
    data=analysis_df
).fit()

# =====================
# 10-4. 門檻效果迴歸
# =====================

model4 = smf.ols(
    formula="""
    ln_tourists ~ rainfall + heavy_rain
    + temperature + rainy_days + sunshine
    + C(region) + C(type)
    + heavy_rain:C(type)
    """,
    data=analysis_df
).fit()

# =====================
# 10-5. 輸出迴歸結果文字檔
# =====================

with open("results/regression_results.txt", "w", encoding="utf-8") as f:
    f.write("模型一：基本多變量迴歸\n")
    f.write("=" * 80 + "\n")
    f.write(model1.summary().as_text())
    f.write("\n\n")

    f.write("模型二：交互作用迴歸 rainfall × region / rainfall × type\n")
    f.write("=" * 80 + "\n")
    f.write(model2.summary().as_text())
    f.write("\n\n")

    f.write("模型三：Lag 延遲效果迴歸\n")
    f.write("=" * 80 + "\n")
    f.write(model3.summary().as_text())
    f.write("\n\n")

    f.write("模型四：門檻效果迴歸 heavy_rain\n")
    f.write("=" * 80 + "\n")
    f.write(model4.summary().as_text())

print("迴歸分析完成！結果已輸出到 results/regression_results.txt")

# =====================
# 11. Random Forest 特徵重要性
# =====================

ml_df = analysis_df[[
    "ln_tourists",
    "rainfall",
    "temperature",
    "rainy_days",
    "sunshine",
    "rain_lag1",
    "rain_lag2",
    "heavy_rain",
    "region",
    "type"
]].copy()

ml_df = pd.get_dummies(
    ml_df,
    columns=["region", "type"],
    drop_first=True
)

X = ml_df.drop(columns=["ln_tourists"])
y = ml_df["ln_tourists"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)
r2 = r2_score(y_test, pred)

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_excel("results/random_forest_importance.xlsx", index=False)

plt.figure(figsize=(8, 5))
plt.barh(importance["feature"], importance["importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("results/random_forest_importance.png", dpi=300)
plt.show()

print("Random Forest 完成！")
print("R2：", r2)
print("特徵重要性已輸出到 results/random_forest_importance.xlsx")
print("特徵重要性圖已輸出到 results/random_forest_importance.png")
print(importance)