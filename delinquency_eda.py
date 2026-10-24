"""
EDA - Delinquency Prediction Dataset
Run:  python delinquency_eda.py   (dataset expected at data/Delinquency_prediction_dataset.xlsx)
Needs: pandas, numpy, matplotlib, seaborn, scipy, openpyxl
Install: pip install pandas numpy matplotlib seaborn scipy openpyxl
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # remove this line if you want plt.show() windows
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

FILE = "data/Delinquency_prediction_dataset.xlsx"   # place the dataset here (not included in this repo)
OUT = "eda_plots"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")
TARGET = "Delinquent_Account"


def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=130)
    plt.close()


# ------------------------------------------------------------------
# 1. LOAD + FIRST LOOK
# ------------------------------------------------------------------
df = pd.read_excel(FILE)
print("Shape:", df.shape)                 # rows, columns
print(df.head())
print(df.info())                          # dtypes + non-null counts
print(df.describe().T)                    # numeric summary

# ------------------------------------------------------------------
# 2. DATA QUALITY CHECKS
# ------------------------------------------------------------------
# 2a. Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print("\nMissing values:\n", pd.DataFrame({"count": missing, "pct": missing_pct})[missing > 0])

plt.figure(figsize=(9, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing values map (light stripes = missing)")
save("01_missing_map")

# 2b. Duplicates
print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate Customer_IDs:", df["Customer_ID"].duplicated().sum())

# 2c. Inconsistent category labels (spelling / case)
print("\nEmployment_Status raw values:\n", df["Employment_Status"].value_counts())

# 2d. Range checks against the data guide
print("\nCredit_Utilization > 1 :", (df["Credit_Utilization"] > 1).sum())
print("Credit_Score outside 300-850:", (~df["Credit_Score"].dropna().between(300, 850)).sum())

# ------------------------------------------------------------------
# 3. CLEANING
# ------------------------------------------------------------------
# 3a. Standardise Employment_Status:  'EMP', 'employed', 'Employed' -> 'Employed'
emp_map = {
    "emp": "Employed", "employed": "Employed",
    "self-employed": "Self-Employed",
    "unemployed": "Unemployed",
    "retired": "Retired",
}
df["Employment_Status"] = (df["Employment_Status"].str.strip().str.lower()
                           .map(emp_map))
print("\nEmployment_Status after cleaning:\n", df["Employment_Status"].value_counts())

# 3b. Utilization & DTI are stored as fractions (0.49), not percentages. Convert to %
df["Credit_Utilization_pct"] = df["Credit_Utilization"] * 100
df["DTI_pct"] = df["Debt_to_Income_Ratio"] * 100

# 3c. Flag, then cap, utilization above 100%
df["Util_over_100"] = (df["Credit_Utilization_pct"] > 100).astype(int)
df["Credit_Utilization_pct"] = df["Credit_Utilization_pct"].clip(upper=100)

# 3d. Impute missing numerics with MEDIAN (robust to outliers) + keep a 'was missing' flag
for col in ["Income", "Loan_Balance", "Credit_Score"]:
    df[f"{col}_missing"] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(df[col].median())

# 3e. Month_1..Month_6 are text -> numeric codes (0 on-time, 1 late, 2 missed)
months = [f"Month_{i}" for i in range(1, 7)]
pay_map = {"On-time": 0, "Late": 1, "Missed": 2}
for m in months:
    df[m + "_code"] = df[m].map(pay_map)

# ------------------------------------------------------------------
# 4. FEATURE ENGINEERING (useful summaries of the 6-month history)
# ------------------------------------------------------------------
codes = [m + "_code" for m in months]
df["Late_Months"] = (df[months] == "Late").sum(axis=1)
df["Missed_Months"] = (df[months] == "Missed").sum(axis=1)
df["On_Time_Months"] = (df[months] == "On-time").sum(axis=1)
df["Payment_Score"] = df[codes].sum(axis=1)             # 0 = perfect, 12 = worst
df["Recent_3M_Score"] = df[codes[-3:]].sum(axis=1)       # last 3 months only
df["Loan_to_Income"] = df["Loan_Balance"] / df["Income"]

# ------------------------------------------------------------------
# 5. UNIVARIATE ANALYSIS
# ------------------------------------------------------------------
# 5a. Target balance
counts = df[TARGET].value_counts()
print("\nTarget distribution:\n", counts, "\n", (counts / len(df) * 100).round(1))
plt.figure(figsize=(4.5, 4))
ax = sns.countplot(x=TARGET, data=df, palette=["#4C9F70", "#D9534F"], hue=TARGET, legend=False)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.title("Target: Delinquent_Account (0=No, 1=Yes)")
save("02_target_balance")

# 5b. Numeric distributions (histogram + KDE)
num_cols = ["Age", "Income", "Credit_Score", "Credit_Utilization_pct", "Missed_Payments",
            "Loan_Balance", "DTI_pct", "Account_Tenure"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, c in zip(axes.ravel(), num_cols):
    sns.histplot(df[c], kde=True, ax=ax, color="#3B7DD8")
    ax.set_title(f"{c}\nskew={df[c].skew():.2f}")
save("03_numeric_distributions")

# 5c. Boxplots + IQR outlier count
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, c in zip(axes.ravel(), num_cols):
    sns.boxplot(y=df[c], ax=ax, color="#F0AD4E")
    ax.set_title(c)
save("04_boxplots")

print("\nOutliers by IQR rule:")
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    n = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()
    print(f"  {c:25s}: {n}")

# 5d. Categorical counts
cat_cols = ["Employment_Status", "Credit_Card_Type", "Location"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, c in zip(axes, cat_cols):
    order = df[c].value_counts().index
    sns.countplot(x=c, data=df, order=order, ax=ax, hue=c, palette="viridis", legend=False)
    ax.set_title(c)
    ax.tick_params(axis="x", rotation=30)
save("05_categorical_counts")

# 5e. Payment status per month
pay_long = df[months].melt(var_name="Month", value_name="Status")
plt.figure(figsize=(9, 4.5))
sns.countplot(x="Month", hue="Status", data=pay_long,
              hue_order=["On-time", "Late", "Missed"],
              palette=["#4C9F70", "#F0AD4E", "#D9534F"])
plt.title("Payment status by month")
save("06_payment_status_by_month")

# ------------------------------------------------------------------
# 6. BIVARIATE ANALYSIS vs TARGET
# ------------------------------------------------------------------
# 6a. Numeric features split by target
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, c in zip(axes.ravel(), num_cols):
    sns.boxplot(x=TARGET, y=c, data=df, ax=ax, hue=TARGET,
                palette=["#4C9F70", "#D9534F"], legend=False)
    ax.set_title(f"{c} vs {TARGET}")
save("07_numeric_vs_target")

# 6b. Statistical test for each numeric feature (Mann-Whitney U: no normality needed)
print("\nMann-Whitney U test (numeric vs target):")
for c in num_cols + ["Late_Months", "Missed_Months", "Payment_Score", "Recent_3M_Score"]:
    a = df.loc[df[TARGET] == 0, c]
    b = df.loc[df[TARGET] == 1, c]
    p = stats.mannwhitneyu(a, b).pvalue
    print(f"  {c:25s} mean(0)={a.mean():10.2f}  mean(1)={b.mean():10.2f}  p={p:.3f}"
          f"  {'<-- significant' if p < 0.05 else ''}")

# 6c. Delinquency rate per category + chi-square test
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
print("\nChi-square test (categorical vs target):")
for ax, c in zip(axes, cat_cols):
    rate = df.groupby(c)[TARGET].mean().sort_values(ascending=False) * 100
    sns.barplot(x=rate.index, y=rate.values, ax=ax, hue=rate.index, palette="rocket", legend=False)
    ax.axhline(df[TARGET].mean() * 100, color="k", ls="--", label="overall")
    ax.set_ylabel("Delinquency rate (%)")
    ax.set_title(c)
    ax.tick_params(axis="x", rotation=30)
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[c], df[TARGET]))
    print(f"  {c:20s} chi2={chi2:.2f}  p={p:.3f}")
save("08_delinquency_rate_by_category")

# 6d. Delinquency rate by Missed_Payments and by payment score
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
df.groupby("Missed_Payments")[TARGET].mean().mul(100).plot(kind="bar", ax=axes[0], color="#D9534F")
axes[0].set_title("Delinquency rate (%) by Missed_Payments")
df.groupby("Payment_Score")[TARGET].mean().mul(100).plot(kind="bar", ax=axes[1], color="#3B7DD8")
axes[1].set_title("Delinquency rate (%) by 6-month Payment_Score")
save("09_rate_by_missed_and_score")

# ------------------------------------------------------------------
# 7. MULTIVARIATE ANALYSIS
# ------------------------------------------------------------------
corr_cols = num_cols + [TARGET, "Late_Months", "Missed_Months", "Payment_Score", "Loan_to_Income"]
plt.figure(figsize=(11, 9))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation heatmap")
save("10_correlation_heatmap")

print("\nCorrelation with target:\n",
      df[corr_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False).round(3))

# Pairplot on a few key features
sns.pairplot(df[["Credit_Score", "Income", "Credit_Utilization_pct", "DTI_pct", TARGET]],
             hue=TARGET, palette=["#4C9F70", "#D9534F"], plot_kws={"alpha": .5, "s": 20})
plt.savefig(f"{OUT}/11_pairplot.png", dpi=110)
plt.close()

# Heatmap: delinquency rate by Employment x Card type
pivot = df.pivot_table(index="Employment_Status", columns="Credit_Card_Type",
                       values=TARGET, aggfunc="mean") * 100
plt.figure(figsize=(8, 4.5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds")
plt.title("Delinquency rate (%): Employment x Card type")
save("12_employment_x_card")

# ------------------------------------------------------------------
# 8. SAVE CLEANED DATA FOR MODELLING
# ------------------------------------------------------------------
df.to_csv("delinquency_cleaned.csv", index=False)
print("\nDone. Plots in ./%s, cleaned data in delinquency_cleaned.csv" % OUT)
