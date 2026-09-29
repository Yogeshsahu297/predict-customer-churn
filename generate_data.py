"""
Generates a synthetic dataset that mirrors the structure and column names of
the classic Kaggle "Credit Card Customers" / BankChurners dataset used in the
Udacity "Predict Customer Churn" project.

Run this once to produce data/bank_data.csv. If you have the real
BankChurners.csv from Kaggle, you can simply drop it into data/bank_data.csv
instead and skip this script -- churn_library.py does not care which one it
reads, as long as the column names match.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000

attrition = np.random.choice(
    ["Existing Customer", "Attrited Customer"], size=N, p=[0.84, 0.16]
)

gender = np.random.choice(["M", "F"], size=N)
education = np.random.choice(
    ["High School", "Graduate", "Uneducated", "College",
     "Post-Graduate", "Doctorate", "Unknown"],
    size=N, p=[0.20, 0.30, 0.15, 0.12, 0.08, 0.05, 0.10]
)
marital = np.random.choice(
    ["Married", "Single", "Divorced", "Unknown"], size=N, p=[0.46, 0.39, 0.09, 0.06]
)
income = np.random.choice(
    ["Less than $40K", "$40K - $60K", "$60K - $80K",
     "$80K - $120K", "$120K +", "Unknown"],
    size=N, p=[0.35, 0.18, 0.14, 0.15, 0.08, 0.10]
)
card = np.random.choice(
    ["Blue", "Silver", "Gold", "Platinum"], size=N, p=[0.93, 0.05, 0.015, 0.005]
)

df = pd.DataFrame({
    "CLIENTNUM": np.arange(768805000, 768805000 + N),
    "Attrition_Flag": attrition,
    "Customer_Age": np.random.randint(26, 74, size=N),
    "Gender": gender,
    "Dependent_count": np.random.randint(0, 6, size=N),
    "Education_Level": education,
    "Marital_Status": marital,
    "Income_Category": income,
    "Card_Category": card,
    "Months_on_book": np.random.randint(13, 57, size=N),
    "Total_Relationship_Count": np.random.randint(1, 7, size=N),
    "Months_Inactive_12_mon": np.random.randint(0, 7, size=N),
    "Contacts_Count_12_mon": np.random.randint(0, 7, size=N),
    "Credit_Limit": np.round(np.random.uniform(1438, 34516, size=N), 1),
    "Total_Revolving_Bal": np.random.randint(0, 2517, size=N),
})
df["Avg_Open_To_Buy"] = np.round(df["Credit_Limit"] - df["Total_Revolving_Bal"], 1)
df["Total_Amt_Chng_Q4_Q1"] = np.round(np.random.uniform(0, 3.4, size=N), 3)
df["Total_Trans_Amt"] = np.random.randint(510, 18500, size=N)
df["Total_Trans_Ct"] = np.random.randint(10, 140, size=N)
df["Total_Ct_Chng_Q4_Q1"] = np.round(np.random.uniform(0, 3.7, size=N), 3)
df["Avg_Utilization_Ratio"] = np.round(
    np.where(df["Credit_Limit"] > 0, df["Total_Revolving_Bal"] / df["Credit_Limit"], 0), 3
)

# Nudge a few numeric features so attrited customers look plausibly different
# (keeps the resulting models from being pure noise).
mask = df["Attrition_Flag"] == "Attrited Customer"
df.loc[mask, "Total_Trans_Ct"] = (df.loc[mask, "Total_Trans_Ct"] * 0.55).astype(int)
df.loc[mask, "Total_Trans_Amt"] = (df.loc[mask, "Total_Trans_Amt"] * 0.6).astype(int)
df.loc[mask, "Total_Relationship_Count"] = np.clip(
    df.loc[mask, "Total_Relationship_Count"] - 1, 1, 6
)
df.loc[mask, "Months_Inactive_12_mon"] = np.clip(
    df.loc[mask, "Months_Inactive_12_mon"] + 1, 0, 6
)

# The two "Naive Bayes" columns present in the original Kaggle export -- the
# project instructions have you drop these immediately on load.
df["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"] = np.random.uniform(0, 1, size=N)
df["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"] = np.random.uniform(0, 1, size=N)

df.to_csv("data/bank_data.csv", index=False)
print(f"Wrote data/bank_data.csv with shape {df.shape}")
