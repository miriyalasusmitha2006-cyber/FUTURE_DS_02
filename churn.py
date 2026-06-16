import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load dataset
# ---------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

# ---------------------------
# 2. Data Cleaning
# ---------------------------
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# ---------------------------
# 3. Churn conversion
# ---------------------------
df['ChurnFlag'] = df['Churn'].apply(lambda x: 1 if x == "Yes" else 0)

# ---------------------------
# 4. Business Metrics
# ---------------------------
churn_rate = df['ChurnFlag'].mean() * 100
retention_rate = 100 - churn_rate

print("\n--- BUSINESS METRICS ---")
print("Churn Rate:", churn_rate)
print("Retention Rate:", retention_rate)

# ---------------------------
# 5. Churn by Contract
# ---------------------------
contract_churn = df.groupby('Contract')['ChurnFlag'].mean() * 100

contract_churn.plot(kind='bar')
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn %")
plt.savefig("graph1.png")
plt.show()

# ---------------------------
# 6. Tenure Analysis
# ---------------------------
df['tenure_group'] = pd.cut(df['tenure'],
                             bins=[0,12,24,48,72],
                             labels=['0-12','12-24','24-48','48-72'])

tenure_churn = df.groupby('tenure_group')['ChurnFlag'].mean() * 100

tenure_churn.plot(kind='bar')
plt.title("Churn by Tenure Group")
plt.ylabel("Churn %")
plt.savefig("tenure.png")
plt.show()

# ---------------------------
# 7. Internet Service Impact
# ---------------------------
internet_churn = df.groupby('InternetService')['ChurnFlag'].mean() * 100

internet_churn.plot(kind='bar')
plt.title("Churn by Internet Service")
plt.ylabel("Churn %")
plt.savefig("internet_service.png")
plt.show()

# ---------------------------
# 8. Monthly Charges vs Churn
# ---------------------------
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("monthly_charges.png")
plt.show()