import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Drop 'CustomerID' and 'Gender' columns
df.drop(columns=["CustomerID", "Gender"], inplace=True)

# Normalize the remaining numerical features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Save the preprocessed dataset
df_scaled.to_csv("mall_customers_preprocessed2.csv", index=False)

# Display the first few rows
print(df_scaled.head())
