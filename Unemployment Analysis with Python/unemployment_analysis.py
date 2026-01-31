# =====================================
# 1. Import Required Libraries
# =====================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =====================================
# 2. Load Datasets
# =====================================
df_india = pd.read_csv("Unemployment in India.csv")
df_covid = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

print("India Dataset:")
print(df_india.head())

print("\nCovid Dataset:")
print(df_covid.head())

# =====================================
# 3. Data Cleaning
# =====================================

# Rename columns for consistency
df_india.columns = df_india.columns.str.strip()
df_covid.columns = df_covid.columns.str.strip()

# Convert Date column
df_india['Date'] = pd.to_datetime(df_india['Date'], dayfirst=True)
df_covid['Date'] = pd.to_datetime(df_covid['Date'], dayfirst=True)

# Drop missing values
df_india.dropna(inplace=True)
df_covid.dropna(inplace=True)

# =====================================
# 4. Unemployment Trend Over Time
# =====================================

plt.plot(df_india['Date'], df_india['Estimated Unemployment Rate (%)'])
plt.title("Unemployment Rate in India Over Time")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# =====================================
# 5. Covid-19 Impact Analysis
# =====================================

pre_covid = df_india[df_india['Date'].dt.year < 2020]
covid = df_india[df_india['Date'].dt.year >= 2020]

plt.plot(pre_covid['Date'],
         pre_covid['Estimated Unemployment Rate (%)'],
         label="Pre-Covid")

plt.plot(covid['Date'],
         covid['Estimated Unemployment Rate (%)'],
         label="Covid Period")

plt.title("Impact of Covid-19 on Unemployment in India")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()

# =====================================
# 6. Monthly / Seasonal Trend
# =====================================

df_india['Month'] = df_india['Date'].dt.month
monthly_avg = df_india.groupby('Month')['Estimated Unemployment Rate (%)'].mean()

sns.barplot(x=monthly_avg.index, y=monthly_avg.values)
plt.title("Average Monthly Unemployment Rate (Seasonal Trend)")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# =====================================
# 7. Yearly Unemployment Trend
# =====================================

df_india['Year'] = df_india['Date'].dt.year
yearly_avg = df_india.groupby('Year')['Estimated Unemployment Rate (%)'].mean()

sns.lineplot(x=yearly_avg.index, y=yearly_avg.values, marker='o')
plt.title("Yearly Average Unemployment Rate in India")
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")
plt.show()

# =====================================
# 8. Region-wise Analysis
# =====================================

region_avg = df_india.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values()

region_avg.plot(kind='barh')
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Region")
plt.show()

# =====================================
# 9. Key Insights
# =====================================

print("\nKey Insights:")
print("Highest Unemployment Rate:",
      df_india['Estimated Unemployment Rate (%)'].max())

print("Lowest Unemployment Rate:",
      df_india['Estimated Unemployment Rate (%)'].min())

print("Average Unemployment Rate:",
      round(df_india['Estimated Unemployment Rate (%)'].mean(), 2))

print("\nPre-Covid Average:",
      round(pre_covid['Estimated Unemployment Rate (%)'].mean(), 2))

print("Covid Period Average:",
      round(covid['Estimated Unemployment Rate (%)'].mean(), 2))

# =====================================
# 10. Conclusion
# =====================================

print("\nConclusion:")
print("The unemployment rate in India increased sharply during the Covid-19 period.")
print("Clear seasonal patterns are observed across months.")
print("Regional differences highlight uneven employment recovery.")
print("This analysis can help policymakers design targeted employment policies.")
