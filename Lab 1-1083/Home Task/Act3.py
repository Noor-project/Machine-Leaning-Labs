import os
import time
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"
BASE = os.path.dirname(os.path.abspath(__file__))
local_file = os.path.join(BASE, "covid_combined.csv")

# Download once and keep a local copy. pd.read_csv(url) prints nothing while it
# downloads, so a slow connection looks exactly like a frozen script.
if not os.path.exists(local_file):
    print("Downloading dataset (large file, about 170,000 rows)...")
    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
    except Exception as e:
        raise SystemExit(
            f"Download failed: {e}\n"
            "Open the URL in your browser, save the file as 'covid_combined.csv' "
            "in the same folder as this script, then run again."
        )
    with open(local_file, "wb") as f:
        f.write(content)
    print(f"Downloaded in {time.time() - start:.1f} seconds")

df = pd.read_csv(local_file, parse_dates=['Date'])

print("Columns:", df.columns.tolist())

# 1. Extract only Pakistan's data
pk = df[df['Country/Region'] == 'Pakistan'].copy()
pk = pk.sort_values('Date').reset_index(drop=True)
print("\nPakistan rows:", pk.shape[0])

# 2. Handle missing values
print("\nMissing values:")
print(pk.isnull().sum())

# Province/State is empty for every Pakistan row (country-level data), so drop it
pk = pk.drop(columns=['Province/State'])
# Confirmed/Recovered/Deaths are cumulative counts: forward-fill any gaps, then 0 for leading gaps
num_cols = ['Confirmed', 'Recovered', 'Deaths']
pk[num_cols] = pk[num_cols].ffill().fillna(0)
print("\nMissing values after handling:")
print(pk.isnull().sum())

# 3. Line chart: cases over time (cumulative confirmed)
plt.figure(figsize=(10, 5))
plt.plot(pk['Date'], pk['Confirmed'], label='Confirmed (cumulative)')
plt.xlabel('Date')
plt.ylabel('Cases')
plt.title('COVID-19 Confirmed Cases in Pakistan')
plt.legend()
plt.grid(True)
plt.show()

# 4. Day with the highest confirmed cases
# "Confirmed" is cumulative, so its maximum is just the last day.
last = pk.loc[pk['Confirmed'].idxmax()]
print(f"\nHighest cumulative confirmed: {int(last['Confirmed'])} on {last['Date'].date()}")

# The more meaningful answer: the day with the most NEW cases
pk['Daily_Cases'] = pk['Confirmed'].diff().fillna(pk['Confirmed'].iloc[0])
peak = pk.loc[pk['Daily_Cases'].idxmax()]
print(f"Highest daily new cases: {int(peak['Daily_Cases'])} on {peak['Date'].date()}")

# 5. Outliers in daily cases using a boxplot
plt.figure(figsize=(6, 5))
plt.boxplot(pk['Daily_Cases'])
plt.ylabel('Daily New Cases')
plt.title('Boxplot of Daily Cases (Pakistan)')
plt.show()

Q1 = pk['Daily_Cases'].quantile(0.25)
Q3 = pk['Daily_Cases'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
n_out = ((pk['Daily_Cases'] < lower) | (pk['Daily_Cases'] > upper)).sum()
print(f"\nIQR limits: {lower:.0f} to {upper:.0f}")
print(f"Number of outlier days: {n_out}")