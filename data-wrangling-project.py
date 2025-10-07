# %%

import yfinance as yf
import pandas as pd 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# %%

ticker_cocoa = yf.Ticker("CC=F")
ticker_coffee = yf.Ticker("KC=F")
"""
compare the price of the cocoa & Coffee commodity with the price cacoa based drinks in the coffee dataset
https://www.kaggle.com/datasets/ayeshasiddiqa123/coffee-dataset
ticker = yf.Ticker("CC=F")


null hypothesis (H₀) and an alternative hypothesis (H₁):
H₀: There is no relationship between the price and the cocoa Comodity price.
H₁: There is a significant relationship between price and the cocoa Comodity price.

"""

# Get historical market data
# Download 2 years of historical data (daily)
hist_cocoa = ticker_cocoa.history(period="2y")
hist_coffee = ticker_coffee.history(period="3y")


# prepare stock dataframe
# change granularity frome datetime to date (for joining with other data)
df_ticker_cocoa = hist_cocoa[["Close"]].reset_index()  # reset_index to bring Date back as a column
df_ticker_cocoa["Date"] = pd.to_datetime(df_ticker_cocoa["Date"]).dt.date  # keep only date


df_ticker_coffee = hist_coffee[["Close"]].reset_index()  # reset_index to bring Date back as a column
df_ticker_coffee["Date"] = pd.to_datetime(df_ticker_cocoa["Date"]).dt.date  # keep only date

# df_merged.head()

# %%
# prepare sales dataframe
sales = pd.read_csv("Coffe_sales.csv")
sales["Date"] = pd.to_datetime(sales["Date"]).dt.date  # ensure same type

# %%
# find all drinks being served and divide between coffee and cocoa
print(sales["coffee_name"].unique())
coffee_drinks_list = ['Latte', 'Americano', 'Americano with Milk', 'Cortado', 'Espresso', 'Cappuccino']
cocoa_drinks_list = ["Cocoa","Hot Chocolate"]

# %%
# Filter for each bean type
df_cocoa = sales.copy()
df_cocoa = df_cocoa[df_cocoa["coffee_name"].isin(cocoa_drinks_list)]
print(len(df_cocoa))

df_coffee = sales.copy()
df_coffee = df_coffee[df_coffee["coffee_name"].isin(coffee_drinks_list)]

print(len(df_coffee))

# %%
# Merge COCOA
df_merged_cocoa = pd.merge(df_cocoa, df_ticker_cocoa, how="left", on="Date")

# controlling the length of the dataframes (no duplicated rows by merging)
print(len(df_cocoa))
print(len(df_merged_cocoa))

df_merged_cocoa.head()

# %%
# Merge COFFEE
df_merged_coffee = pd.merge(df_coffee, df_ticker_coffee, how="left", on="Date")

# controlling the length of the dataframes (no duplicated rows by merging)
print(len(df_coffee))
print(len(df_merged_coffee))

df_merged_coffee.head()

# %%
# checking for empty values
print("Empty values in df_merged_cocoa")
print(df_merged_cocoa[["money", "Close"]].isna().sum())
emty_values = df_merged_cocoa[df_merged_cocoa[ "Close"].isna()]

# %% [markdown]
# Why some days are missing:
# 
# Weekends and holidays:
#  - Futures markets only trade on weekdays and close on public holidays.
#  - So no data exists for Saturdays, Sundays, or exchange holidays.
# 
# Since the price doesnt change during that time I will fill the values from the previous day

# %%
# Sort by Date
df_merged_cocoa = df_merged_cocoa.sort_values(by=["Date"]).reset_index(drop=True)
# Fill down missing values (forward fill)
df_merged_cocoa[["Close"]] = df_merged_cocoa[["Close"]].ffill()

# checking for empty values again
print("Empty values in df_merged_cocoa after forward fill")
print(df_merged_cocoa[["money", "Close"]].isna().sum())
emty_values = df_merged_cocoa[df_merged_cocoa[ "Close"].isna()]

# %%
# Sort by Date
df_merged_coffee = df_merged_coffee.sort_values(by=["Date"]).reset_index(drop=True)
# Fill down missing values (forward fill)
df_merged_coffee[["Close"]] = df_merged_coffee[["Close"]].ffill()

# checking for empty values again
print("Empty values in df_merged_coffee after forward fill")
print(df_merged_coffee[["money", "Close"]].isna().sum())
emty_values = df_merged_coffee[df_merged_coffee[ "Close"].isna()]

# %% [markdown]
# ### T-test
# - Purpose: Compare the means of two groups (independent or paired).
# - Not suitable directly for testing correlation between two continuous variables.
# 
# ### Chi-square test
# - Purpose: Test for independence between categorical variables.
# - Not suitable directly for two continuous variables.
# ### Kolmogorov-Smirnov
# - Compares distributions, not relationships.
# - K-S does not test correlation. It only tests whether distributions are similar.
# 
# Most appropriate for the hypothesis with continuous money and Close:
# - Pearson correlation -> tests linear relationship
# - Spearman correlation -> tests monotonic relationship (more general)
# - Regression -> predicts money from Close and gives significance of the relationship

# %%
# Test linear  correlation of Cocoa
print("--- Test linear  correlation of Cocoa --- ")
# assuming df_merged has 'money' (sales) and 'Close' (cocoa price)
r, p_value = pearsonr(df_merged_cocoa["money"], df_merged_cocoa["Close"])

print("Pearson correlation coefficient:", r)
print("p-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Refuse the null hypothesis H₀: -> There is a significant relationship between price and the cocoa Comodity price to be found in the data.")
else:
    print("Cannot refuse the null hypthesis: -> There is NO significant relationship between price  and the cocoa Comodity price to be found in the data.")

# %%
# Test monotonic correlation
print("--- Test monotonic correlation of Cocoa --- ")

rho, p_value = spearmanr(df_merged_cocoa["money"], df_merged_cocoa["Close"])

print("Spearman correlation:", rho)
print("p-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Refuse the null hypothesis H₀: -> There is a significant relationship between price and the cocoa Comodity price to be found in the data.")
else:
    print("Cannot refuse the null hypthesis: -> There is NO significant relationship between price  and the cocoa Comodity price to be found in the data.")

# %%
print("########## COFFEE #################")

# Test linear  correlation of COFFEE
print("--- Test linear  correlation of COFFEE --- ")
# assuming df_merged has 'money' (sales) and 'Close' (cocoa price)
r, p_value = pearsonr(df_merged_coffee["money"], df_merged_coffee["Close"])

print("Pearson correlation coefficient:", r)
print("p-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Refuse the null hypothesis H₀: -> There is a significant relationship between price and the coffee Comodity price to be found in the data.")
else:
    print("Cannot refuse the null hypthesis: -> There is NO significant relationship between price  and the coffee Comodity price to be found in the data.")

# %%
print("--- Test monotonic correlation of COFFEE --- ")

rho, p_value = spearmanr(df_merged_coffee["money"], df_merged_coffee["Close"])

print("Spearman correlation:", rho)
print("p-value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Refuse the null hypothesis H₀: -> There is a significant relationship between price and the cocoa Comodity price to be found in the data.")
else:
    print("Cannot refuse the null hypthesis: -> There is NO significant relationship between price  and the cocoa Comodity price to be found in the data.")

# %% [markdown]
# # Conclustion for Predictions
# Since there is a correlation between the coffee price and the commodity price and no significant correlation between cocoa and the comodity prices, I will focus on coffee and try to predict accurate prices based on the commodity price:
# 
# ### ARIMA (AutoRegressive Integrated Moving Average) is a time series forecasting model.
# It’s best when your target variable depends on its own past values (autoregression) and/or trends over time.
# Typical use case: predicting future sales based on past sales.
# It can also include exogenous variables (like cocoa price) via ARIMAX.
# -> Best to predict future sales and account for seasonality, trends, autocorrelation
# 
# ### Ordinary least squares (OLS) regression
# - Observations are independent (no time dependency)
# - Linear relationship between Close and money
# -> Best for testing H₀/H₁ (does cocoa price affect sales?)

# %%
df_merged_coffee.head()

# %%
drink_list = []
rsme_list = []
r2_list = []


for drink in coffee_drinks_list:
    df_filtered = df_merged_coffee.copy()
    df_filtered = df_filtered[df_filtered["coffee_name"] == drink]

    drink_list.append(drink)

    X = df_filtered['Close']  # predictor
    y = df_filtered['money']  # response

    # Add a constant term for the intercept
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Predict new values
    df_filtered['money_pred'] = model.predict(X)

    # RMSE
    rmse = np.sqrt(mean_squared_error(y, df_filtered['money_pred']))
    rsme_list.append(rmse)

    # R²
    r2 = r2_score(y, df_filtered['money_pred'])
    r2_list.append(r2)


    # Plot actual vs predicted
    plt.scatter(df_filtered['Close'], df_filtered['money'], label='Actual')
    plt.plot(df_filtered['Close'], df_filtered['money_pred'], color='red', label='Predicted')
    plt.xlabel('Coffee commodity Price (Close)')
    plt.ylabel('Sales (money)')
    plt.title(f'Linear Regression: Predict Price of {drink} from comodity Price')
    plt.legend()
    plt.show()



for i, drink in enumerate(drink_list):
    print("---------------------------------------------------")
    print(drink)
    print("RMSE:", rsme_list[i])
    print("R²:", r2_list[i])

# %%
df_merged_coffee.head
for drink in coffee_drinks_list:
    df_filtered = df_merged_coffee.copy()
    df_filtered = df_filtered[df_filtered["coffee_name"] == drink]
    # Convert 'Date' column to datetime
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    df_filtered = df_filtered.set_index('Date')

     # Plot Age and Score for the two rows
    df_filtered[['money', 'Close']].plot(kind='line', figsize=(10,5))
    plt.title(f'Close and Money for {drink}')
    plt.ylabel('Value')
    plt.xlabel('Date')
    plt.show()

# %% [markdown]
# ### Merge ticker with calander to have quotes for every day

# %%

df_coffee_sales_copy = df_merged_coffee.set_index("Date")
# Drop rows where index is NaT
df_ticker_data_copy = df_ticker_coffee.copy()

# Define start and end dates
start_date = '2024-01-01'
end_date = '2025-10-06'

# Create daily date range
dates = pd.date_range(start=start_date, end=end_date, freq='D')
# Convert to DataFrame
df_calendar = pd.DataFrame({'Date': dates})

df_ticker_data_copy["Date"] = pd.to_datetime(df_ticker_data_copy["Date"])
df_merged_calendar_coffee = pd.merge(df_calendar, df_ticker_data_copy, how="left", on="Date")

# order by date
df_merged_calendar_coffee = df_merged_calendar_coffee.sort_values(by="Date")

df_merged_calendar_coffee[["Close"]] = df_merged_calendar_coffee[["Close"]].ffill()
# droping first row (blank)
df_merged_calendar_coffee = df_merged_calendar_coffee.dropna(subset=["Close"])

# Set 'Date' as the index
df_merged_calendar_coffee = df_merged_calendar_coffee.set_index('Date')

print(df_merged_calendar_coffee.head(20))


# %%


for drink in coffee_drinks_list:
    df_filtered = df_coffee_sales_copy[df_coffee_sales_copy["coffee_name"] == drink].copy()

    # Remove duplicate dates
    df_filtered = df_filtered.groupby(df_filtered.index).agg({
        'money': 'first',
        'coffee_name': 'first' 
    })

    # Endogenous variable: coffee price (Money)
    y = df_filtered['money']

    # Exogenous variable: stock Close
    X = df_merged_calendar_coffee[['Close']].loc[y.index]
    forecast_steps = 180
    X_test = df_merged_calendar_coffee[['Close']].iloc[-forecast_steps:]

    # # Fit ARIMAX
    model = SARIMAX(y, exog=X, order=(1,1,1))
    model_fit = model.fit(disp=False)

    # model_fit.predict(exog=X_predict)
    forecast = model_fit.get_forecast(steps=forecast_steps, exog=X_test)
    forecast_values = forecast.predicted_mean

    forecast_values.index = X_test.index

    # Plot
    y.plot(label='Actual', figsize=(10,5))
    forecast_values.plot(label='Forecast', color='red')

    plt.title(f'Forecast Coffee Price for {drink}')
    plt.xlabel('Date')
    plt.ylabel('Money')
    plt.legend()
    plt.show()

      

# %% [markdown]
# # Merge 

# %% [markdown]
# # testing for seasonality in the Comodity price

# %%

# Multiplicative Decomposition
decomposition_mult = seasonal_decompose(df_merged_calendar_coffee, model='additive', period=90)

plt.figure(figsize=(14, 12))
plt.subplot(411)
plt.plot(decomposition_mult.observed)
plt.title('Original Time Series (Multiplicative)')
plt.grid(True, alpha=0.3)

plt.subplot(412)
plt.plot(decomposition_mult.trend)
plt.title('Trend')
plt.grid(True, alpha=0.3)

plt.subplot(413)
plt.plot(decomposition_mult.seasonal)
plt.title('Seasonality')
plt.grid(True, alpha=0.3)

plt.subplot(414)
plt.plot(decomposition_mult.resid)
plt.title('Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


