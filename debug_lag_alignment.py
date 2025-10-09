import pandas as pd

df = pd.read_csv('EURUSD_multi_timeframe_enriched.csv')
print(df[['daily_close', 'daily_direction', 'day_outcome_lag1']].head(30))
print('Check: day_outcome_lag1.shift(-1) vs daily_direction')
print(pd.DataFrame({
    'day_outcome_lag1.shift(-1)': df['day_outcome_lag1'].shift(-1).head(30),
    'daily_direction': df['daily_direction'].head(30)
}))
print('Check: daily_direction calculation')
print(df['daily_close'].head(10).diff())
