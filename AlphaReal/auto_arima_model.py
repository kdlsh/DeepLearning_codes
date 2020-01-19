#pip install pyramid-arima

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyramid import auto_arima

df = pd.read_csv('data.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.index = df.Month
df = df.drop('Month', axis=1)
df.head()

plt.plot(df)
plt.xlabel('time')
plt.ylabel('passengers')
plt.plot()

train = df[:int(0.8*(len(df)))]
test = df[int(0.8*(len(df))):]
print(train.shape)
print(test.shape)

model = auto_arima(train, trace=True, start_q=0, start_p=0, start_Q=0,
				max_p=10, max_q=10, max_P=10, max_Q=10, seasonal=True,
				stepwise=False, suppress_warnings=True, D=1, max_D=10,
				error_action='ignore', approximation=False)
mode.fit(train)

y_pred = model.predict(n_periods=len(valid))
from sklearn.metrics import r2_score
acc = r2_score(valid.values, y_pred)
print(acc)