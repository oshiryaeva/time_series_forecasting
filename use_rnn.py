# Tutorial: https://curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
# Data: https://github.com/CSSEGISandData/COVID-19

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams
from sklearn.preprocessing import MinMaxScaler

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(url, index_col=0)

# Удаляем столбцы province, country, latitude и longitude за ненадобностью:
df = df.iloc[:, 4:]
# Проверяем, есть ли пустые значения:
df.isnull().sum().sum()
# Суммируем ряды, получаем совокупное количество случаев за день:
daily_cases = df.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
daily_cases.head()
plt.plot(daily_cases)
plt.title("Cumulative daily cases")
plt.show()

# Убираем накопление, вычитая текущее значение из предыдущего.
# Первое значение последовательности сохраняем.
daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
print("daily_cases.head()")
print(daily_cases.head())
plt.plot(daily_cases)
plt.title("Daily cases")
plt.show()

# Смотрим, за сколько дней у нас данные
print("daily_cases.shape")
print(daily_cases.shape)

# ~3/4 рядов возьмём для обучения, ~1/4 для проверки
test_data_size = 100
train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]
print("train_data.shape")
print(train_data.shape)

# Нормализуем данные (приведём их к значениям между 0 и 1) для повышения точности и скорости обучения
# Для нормализации возьмем MinMaxScaler из scikit-learn:
scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))
train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))
all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))
print("train_data.shape")
print(train_data.shape)
print("test_data.shape")
print(test_data.shape)
print("all_data.shape")
print(all_data.shape)


def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 5

X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

X_all, y_all = create_sequences(all_data, seq_length)
X_all = torch.from_numpy(X_all).float()
y_all = torch.from_numpy(y_all).float()

model = torch.load("model.pt")
model.eval()
print(model)

DAYS_TO_PREDICT = 5
with torch.no_grad():
    test_seq = X_all[:1]
    preds = []
    for _ in range(DAYS_TO_PREDICT):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()
    true_cases = scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()
    predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()
print(daily_cases.index[-1])

predicted_index = pd.date_range(
    start=daily_cases.index[-1],
    periods=DAYS_TO_PREDICT + 1,
    closed='right'
)
predicted_cases = pd.Series(
    data=predicted_cases,
    index=predicted_index
)

plt.plot(
    daily_cases.index[:len(train_data)],
    scaler.inverse_transform(train_data).flatten(),
    label='Historical Daily Cases'
)
plt.plot(
    daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
    true_cases,
    label='Real Daily Cases'
)
plt.plot(
    daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
    predicted_cases,
    label='Predicted Daily Cases'
)
plt.legend()
