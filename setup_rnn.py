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

from model import CoronaVirusPredictor

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Данные о количестве зарегистрированных случаев covid-19 по странам в день. Датасет обновляется ежедневно.
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(url, index_col=0)
print(df.head(5))

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
daily_cases.head()
plt.plot(daily_cases)
plt.title("Daily cases")
plt.show()

# Смотрим, за сколько дней у нас данные
print(daily_cases.shape)

# ~3/4 рядов возьмём для обучения, 1/4 для проверки
test_data_size = 100
train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]
print(train_data.shape)

# Нормализуем данные (приведём их к значениям между 0 и 1) для повышения точности и скорости обучения
# Для нормализации возьмем MinMaxScaler из scikit-learn:
scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))
train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))


# Разобьем данные на меньшие последовательности случаев за день:
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
# Каждый пример данных, используемый для тренировки, содержит последовательность из
# 5 точек данных и меткис с реальным значением, которое должна уметь предсказывать модель
print(X_train.shape)
print(X_train[:2])
print(y_train.shape)
print(y_train[:2])
print(train_data[:10])

# Функция тренировки модели
def train_model(
        model,
        train_data,
        train_labels,
        test_data=None,
        test_labels=None
):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 60
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        model.reset_hidden_state()
        y_pred = model(X_train)
        loss = loss_fn(y_pred.float(), y_train)
        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()
            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')
        train_hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model.eval(), train_hist, test_hist

# Создаём экземпляр модели и обучаем её
model = CoronaVirusPredictor(
    n_features=1,
    n_hidden=512,
    seq_len=seq_length,
    n_layers=2
)
model, train_hist, test_hist = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
)

torch.save(model, "model.pt")

# Смотрим потери
plt.plot(train_hist, label="Training loss")
plt.plot(test_hist, label="Test loss")
plt.ylim((0, 5))
plt.legend()
plt.show()
