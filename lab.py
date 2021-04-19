# Tutorial: https://curiousily.com/posts/time-series-forecasting-with-lstm-for-daily-coronavirus-cases/
# Data: https://github.com/CSSEGISandData/COVID-19

from pathlib import Path

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

daily_cases = df.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
print(daily_cases.head())
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
print("Days in dataset:")
print(daily_cases.shape)

# ~3/4 рядов возьмём для обучения, ~1/4 для проверки
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
all_data = scaler.transform(np.expand_dims(daily_cases, axis=1))


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
X_train, Y_train = create_sequences(train_data, seq_length)
X_test, Y_test = create_sequences(test_data, seq_length)

X_all, Y_all = create_sequences(all_data, seq_length)
X_all = torch.from_numpy(X_all).float()
Y_all = torch.from_numpy(Y_all).float()

# Преобразование массивов NumPy в тензоры PyTorch
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()
X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).float()

# Каждый пример данных, используемый для тренировки, содержит последовательность из
# 5 точек данных и метки с реальным значением, которое должна уметь предсказывать модель
print(X_train.shape)
print(X_train[:2])
print(Y_train.shape)
print(Y_train[:2])
print(train_data[:10])

print("X_train.shape")
print(X_train.shape)
print("Y_train.shape")
print(Y_train.shape)


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
    num_epochs = 500
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    for t in range(num_epochs):
        model.reset_hidden_state()
        y_pred = model(X_train)
        loss = loss_fn(y_pred.float(), Y_train)
        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), Y_test)
            test_hist[t] = test_loss.item()
            print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        else:
            print(f'Epoch {t} train loss: {loss.item()}')
        train_hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return model.eval(), train_hist, test_hist

model_ready_to_use = Path("model.pt")
if model_ready_to_use.is_file():
    model = torch.load("model.pt")
    model.eval()
    print(model)
else:
    model = CoronaVirusPredictor(
        n_features=1,
        n_hidden=512,
        seq_len=seq_length,
        n_layers=3
    )
    model, train_hist, test_hist = train_model(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test
    )
    torch.save(model, "model.pt")
    # Смотрим потери
    plt.plot(train_hist, label="Training loss")
    plt.plot(test_hist, label="Test loss")
    plt.ylim((0, 15))
    plt.legend()
    plt.show()

DAYS_TO_PREDICT = 5
with torch.no_grad():
    test_seq = X_test[:1]
    preds = []
    for _ in range(len(X_test)):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

true_cases = scaler.inverse_transform(
    np.expand_dims(Y_test.flatten().numpy(), axis=0)
).flatten()
predicted_cases = scaler.inverse_transform(
    np.expand_dims(preds, axis=0)
).flatten()

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
plt.show()
