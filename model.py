import torch
from torch import nn


# Класс создания модели, наследующий torch.nn.Module:
class CoronaVirusPredictor(nn.Module):
    # В конструкторе инициализируем поля и создаем слои модели
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaVirusPredictor, self).__init__()
        # количество блоков LSTM на слой
        self.n_hidden = n_hidden
        # количество временных шагов в каждом входном потоке
        self.seq_len = seq_len
        # количество скрытых слоев (всего получается n_hidden * n_layers LSTM блока.
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            # ожидаемое количество признаков во входном потоке
            input_size=n_features,
            # ожидаемое количество features в скрытом слое
            hidden_size=n_hidden,
            # количество рекуррентных слоев
            num_layers=n_layers,
            # прореживание выхода каждого рекуррентного слоя, кроме последнего
            dropout=0.5
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    # LSTM требует сброса состояния после каждого примера
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )

    # Получаем все последовательности и пропускаем через слой LSTM вместе за раз.
    # Берём вывод последнего временного шага и пропускаем его через линейный слой для получения прогноза.
    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = \
            lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred
