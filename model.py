import torch
from torch import nn, optim

# Класс создания модели, наследующий torch.nn.Module:
class CoronaVirusPredictor(nn.Module):
    # В конструкторе инициализируем поля и создаем слои модели
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaVirusPredictor, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    # Здесь используется LSTM (long short-term memory, долгая краткосрочная память —
    # разновидность архитектуры рекуррентных нейронных сетей
    # Она требует сброса состояния после каждого примера
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
