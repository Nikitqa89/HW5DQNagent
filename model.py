import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


# ======= Нейронная сеть для Q-Learning ========
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        # Список линейных слоёв: вход -> скрытые слои
        self.layers = nn.ModuleList()

        # Первый линейный слой: вход -> первый скрытый слой
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))

        # Промежуточные скрытые слои (если их несколько)
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        # Выходной слой: последний скрытый -> выход (Q-значения)
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # Прямой проход через все скрытые слои с функцией активации ReLU
        for layer in self.layers:
            x = F.relu(layer(x))

        # Выходной слой (Q-значения для каждого действия)
        x = self.output_layer(x)
        return x

    def save(self, file_path='model.pth'):
        # Сохраняем веса модели
        # Если указан только файл — сохраняем в ./model/
        if not os.path.isabs(file_path) and not os.path.dirname(file_path):
            folder = './model'
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, file_path)
        else:
            # Если указан путь — создаем директорию, если её нет
            folder = os.path.dirname(file_path)
            os.makedirs(folder, exist_ok=True)

        # Сохраняем параметры модели
        torch.save(self.state_dict(), file_path)


# ======= Тренировщик сети Q-Learning ========
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr  # Скорость обучения
        self.gamma = gamma  # Коэффициент дисконтирования
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # Функция потерь (среднеквадратичная ошибка)

    def train_step(self, state, action, reward, next_state, done):
        # Преобразование входных данных в тензоры
        state = torch.from_numpy(np.array(state)).float()
        next_state = torch.from_numpy(np.array(next_state)).float()
        action = torch.from_numpy(np.array(action)).long()
        reward = torch.from_numpy(np.array(reward)).float()

        # Если только один элемент, добавляем размерность (batch_size = 1)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Предсказание текущих Q-значений
        pred = self.model(state)

        # Создаём копию предсказания, чтобы изменить целевые значения
        target = pred.clone()

        # Обновляем Q-значения с использованием формулы Беллмана
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:  # Если эпизод не завершён
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Обновляем только значение Q для выбранного действия
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Шаг оптимизации
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)  # Вычисляем ошибку между target и prediction
        loss.backward()
        self.optimizer.step()