import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer

class Agent:
    def __init__(self, lr=0.001, gamma=0.9, max_memory=100_000, batch_size=1000,
                 hidden_sizes=[256, 128],  # ⬅ Теперь список слоёв!
                 start_epsilon=80, min_epsilon=5, epsilon_decay_rate=0.995):
        # Количество сыгранных игр
        self.n_games = 0

        # === Параметры ε-жадной стратегии ===
        self.start_epsilon = start_epsilon        # Начальное значение ε
        self.epsilon = start_epsilon              # Текущее значение ε
        self.min_epsilon = min_epsilon            # Минимально допустимое значение ε
        self.epsilon_decay_rate = epsilon_decay_rate  # Скорость экспоненциального уменьшения ε

        # === Гиперпараметры обучения ===
        self.gamma = gamma                        # Коэффициент дисконтирования
        self.batch_size = batch_size              # Размер мини-батча

        # === Память для хранения опыта агента ===
        self.memory = deque(maxlen=max_memory)    # Очередь с ограниченным размером

        # === Модель и тренер ===
        self.model = Linear_QNet(11, hidden_sizes, 3)  # ⬅ Передаём список слоёв
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma)

    def get_state(self, game):
        # Получить текущее состояние из среды
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        # Сохранить переход в память
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Обучение на случайной выборке из памяти
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Обучение на одном переходе (для быстрой адаптации)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # === Обновление ε с экспоненциальным спадом ===
        self.epsilon = max(self.min_epsilon, self.start_epsilon * (self.epsilon_decay_rate ** self.n_games))

        # ε-жадная стратегия: случайное действие с вероятностью ε
        final_move = [0, 0, 0]

        if random.randint(0, 100) < self.epsilon:
            # Случайное действие
            move = random.randint(0, 2)
        else:
            # Предсказание действия нейросетью
            state0 = torch.from_numpy(np.array(state)).float()
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        final_move[move] = 1
        return final_move