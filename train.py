import os
import json
import random
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from dqn_agent import Agent
from SnakeEnv import SnakeEnv

# === Настройки обучения ===
NUM_GAMES = 1000             # Общее количество игр (эпизодов) для обучения
PRINT_EVERY = 100            # Частота вывода информации в консоль
RENDER_EVERY = 100           # Частота отрисовки среды (для визуального отслеживания)

# === Создание уникальной папки для сохранения результатов текущего запуска ===
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_folder = os.path.join("runs", f"run_{timestamp}")
os.makedirs(run_folder, exist_ok=True)

# === Гиперпараметры и параметры наград ===
params = {
    "num_games": NUM_GAMES,
    "lr": 0.001,                   # Скорость обучения
    "gamma": 0.99,                 # Коэффициент дисконтирования (будущие награды)
    "reward_food": 10,             # Награда за съедение еды
    "reward_death": -10,           # Штраф за смерть
    "reward_idle": -0.1,          # Штраф за бездействие

    # Параметры агента DQN
    "max_memory": 100_000,         # Максимальный размер памяти опыта
    "batch_size": 1000,            # Размер батча для обучения из памяти
    "hidden_sizes": [256, 128],    # Структура скрытых слоёв нейросети

    # === ε-жадная стратегия ===
    "start_epsilon": 95,           # Начальное значение ε (вероятность случайного действия)
    "min_epsilon": 0,              # Минимальное значение ε
    "epsilon_decay_rate": 0.995,   # Скорость уменьшения ε после каждой игры

    "seed": 42                     # Фиксированное значение seed для воспроизводимости
}

# === Функция фиксации сидов для воспроизводимых результатов ===
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(params["seed"])

# === Сохраняем параметры эксперимента в JSON-файл ===
with open(os.path.join(run_folder, "params.json"), "w") as f:
    json.dump(params, f, indent=4)

# === Функция для построения графика результатов обучения ===
def plot(scores, save_path=None, ma_window=50):
    plt.figure(figsize=(10, 5))
    plt.title('Результаты тренировки')
    plt.xlabel('Кол-во игр')
    plt.ylabel('Очки')

    # Сырые данные — очки за каждую игру
    plt.plot(scores, label='Очки за игру', color='blue', linestyle='-')

    # Добавляем скользящее среднее для сглаживания
    if len(scores) >= ma_window:
        moving_avg = [np.mean(scores[max(0, i - ma_window):i + 1]) for i in range(len(scores))]
        plt.plot(moving_avg, label=f'Скользящее среднее ({ma_window})',
                 color='orange', linestyle='--')

    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()

# === Основной цикл обучения ===
def train():
    scores = []           # Список очков за каждую игру
    total_score = 0       # Общая сумма очков
    record = 0            # Лучший результат

    # Создаем DQN-агента с заданными параметрами
    agent = Agent(
        lr=params["lr"],
        gamma=params["gamma"],
        max_memory=params["max_memory"],
        batch_size=params["batch_size"],
        hidden_sizes=params["hidden_sizes"],
        start_epsilon=params["start_epsilon"],
        min_epsilon=params["min_epsilon"],
        epsilon_decay_rate=params["epsilon_decay_rate"]
    )

    # Вспомогательная функция создания новой среды
    def create_env(render):
        return SnakeEnv(
            render=render,
            rewards={
                "food": params["reward_food"],
                "death": params["reward_death"],
                "idle": params["reward_idle"]
            }
        )

    # Инициализируем первую среду
    game = create_env(render=False)

    # Основной цикл по эпизодам
    while agent.n_games < NUM_GAMES:
        # Получаем текущее состояние среды
        state_old = agent.get_state(game)

        # Получаем действие от агента на основе ε-жадной стратегии
        final_move = agent.get_action(state_old)

        # Выполняем шаг игры
        reward, done, score = game.play_step(final_move)

        # Получаем новое состояние
        state_new = agent.get_state(game)

        # Краткосрочное обучение на текущем переходе
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Сохраняем опыт в память
        agent.remember(state_old, final_move, reward, state_new, done)

        # Если эпизод завершён
        if done:
            agent.n_games += 1

            # Обучение на случайной выборке из памяти
            agent.train_long_memory()

            # Сохраняем статистику
            scores.append(score)
            total_score += score

            # Сохраняем модель, если достигнут новый рекорд
            if score > record:
                record = score
                agent.model.save(os.path.join(run_folder, "best_model.pth"))

            # Выводим информацию каждые PRINT_EVERY игр
            if agent.n_games % PRINT_EVERY == 0:
                avg_score = total_score / len(scores)
                print(f'Игра: {agent.n_games}, Очки: {score}, Рекорд: {record}, Среднее: {avg_score:.2f}')

            # Пересоздаем среду, с отрисовкой при необходимости
            render = agent.n_games % RENDER_EVERY == 0
            game = create_env(render=render)

    # Построение графика обучения
    plot(scores, save_path=os.path.join(run_folder, "training_plot.png"))

    # Сводная статистика обучения
    avg_last_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    summary = {
        "Best score": record,
        "Average score (last 100)": round(avg_last_100, 2),
        "Total games": agent.n_games
    }
    summary.update(params)

    # Сохраняем сводную информацию в текстовый файл
    with open(os.path.join(run_folder, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print(f"Обучение завершено. Папка с результатами: {run_folder}")

# === Точка входа — запуск обучения ===
if __name__ == '__main__':
    train()