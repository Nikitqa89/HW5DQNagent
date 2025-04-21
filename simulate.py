import os
import json
import torch
import numpy as np
from SnakeEnv import SnakeEnv
from model import Linear_QNet

# === Ручной ввод путей===
MODEL_PATH = "runs/run_20250421-153841/best_model.pth"
PARAMS_PATH = "runs/run_20250421-153841/params.json"

# === Выбор действия ===
def get_action(model, state):
    state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))
    with torch.no_grad():
        prediction = model(state_tensor)
    move = torch.argmax(prediction).item()
    final_move = [0, 0, 0]
    final_move[move] = 1
    return final_move

# === Основная функция симуляции ===
def simulate(num_episodes=3, render=True):
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")
    if not os.path.isfile(PARAMS_PATH):
        raise FileNotFoundError(f"Файл параметров не найден: {PARAMS_PATH}")

    with open(PARAMS_PATH, "r") as f:
        params = json.load(f)

    input_size = 11
    output_size = 3
    hidden_sizes = params.get("hidden_sizes", [256, 128])
    rewards = {
        "food": params.get("reward_food", 10),
        "death": params.get("reward_death", -10),
        "idle": params.get("reward_idle", 0)
    }

    model = Linear_QNet(input_size, hidden_sizes, output_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    env = SnakeEnv(render=render, rewards=rewards)

    for episode in range(1, num_episodes + 1):
        print(f"\nСимуляция #{episode}")
        env.reset()
        done = False
        total_score = 0

        while not done:
            state = env.get_state()
            action = get_action(model, state)
            reward, done, score = env.play_step(action)
            total_score = score

        print(f"Очки: {total_score}")

if __name__ == '__main__':
    simulate()