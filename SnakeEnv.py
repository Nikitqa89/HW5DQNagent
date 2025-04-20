import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

# Возможные направления движения змейки
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Точка на поле
Point = namedtuple('Point', 'x, y')

# Цвета в формате RGB
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Размер одного блока и скорость игры
BLOCK_SIZE = 20
SPEED = 20

class SnakeEnv:
    def __init__(self, w=640, h=480, render=False, rewards=None):
        self.w = w
        self.h = h
        self.render = render

        # Награды за разные события
        self.reward_food = rewards.get("food", 10) if rewards else 10
        self.reward_death = rewards.get("death", -10) if rewards else -10
        self.reward_idle = rewards.get("idle", 0) if rewards else 0

        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('arial', 25)

        self.reset()

    def reset(self):
        # Сброс игры: начальное направление и положение змейки
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0  # для ограничения по времени

    def _place_food(self):
        # Случайное размещение еды на поле, не внутри змейки
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1

        # Обработка выхода из игры
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.render:
                    pygame.display.quit()
                    self.render = False
                    self.display = None
                return self.reward_death, True, self.score

        self._move(action)           # Обновляем направление и позицию головы
        self.snake.insert(0, self.head)  # Добавляем новую голову

        reward = self.reward_idle
        game_over = False

        # Проверка на проигрыш: столкновение или слишком долго без еды
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = self.reward_death
            return reward, game_over, self.score

        # Если съели еду
        if self.head == self.food:
            self.score += 1
            reward = self.reward_food
            self._place_food()
        else:
            self.snake.pop()  # Удаляем хвост, если не съели еду

        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Столкновение со стенами
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # Столкновение с телом
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        # Отрисовка всех объектов на экране
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = self.font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # Интерпретируем действие: [1, 0, 0] — прямо, [0, 1, 0] — направо, [0, 0, 1] — налево
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # продолжаем в том же направлении
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # поворот по часовой стрелке
        else:
            new_dir = clock_wise[(idx - 1) % 4]  # поворот против часовой стрелки

        self.direction = new_dir

        # Обновляем координаты головы в зависимости от нового направления
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):
        # Возвращает состояние окружающей среды (11 параметров)
        head = self.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Опасность в направлении движения
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Опасность справа
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Опасность слева
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Направление движения (one-hot)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Расположение еды относительно головы
            self.food.x < head.x,  # еда слева
            self.food.x > head.x,  # еда справа
            self.food.y < head.y,  # еда сверху
            self.food.y > head.y   # еда снизу
        ]

        return np.array(state, dtype=int)