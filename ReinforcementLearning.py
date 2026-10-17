import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# =========================================================
# 1. ПАРАМЕТРЫ СРЕДЫ
# =========================================================
GRID = 5
START = (0, 0)
GOAL  = (4, 4)
OBSTACLES = {(1, 1), (2, 2), (3, 1), (1, 3)}   # стеллажи

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]    # вверх, вниз, влево, вправо
MAX_STEPS = 100

def step(state, action):
    """Возвращает (next_state, reward, done)."""
    dx, dy = ACTIONS[action]
    x, y = state[0] + dx, state[1] + dy

    # упёрлись в границу — стоим, штраф
    if not (0 <= x < GRID and 0 <= y < GRID):
        return state, -5, False
    # упёрлись в препятствие — стоим, штраф
    if (x, y) in OBSTACLES:
        return state, -5, False

    new_state = (x, y)
    if new_state == GOAL:
        return new_state, +100, True
    return new_state, -1, False

# =========================================================
# 2. ГИПЕРПАРАМЕТРЫ Q-LEARNING
# =========================================================
ALPHA         = 0.1
GAMMA         = 0.95
EPSILON       = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.995
EPISODES      = 1000

# Q-таблица: для каждого состояния — вектор из 4 значений
Q = defaultdict(lambda: np.zeros(4))

rewards_per_episode = []

# =========================================================
# 3. ЦИКЛ ОБУЧЕНИЯ
# =========================================================
for episode in range(EPISODES):
    state = START
    total_reward = 0

    for t in range(MAX_STEPS):
        # ε-greedy выбор действия
        if random.random() < EPSILON:
            action = random.randint(0, 3)       # исследование
        else:
            action = int(np.argmax(Q[state]))               # использование

        next_state, reward, done = step(state, action)

        # --- Обновление Q по Беллману ---
        best_next = np.max(Q[next_state])
        td_target = reward + GAMMA * best_next
        Q[state][action] += ALPHA * (td_target - Q[state][action])

        state = next_state
        total_reward += reward
        if done:
            break

    # уменьшаем ε — меньше случайности со временем
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 100 == 0:
        avg = np.mean(rewards_per_episode[-100:])
        print(f"Эпизод {episode+1:4d} | средняя награда за 100 эп.: {avg:8.2f} | ε={EPSILON:.3f}")

# =========================================================
# 4. ГРАФИК ОБУЧЕНИЯ
# =========================================================
def moving_average(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')

plt.figure(figsize=(10, 4))
plt.plot(rewards_per_episode, alpha=0.3, label='награда за эпизод')
plt.plot(moving_average(rewards_per_episode, 100),
         color='red', linewidth=2, label='скользящее среднее (100)')
plt.xlabel('Эпизод'); plt.ylabel('Суммарная награда')
plt.title('Кривая обучения Q-learning')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# =========================================================
# 5. ВИЗУАЛИЗАЦИЯ ВЫУЧЕННОЙ ПОЛИТИКИ
# =========================================================
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-0.5, GRID - 0.5)
ax.set_ylim(GRID - 0.5, -0.5)       # инвертируем ось Y — как матрица
ax.set_xticks(range(GRID)); ax.set_yticks(range(GRID))
ax.grid(True)

# препятствия
for (r, c) in OBSTACLES:
    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
# цель
ax.add_patch(plt.Rectangle((GOAL[1] - 0.5, GOAL[0] - 0.5), 1, 1, color='green', alpha=0.6))
# старт
ax.add_patch(plt.Rectangle((START[1] - 0.5, START[0] - 0.5), 1, 1, color='blue', alpha=0.6))

# стрелки = лучший выученный ход из каждой клетки
arrow = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}
for r in range(GRID):
    for c in range(GRID):
        if (r, c) in OBSTACLES:
            continue
        a = int(np.argmax(Q[(r, c)]))
        dx, dy = arrow[a]
        ax.arrow(c, r, dx, dy, head_width=0.12, head_length=0.12,
                 fc='red', ec='red', length_includes_head=True)

plt.title('Выученная политика (стрелки — лучшее действие)')
plt.show()

# =========================================================
# 6. "ЖАДНЫЙ" ПРОХОД ОТ СТАРТА ДО ЦЕЛИ
# =========================================================
path = [START]
s = START
for _ in range(MAX_STEPS):
    if s == GOAL:
        break
    a = int(np.argmax(Q[s]))
    ns, _, done = step(s, a)
    if ns == s:                # упёрлись — тупик
        print("Агент застрял в", s); break
    path.append(ns)
    s = ns
    if done:
        break

print("\nОптимальный путь:", path)
print("Длина пути:", len(path) - 1, "шагов")