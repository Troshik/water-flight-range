import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def euler_method(v0, alpha, k, m, g, dt):
    alpha_rad = np.radians(alpha)

    x, y = 0, 0
    vx, vy = v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)
    t = 0
    max_h = 0

    x_vals, y_vals = [], []

    while y >= 0 and m > 0:
        x_vals.append(x)
        y_vals.append(y)
        max_h = max(max_h, y)

        ax = -k / m * vx
        ay = -g - k / m * vy

        vx += ax * dt
        vy += ay * dt

        x += vx * dt
        y += vy * dt

        t += dt

    return x_vals, y_vals, round(x - vx * dt, 3), round(max_h, 3), t - dt


def ternary_search(v0, k, m, g, dt, low=0, high=90, epsilon=1e-3):
    while high - low > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        _, _, x1, _, _ = euler_method(v0, mid1, k, m, g, dt)
        _, _, x2, _, _ = euler_method(v0, mid2, k, m, g, dt)

        if x1 < x2:
            low = mid1
        else:
            high = mid2

    return (low + high) / 2


# Параметры задачи
v0 = 10  # Начальная скорость (м/с)
alpha = 44  # Угол наклона (градусы)
k = 0.0002213  # Коэффициент сопротивления
m = 0.001  # Масса (кг)
g = 9.81  # Ускорение свободного падения (м/с^2)
dt = 0.01  # Шаг времени (с)


opt_angle = ternary_search(v0, k, m, g, dt)
x_vals, y_vals, x, _, _ = euler_method(v0, opt_angle, k, m, g, dt)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals)
plt.title(f"Траектория полёта струи воды\nОптимальный угол = {round(opt_angle, 4)}")
plt.xlabel(f"Расстояние (м)\nДальность = {x}")
plt.ylabel("Высота (м)")
plt.grid()
plt.axis('equal')
plt.show()


def x_table():
    results = []

    for angle in range(0, 91, 5):
        x_vals, y_vals, x, _, _ = euler_method(v0, angle, k, m, g, dt)
        results.append({"Угол": angle, "Дальность полета": x})

    results_df = pd.DataFrame(results)
    print(results_df)


def x_show():
    data = []
    for i in range(1, 91):
        _, _, x, _, _ = euler_method(v0, i+0.5, k, m, g, dt)
        data.append([i+0.5, x])
        _, _, x, _, _ = euler_method(v0, i, k, m, g, dt)
        data.append([i, x])

    headers = ["Угол (градусы)", "Дальность полета (м)"]
    df = pd.DataFrame(data, columns=headers)
    plt.figure(figsize=(10, 6))
    plt.plot(df[headers[0]], df[headers[1]], marker='o', markersize=1, linestyle='-', color='b', label='Дальность полета')
    plt.title("Зависимость дальности полета от угла наклона")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, 91, 5))
    plt.legend()
    plt.show()

def y_show():
    data = []
    for angle in range(0, 91, 1):
        _, _, _, max_height, _ = euler_method(v0, angle, k, m, g, dt)
        data.append([angle, max_height])

    headers = ["Угол (градусы)", "Максимальная выссота (м)"]
    df = pd.DataFrame(data, columns=headers)
    plt.figure(figsize=(10, 6))
    plt.plot(df[headers[0]], df[headers[1]], marker='o', markersize=1, linestyle='-', color='r', label='Максимальная высота')
    plt.title("Зависимость максимальной высоты от угла наклона")
    plt.xlabel(headers[0])
    plt.ylabel(headers[1])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(range(0, 91, 5))
    plt.legend()
    plt.show()


def generate_flight_time_data():
    times = []
    angles = list(range(0, 91, 1))
    for angle in angles:
        _, _, _, _, t = euler_method(v0, angle, k, m, g, dt)
        times.append(t)

    plt.figure(figsize=(10, 6))
    plt.plot(angles, times, marker='o', markersize=1,  color='g', linestyle='-', label="Время полета")
    plt.title("Зависимость времени полета от угла наклона")
    plt.xlabel("Угол (градусы)")
    plt.ylabel("Время полета (с)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.show()

