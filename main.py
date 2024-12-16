import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def euler_method(v0, alpha, k, m, g, dt):
    alpha_rad = np.radians(alpha)

    x, y = 0, 0
    vx, vy = v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)
    t = 0

    x_vals, y_vals = [x], [y]

    while y >= 0 and m > 0:

        ax = -k / m * vx
        ay = -g - k / m * vy

        vx += ax * dt
        vy += ay * dt

        x += vx * dt
        y += vy * dt

        x_vals.append(x)
        y_vals.append(y)

        t += dt

    return x_vals, y_vals, round(x, 3)


def ternary_search(v0, k, m, g, dt, low=0, high=90, epsilon=1e-3):
    while high - low > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        _, _, x1 = euler_method(v0, mid1, k, m, g, dt)
        _, _, x2 = euler_method(v0, mid2, k, m, g, dt)

        if x1 < x2:
            low = mid1
        else:
            high = mid2

    opt_angle = (low + high) / 2
    return opt_angle


# Параметры задачи
v0 = 10  # Начальная скорость (м/с)
alpha = 44  # Угол наклона (градусы)
k = 0.0002213  # Коэффициент сопротивления
m = 0.002  # Масса (кг)
g = 9.81  # Ускорение свободного падения (м/с^2)
dt = 0.01  # Шаг времени (с)


results = []

for angle in range(0, 91, 5):
    x_vals, y_vals, x = euler_method(v0, angle, k, m, g, dt)
    results.append({"Угол": angle, "Дальность полета": x})

results_df = pd.DataFrame(results)
print(results_df)


opt_angle = ternary_search(v0, k, m, g, dt)
x_vals, y_vals, x = euler_method(v0, opt_angle, k, m, g, dt)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals)
plt.title(f"Траектория полёта струи воды\nОптимальный угол = {round(opt_angle, 4)}")
plt.xlabel(f"Расстояние (м)\nДальность = {x}")
plt.ylabel("Высота (м)")
plt.grid()
plt.axis('equal')
plt.show()
