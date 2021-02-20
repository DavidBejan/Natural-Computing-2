import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import sys

rand.seed(1)
GLOBAL_MINIMUM = 0

def init_swarm_pos(particles, lower_limit, upper_limit):
    return lower_limit + rand.rand(particles) * (upper_limit-lower_limit)

f = lambda x: x**2


def plot_path(particles, path, margin=10, title="", is_save=True, filename="img/path.png"):
    max_value = path.max()
    min_value = path.min()
    x_bound = -min_value if abs(min_value) > max_value else max_value
    x_vals = np.linspace(-x_bound-margin, x_bound+margin, num=1000)
    plt.plot(x_vals, f(x_vals), label="f(x)")
    for i in range(particles):
        plt.plot(path[:, i], f(path[:, i]), 'o')
        plt.plot(path[:, i], f(path[:, i]))
        plt.plot(path[0, i], f(path[0, i]), 'o', label="Start", c="red")
        plt.plot(path[-1, i], f(path[-1, i]), 'o', label="End", c="blue")
    plt.ylabel("Y")
    plt.xlabel("X")
    plt.title(title)
    plt.legend()
    if is_save:
        plt.savefig(filename)
    plt.show()


def pso(particles, omega, iterations=10000, epsilon=5, a1=1.5, a2=1.5, lower_limit=-100, upper_limit=100):
    swarm_pos = init_swarm_pos(particles, lower_limit, upper_limit)
    path = [swarm_pos.copy()] 
    velocities = [10]#np.ones((particles))
    local_best = upper_limit * np.ones((particles))
    global_best = upper_limit
    min_found = False
    for _ in range(iterations):
        for i in range(particles):
            r1, r2 = rand.uniform(0.001, 1, size=2)
            velocities[i] = omega*velocities[i] + a1*r1*(local_best[i]-swarm_pos[i]) + a2*r2*(global_best-swarm_pos[i])
            swarm_pos[i] += velocities[i]

            if f(swarm_pos[i]) < f(local_best[i]):
                local_best[i] = swarm_pos[i]
            if f(swarm_pos[i]) < f(global_best):
                global_best = swarm_pos[i]

            min_found = abs(swarm_pos[i]) < GLOBAL_MINIMUM+epsilon

        path.append(swarm_pos.copy())
        if min_found:
            break

    return path

def main():
    particles = 1
    omegas = [0, 0.1, 0.5, 0.75]
    for omega in omegas:
        path = pso(particles, omega)
        plot_path(particles, np.array(path), title="Path for omega="+str(omega)+" ("+str(len(path)-1)+" epochs)", 
                        filename="img/omega"+str(omega).replace('.','_')+".png")


if __name__ == "__main__":
    main()
