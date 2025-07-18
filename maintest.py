import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time

DIMENSIONS = 2
GLOBAL_BEST = 0
B_LO = -5
B_HI = 5
POPULATION = 100
V_MAX = 0.1
PERSONAL_C = 2.5
SOCIAL_C = 2.5
CONVERGENCE = 0.01
MAX_ITER = 1000

class Particle():
    def __init__(self, x, y, z, velocity):
        self.pos = [x, y]
        self.pos_z = z
        self.velocity = velocity
        self.best_pos = self.pos.copy()

class Swarm():
    def __init__(self, pop, v_max):
        self.particles = []
        self.best_pos = None
        self.best_pos_z = math.inf

        for _ in range(pop):
            x = np.random.uniform(B_LO, B_HI)
            y = np.random.uniform(B_LO, B_HI)
            z = cost_function(x, y)
            velocity = np.random.rand(2) * v_max
            particle = Particle(x, y, z, velocity)
            self.particles.append(particle)
            if self.best_pos is None or particle.pos_z < self.best_pos_z:
                self.best_pos = particle.pos.copy()
                self.best_pos_z = particle.pos_z

def cost_function(x, y, a=20, b=0.2, c=2*math.pi):
    term_1 = np.exp((-b * np.sqrt(0.5 * (x ** 2 + y ** 2))))
    term_2 = np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return -1 * a * term_1 - term_2 + a + np.exp(1)

def particle_swarm_optimization():
    x = np.linspace(B_LO, B_HI, 50)
    y = np.linspace(B_LO, B_HI, 50)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure("Particle Swarm Optimization")

    swarm = Swarm(POPULATION, V_MAX)
    inertia_weight = 0.5 + (np.random.rand() / 2)

    curr_iter = 0
    start_time = time.time()

    while curr_iter < MAX_ITER:
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ac = ax.contourf(X, Y, cost_function(X, Y), cmap='viridis')
        fig.colorbar(ac)

        for particle in swarm.particles:
            for i in range(DIMENSIONS):
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                personal = PERSONAL_C * r1 * (particle.best_pos[i] - particle.pos[i])
                social = SOCIAL_C * r2 * (swarm.best_pos[i] - particle.pos[i])
                new_velocity = inertia_weight * particle.velocity[i] + personal + social
                particle.velocity[i] = max(min(new_velocity, V_MAX), -V_MAX)

            ax.scatter(particle.pos[0], particle.pos[1], marker='*', c='r')
            ax.arrow(particle.pos[0], particle.pos[1], particle.velocity[0], particle.velocity[1],
                     head_width=0.1, head_length=0.1, color='k')

            particle.pos = [particle.pos[i] + particle.velocity[i] for i in range(DIMENSIONS)]
            particle.pos_z = cost_function(particle.pos[0], particle.pos[1])

            if particle.pos_z < cost_function(particle.best_pos[0], particle.best_pos[1]):
                particle.best_pos = particle.pos.copy()
                if particle.pos_z < swarm.best_pos_z:
                    swarm.best_pos = particle.pos.copy()
                    swarm.best_pos_z = particle.pos_z

            for i in range(DIMENSIONS):
                if particle.pos[i] > B_HI or particle.pos[i] < B_LO:
                    particle.pos[i] = np.random.uniform(B_LO, B_HI)
                    particle.pos_z = cost_function(particle.pos[0], particle.pos[1])

        plt.subplots_adjust(right=0.95)
        plt.pause(0.00001)

        if abs(swarm.best_pos_z - GLOBAL_BEST) < CONVERGENCE:
            end_time = time.time()
            duration = end_time - start_time
            print(f"The swarm has met convergence criteria after {curr_iter} iterations with {duration:.2f} seconds.")
            break

        curr_iter += 1

    plt.show()

if __name__ == "__main__":
    particle_swarm_optimization()
