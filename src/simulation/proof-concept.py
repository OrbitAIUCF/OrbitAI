import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# some constants
G = 6.64730e-11
earthM = 5.972e24
earthR = 6.371e6
alt = 400e3
r = earthR + alt


# initial conditions
v0 = np.sqrt(G * earthM / r) # m/s
theta0 = 0 # initial angle
omega = v0 / r # angular velocity

# time stuff
dt = 10
t_max = 90 * 60
t = np.arange(0, t_max, dt)

# satellites position over time 
theta = theta0 + omega * t
x = r * np.cos(theta)
y = r * np.sin(theta)

# stuff for the plot
fig, ax = plt.subplots()
ax.set_xlim(-1.5 * r, 1.5 * r)
ax.set_ylim(-1.5 * r, 1.5 * r)
ax.set_aspect('equal')

# add some pretty stars
num = 150
starX = np.random.uniform(-1.5 * r, 1.5 * r, num)
starY = np.random.uniform(-1.5 * r, 1.5 * r, num)
ax.scatter(starX, starY, color = "white", s = 2)

earth = plt.Circle((0,0), earthR * .75, color = '#100b5c')
ax.add_patch(earth)
ax.text(0, 0, "earth", fontsize = 10, color = "#07e015", ha = "center", va = "center")

# initializes our satellite
satellite, = ax.plot([], [], 'ro', markersize = 5)
sat_name = ax.text(0, 0, "satellite", fontsize = 10, color = "red")

# animation stuff
def animate(i):
    satellite.set_data([x[i]], [y[i]])
    sat_name.set_position((x[i] + 2e5, y[i] + 2e5))
    return satellite, sat_name
animation = FuncAnimation(fig, animate, frames = len(t), interval = 50, blit = True)


# some stuff to make the plot look a little nicer
ax.set_facecolor("xkcd:black")
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
plt.title("proof of concept simulation")

plt.show()
