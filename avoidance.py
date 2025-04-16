import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv("new_training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# sort and group by timestamp
df = df.sort_values("timestamp")

# two satellite tracks based on sat_id
sat1_df = df[df["sat_id"] == "sat1"]
sat2_df = df[df["sat_id"] == "sat2"]

# Extract positions for both satellites
positions1 = sat1_df[["position_x", "position_y", "position_z"]].values
positions2 = sat2_df[["position_x", "position_y", "position_z"]].values

num_threats = 14
threats = np.random.uniform(low=-10000, high=10000, size=(num_threats, 3))  # random 

#  calculate distance between two positions
def calculate_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

# apply a maneuver 
def apply_maneuver(pos, threat_pos, maneuver_distance=100):
    direction = pos - threat_pos
    norm = np.linalg.norm(direction)
    if norm > 0:
        # normalize the direction vector
        direction_normalized = direction / norm
        # move the satellite away from the object by maneuver distance
        new_pos = pos + direction_normalized * maneuver_distance
        return new_pos
    return pos

# figure and 3D axis for animation
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Satellite Orbits with Collision Points and Threats")

# earth
ax.scatter(0, 0, 0, color="blue", s=200, label="Earth")

# scatter plot for collision points (will be updated in the animation)
collision_scatter = ax.scatter([], [], [], color="red", s=100, label="Collision Points")
sat1_dot, = ax.plot([], [], [], 'bo', label="SAT_1")
sat2_dot, = ax.plot([], [], [], 'go', label="SAT_2")
sat1_line, = ax.plot([], [], [], label="SAT_1 Orbit")
sat2_line, = ax.plot([], [], [], label="SAT_2 Orbit")

# random threats
threat_scatter = ax.scatter(threats[:, 0], threats[:, 1], threats[:, 2], color="orange", s=100, label="Threats")

# update the animation frame
def update(frame):
    # update the positions for both satellites
    sat1_pos = positions1[:frame]
    sat2_pos = positions2[:frame]

    sat1_line.set_data(sat1_pos[:, 0], sat1_pos[:, 1])
    sat1_line.set_3d_properties(sat1_pos[:, 2])

    sat2_line.set_data(sat2_pos[:, 0], sat2_pos[:, 1])
    sat2_line.set_3d_properties(sat2_pos[:, 2])

    # update the satellite dots after the first frame
    if frame > 0:
        sat1_dot.set_data([sat1_pos[frame-1, 0]], [sat1_pos[frame-1, 1]])
        sat1_dot.set_3d_properties([sat1_pos[frame-1, 2]])

        sat2_dot.set_data([sat2_pos[frame-1, 0]], [sat2_pos[frame-1, 1]])
        sat2_dot.set_3d_properties([sat2_pos[frame-1, 2]])


    collision_points = []
    for i in range(frame):
        # distance between sat1 and debris
        dist_sat1_threat = [calculate_distance(sat1_pos[i], threat) for threat in threats]
        
        # distance between sat2 and debris
        dist_sat2_threat = [calculate_distance(sat2_pos[i], threat) for threat in threats]

        # exclude collision detection between sat1 with sat2
        if min(dist_sat1_threat) < 1.0:
            print(f"Collision detected for sat_1 at frame {frame}!")
            sat1_pos[i] = apply_maneuver(sat1_pos[i], threats[np.argmin(dist_sat1_threat)])

        if min(dist_sat2_threat) < 1.0:
            print(f"Collision detected with threat for sat_2 at frame {frame}!")
            sat2_pos[i] = apply_maneuver(sat2_pos[i], threats[np.argmin(dist_sat2_threat)])

        # add positions to collision_points 
        collision_points.append([sat1_pos[i][0], sat1_pos[i][1], sat1_pos[i][2]])

    # clear previous points before adding new ones
    collision_scatter._offsets3d = ([], [], [])  # clears previous points
    if collision_points:
        collision_scatter._offsets3d = (np.array(collision_points).T[0], np.array(collision_points).T[1], np.array(collision_points).T[2])


    ax.set_xlim([np.min(positions1[:, 0]), np.max(positions2[:, 0])])
    ax.set_ylim([np.min(positions1[:, 1]), np.max(positions2[:, 1])])
    ax.set_zlim([np.min(positions1[:, 2]), np.max(positions2[:, 2])])

    return sat1_dot, sat2_dot, sat1_line, sat2_line, collision_scatter

frames = len(positions1)

ani = FuncAnimation(
    fig, update, frames=frames, interval=100, blit=False
)


plt.tight_layout()
plt.show()


ani.save("avoid_collision_with_threats.gif", writer='pillow')

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

df = pd.read_csv("new_training_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp")

sat1_df = df[df["sat_id"] == "sat1"]
sat2_df = df[df["sat_id"] == "sat2"]

positions1 = sat1_df[["position_x", "position_y", "position_z"]].values
positions2 = sat2_df[["position_x", "position_y", "position_z"]].values

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Satellite Orbits with Collision Points")
ax.scatter(0, 0, 0, color="blue", s=200, label="Earth")

collision_scatter = ax.scatter([], [], [], color="red", s=100, label="Collision Points")
sat1_dot, = ax.plot([], [], [], 'bo', label="SAT_1")
sat2_dot, = ax.plot([], [], [], 'go', label="SAT_2")
sat1_line, = ax.plot([], [], [], label="SAT_1 Orbit")
sat2_line, = ax.plot([], [], [], label="SAT_2 Orbit")

def calculate_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)

def update(frame):
    sat1_pos = positions1[:frame]
    sat2_pos = positions2[:frame]

    sat1_line.set_data(sat1_pos[:, 0], sat1_pos[:, 1])
    sat1_line.set_3d_properties(sat1_pos[:, 2])

    sat2_line.set_data(sat2_pos[:, 0], sat2_pos[:, 1])
    sat2_line.set_3d_properties(sat2_pos[:, 2])

    if frame > 0:
        sat1_dot.set_data([sat1_pos[frame-1, 0]], [sat1_pos[frame-1, 1]])
        sat1_dot.set_3d_properties([sat1_pos[frame-1, 2]])

        sat2_dot.set_data([sat2_pos[frame-1, 0]], [sat2_pos[frame-1, 1]])
        sat2_dot.set_3d_properties([sat2_pos[frame-1, 2]])

    collision_points = []
    for i in range(frame):
        dist = calculate_distance(sat1_pos[i], sat2_pos[i])
        if dist < 1.0: # 1 km
            collision_points.append([sat1_pos[i][0], sat1_pos[i][1], sat1_pos[i][2]])

    collision_scatter._offsets3d = ([], [], [])
    if collision_points:
        collision_scatter._offsets3d = (np.array(collision_points).T[0], np.array(collision_points).T[1], np.array(collision_points).T[2])

    ax.set_xlim([np.min(positions1[:, 0]), np.max(positions2[:, 0])])
    ax.set_ylim([np.min(positions1[:, 1]), np.max(positions2[:, 1])])
    ax.set_zlim([np.min(positions1[:, 2]), np.max(positions2[:, 2])])

    return sat1_dot, sat2_dot, sat1_line, sat2_line, collision_scatter

frames = len(positions1)

ani = FuncAnimation(
    fig, update, frames=frames, interval=100, blit=False
)

plt.tight_layout()
plt.show()

ani.save("avoid_collision.gif", writer='pillow')

'''