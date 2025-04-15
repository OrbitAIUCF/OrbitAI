#Originally Jacob's File, Edited to Include Velocity X,Y,Z in CSV Output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import random

# =============================================================================
# Basic Simulation Settings
# =============================================================================
mu = 398600.4418  # Earth's gravitational parameter in km^3/s^2
Re = 6378.0       # Earth's radius in km
duration = 8100   # Total duration in seconds (8100 sec = 270 intervals of 30 sec)
step = 30         # Time step in seconds
num_steps = duration // step  # 270 time steps
start_time = datetime.now(timezone.utc)
timestamps = [start_time + timedelta(seconds=i * step) for i in range(num_steps)]

# =============================================================================
# Function: Generate orbit from classical elements
# =============================================================================
def generate_orbit_from_elements(sat_id, a, e, inc, raan, argp, M0, timestamps):
    """
    Generate a time-series of ECI positions for one satellite given its orbital elements.
    a   : semimajor axis (km)
    e   : eccentricity
    inc : inclination (radians)
    raan: right ascension (radians)
    argp: argument of perigee (radians)
    M0  : initial mean anomaly (radians)
    """
    n = np.sqrt(mu / a**3)  # mean motion in rad/s
    data = []
    for t in timestamps:
        t_elapsed = (t - timestamps[0]).total_seconds()
        M = (M0 + n * t_elapsed) % (2*np.pi)
        # For nearly circular orbits, approximate true anomaly nu ~ M.
        # (We assume e is very small here.)
        nu = M
        r_val = a * (1 - e**2) / (1 + e * np.cos(nu))
        # Position in the perifocal frame:
        r_pf = np.array([r_val * np.cos(nu), r_val * np.sin(nu), 0])
        # Rotation matrices with 3-1-3 rotation:
        R3_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan),  np.cos(raan), 0],
            [0, 0, 1]
        ])
        R1_inc = np.array([
            [1, 0, 0],
            [0, np.cos(inc), -np.sin(inc)],
            [0, np.sin(inc),  np.cos(inc)]
        ])
        R3_argp = np.array([
            [np.cos(argp), -np.sin(argp), 0],
            [np.sin(argp),  np.cos(argp), 0],
            [0, 0, 1]
        ])
        # Combined transformation matrix from perifocal to ECI:
        Q = R3_raan.dot(R1_inc).dot(R3_argp)
        pos = Q.dot(r_pf)
        data.append({
            "timestamp": t,
            "sat_id": sat_id,
            "position_x": pos[0],
            "position_y": pos[1],
            "position_z": pos[2]
        })
    return data

# =============================================================================
# Generate orbits for SAT_1, SAT_2, SAT_3 as random circular LEOs
# =============================================================================
def random_circular(sat_id):
    altitude = random.uniform(200, 1200)  # altitude in km
    a = Re + altitude                   # semimajor axis in km
    e = 0.0                             # circular orbit
    inc = np.deg2rad(random.uniform(0, 90))
    raan = np.deg2rad(random.uniform(0, 360))
    argp = np.deg2rad(random.uniform(0, 360))
    M0 = np.deg2rad(random.uniform(0, 360))
    return generate_orbit_from_elements(sat_id, a, e, inc, raan, argp, M0, timestamps)

sat1_data = random_circular("SAT_1")
sat2_data = random_circular("SAT_2")
sat3_data = random_circular("SAT_3")

# =============================================================================
# Generate orbits for SAT_4 and SAT_5 with an intersection at an intermediate timestamp.
# We force both to have the same state at a chosen pivot time without making the orbits coplanar.
# =============================================================================
# Common parameters for the shape:
altitude_45 = random.uniform(200, 1200)  # km altitude
a_common = Re + altitude_45
e_common = 0.01  # small eccentricity

# Choose a pivot index (not the first or last point)
pivot_idx = random.randint(1, num_steps - 2)
pivot_time = timestamps[pivot_idx]
delta_t = pivot_idx * step  # time offset (sec) from simulation start

# Compute the mean motion for the common orbit
n_common = np.sqrt(mu / a_common**3)

# To force an intersection at pivot time, choose M0 such that:
# M(pivot_time) = M0 + n_common * delta_t = 0 (mod 2π)
M0_intersection = (- n_common * delta_t) % (2*np.pi)

# For SAT_4, choose one set of orientation elements (equatorial)
inc4 = 0.0
raan4 = 0.0
argp4 = 0.0
M04 = M0_intersection  # so that at pivot time the true anomaly is 0; position in perifocal frame is [r, 0, 0]
sat4_data = generate_orbit_from_elements("SAT_4", a_common, e_common, inc4, raan4, argp4, M04, timestamps)

# For SAT_5, choose a different set for a non-coplanar orbit:
inc5 = np.deg2rad(30)          # 30° inclination
raan5 = np.deg2rad(180)        # 180° RAAN
argp5 = np.deg2rad(180)        # 180° argument of perigee
M05 = M0_intersection         # same initial condition so they intersect at pivot time
sat5_data = generate_orbit_from_elements("SAT_5", a_common, e_common, inc5, raan5, argp5, M05, timestamps)

# =============================================================================
# Combine all satellite data into one DataFrame
# =============================================================================
all_data = sat1_data + sat2_data + sat3_data + sat4_data + sat5_data
df = pd.DataFrame(all_data)

# Step 1: Sort data by satellite and time

df.sort_values(by=["sat_id", "timestamp"], inplace=True)

df[["velocity_x", "velocity_y", "velocity_z"]] = 0.0  # placeholder columns

# Step 2: Compute velocities using finite differences per satellite
unique_sats = df["sat_id"].unique()
dt = 30  # timestep in seconds

for sat in unique_sats:
    sat_df = df[df["sat_id"] == sat].copy()
    pos = sat_df[["position_x", "position_y", "position_z"]].values
    vel = np.zeros_like(pos)

    # Central differences for interior points
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    # Forward/backward difference at ends
    vel[0] = (pos[1] - pos[0]) / dt
    vel[-1] = (pos[-1] - pos[-2]) / dt

    df.loc[sat_df.index, ["velocity_x", "velocity_y", "velocity_z"]] = vel

# Step 3: Assign maneuver labels
# SAT_4 and SAT_5 are assigned a maneuver class = 1,2 at pivot time
# Others = 0 (no action)

df["maneuver_label"] = 0

# Fetch pivot_time from earlier
pivot_mask = (df["sat_id"] == "SAT_4") & (df["position_x"].notnull())
if not pivot_mask.empty:
    pivot_idx = pivot_mask.idxmax()
    #pivot_time = df.loc[pivot_idx, "timestamp"]
    df.loc[(df["sat_id"] == "SAT_4") & (df["timestamp"] == pivot_time), "maneuver_label"] = 1
    df.loc[(df["sat_id"] == "SAT_5") & (df["timestamp"] == pivot_time), "maneuver_label"] = 2  # or same as 1


# Optional: mark SAT_5 as maneuvering instead (or alternate randomly)

# Step 4: Export the extended dataset

df.to_csv("orbits_with_velocity_and_labels.csv", index=False)
print("Extended data with velocity and maneuver labels saved.")

# Print the pivot timestamp (collision timestamp)
print(f"Collision (pivot) occurs at: {pivot_time}")

# =============================================================================
# 3D Plotting using Matplotlib
# =============================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for sat in sorted(df["sat_id"].unique()):
    sat_df = df[df["sat_id"] == sat]
    ax.plot(sat_df["position_x"], sat_df["position_y"], sat_df["position_z"], label=sat)

# Plot Earth as a blue dot at the origin for reference
ax.scatter(0, 0, 0, color='blue', s=200, label='Earth')

# ---------------------------------------------------------------------
# Compute the intersection point from one of the satellite's pivot rows.
# Since SAT_4 and SAT_5 share the same state at the pivot, we can take SAT_4's value.
# ---------------------------------------------------------------------
intersection_row = df[(df["sat_id"] == "SAT_4") & (df["timestamp"] == pivot_time)]
if not intersection_row.empty:
    intersection_point = intersection_row[["position_x", "position_y", "position_z"]].iloc[0].values
    # Plot a red, unfilled circle marker at the intersection point:
    ax.scatter([intersection_point[0]], [intersection_point[1]], [intersection_point[2]],
               s=300, facecolors='none', edgecolors='red', linewidths=2, label='Intersection')
else:
    print("No intersection point found.")

ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.set_title("Satellite Orbits with Intersection at an Intermediate Timestamp")
ax.legend()

plt.show()
