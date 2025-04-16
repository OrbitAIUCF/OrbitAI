'''
This program will propagate orbits of different space objects. Both satellites and debris.

It requires data to be stored in TLE format, ideally a pandas dataframe with columns labeled with TLE Line 1 and TLE Line 2 or the like.

Orbital Propagation is done by using SGP4 (Simplified General Perturbations), a NASA/NORAD algorithm for calculating future positions and velocities using keplerian elements
from the TLE data set.

Propagation just means predicting the future position of an object.
'''

from sgp4.api import Satrec, jday
from sgp4.api import WGS72
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt
import math

#Load the dataframe
df = pd.read_csv("tle_data.csv")
df["tle_line1"] = df["tle_line1"].str.strip()
df["tle_line2"] = df["tle_line2"].str.strip()


def propagate_orbit(tle1, tle2, start_time, duration=8100, step=30):
    '''

    :param tle1: The first line of the TLE
    :param tle2: The second line of the TLE
    :param start_time: the starting time for the propagation
    :param duration: How long to propagate
    :param step: The time interval between each propagated step
    :return: The position and velocity
    '''


    #Satrec is used to represent a satellite's orbital parameters and handle the propagation through the SGP4 model
    #WGS72 (World Geodetic System 1972) model is a standard model for Earth's gravitational parameters.

    #Initialize a Satrec object, twoline2rv method converts the TLE into a satellite record that can be used for propagation
    sat = Satrec.twoline2rv(tle1, tle2, WGS72)

    #Creates a list of timestamps starting from the start time, incremented of "step" seconds until the defined duration.
    #This represents the time points at which the satellite position will be predicted.
    timestamps = []
    for i in range(0, duration + step, step):
        timestamps.append(start_time + timedelta(seconds=i))

    #We will store the positions and velocities at each time step here
    results = []

    #Iterate through all the timestamps
    for t in timestamps:
        #Convert the time stamp from TLE into Julian Date and Fraction
        julian_date, fraction = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)

        '''
        All of the time extraction was done to for input into SGP4, which takes in time variables and will return the following:
        e: the error code from the propagation, where 0 means successful.
        position: the satellite's position in the Earth-centered inertial frame (ECI) coordinate system
        velocity: the satellite's velocity in the ECI
        '''
        e, position, velocity = sat.sgp4(julian_date, fraction)

        if any(np.isnan(position)) or any(np.isnan(velocity)):
            print(f"[SGP4] Bad propagation at {t} for TLE: {tle1.strip()} - skipping")
            return []


        if (e == 0):
            #Append the current time stamp, position, and velocity
            results.append({
                "timestamp": t,
                "position_x": position[0],
                "position_y": position[1],
                "position_z": position[2],
                "velocity_x": velocity[0],
                "velocity_y": velocity[1],
                "velocity_z": velocity[2]
            })
        if math.isnan(position[0]) or math.isnan(velocity[0]):
            print(f"NaN detected at time {t}")
    return results

def propagate_row(row):
    object_id = row["tle_line1"][2:7]
    print(f"Propagating satellite {object_id}")
    return propagate_orbit(row["tle_line1"], row["tle_line2"], start_time)

def graph_positions(df):
    # Iterate through each row (satellite) and create separate plots
    for index, row in df.iterrows():
        propagated_data = row["propagated"]

        # Extract time, x, y, z values
        timestamps = [item["timestamp"] for item in propagated_data]
        x_values = [item["position_x"] for item in propagated_data]
        y_values = [item["position_y"] for item in propagated_data]
        z_values = [item["position_z"] for item in propagated_data]

        # Create x vs. time plot
        plt.figure()  # Create a new figure for each satellite
        plt.plot(timestamps, x_values)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Position X (km)")
        plt.title(f"Position X vs. Time (Satellite {index + 1})")
        plt.grid(True)
        plt.show()

        # Create y vs. time plot
        plt.figure()
        plt.plot(timestamps, y_values)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Position Y (km)")
        plt.title(f"Position Y vs. Time (Satellite {index + 1})")
        plt.grid(True)
        plt.show()

        # Create z vs. time plot
        plt.figure()
        plt.plot(timestamps, z_values)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Position Z (km)")
        plt.title(f"Position Z vs. Time (Satellite {index + 1})")
        plt.grid(True)
        plt.show()

def graph_velocities(df):
    # Iterate through each row (satellite) and create separate plots
    for index, row in df.iterrows():
        propagated_data = row["propagated"]

        # Extract time, x, y, z values
        timestamps = [item["timestamp"] for item in propagated_data]
        x_values = [item["velocity_x"] for item in propagated_data]
        y_values = [item["velocity_y"] for item in propagated_data]
        z_values = [item["velocity_z"] for item in propagated_data]

        # Create x vs. time plot
        plt.figure()  # Create a new figure for each satellite
        plt.plot(timestamps, x_values)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Velocity X (km/s)")
        plt.title(f"Velocity X vs. Time (Satellite {index + 1})")
        plt.grid(True)
        plt.show()

        # Create y vs. time plot
        plt.figure()
        plt.plot(timestamps, y_values)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Velocity Y (km/s)")
        plt.title(f"Velocity Y vs. Time (Satellite {index + 1})")
        plt.grid(True)
        plt.show()

        # Create z vs. time plot
        plt.figure()
        plt.plot(timestamps, z_values)
        plt.xlabel("Time (UTC)")
        plt.ylabel("Velocity Z (km/s)")
        plt.title(f"Velocity Z vs. Time (Satellite {index + 1})")
        plt.grid(True)
        plt.show()

#This function formats the data for model training
def parse_data(propagated_data):
    '''Converts a string with propagated state vectors into a numpy array of shape (B, T, F):
    B is the batch size/index
    T is the timesteps per sequence (number of propagated step vectors per object)
    F is the dimension of the features'''

    array = np.array([[sv["position_x"], sv["position_y"], sv["position_z"], sv["velocity_x"], sv["velocity_y"], sv["velocity_z"]] for sv in propagated_data])
    return array

def data_formatting(df, duration=8100, steps=30):
    # Process each row into a list of sequences

    expected_timesteps = int(duration / steps) + 1

    sequences = []
    #Drop any invalid propagated sequences
    for i, row in df.iterrows():
        parsed = parse_data(row["propagated"])
        if parsed.shape == (expected_timesteps, 6):
            sequences.append(parsed)
        else:
            print(f"Dropping incomplete sequence {i}: shape {parsed.shape}")


    # Convert to (B, T, F) NumPy array
    data_array = np.array(sequences)  # Shape: (B, T, 6)

    '''Formatting the data for model training.'''
    # Extract NORAD ID (First 5 characters after column 2 in TLE Line 1)
    df["object_id"] = df["tle_line1"].str.slice(2, 7).str.strip()

    '''Saving the DF as a csv file.'''
    # Expand the propagated data into separate rows!
    propagated_df = df.explode("propagated").reset_index(drop=True)

    # Convert dictionaries into separate columns!
    propagated_df = propagated_df["propagated"].apply(pd.Series)

    # Reattach the object ID so we know which satellite each state vector belongs to
    propagated_df["object_id"] = df.explode("propagated")["object_id"].reset_index(drop=True)

    # Reorder columns to have 'object_id' first
    column_order = ["object_id"] + [col for col in propagated_df.columns if col != "object_id"]
    propagated_df = propagated_df[column_order]

    # Save to CSV
    propagated_df.to_csv("propagated_orbits.csv", index=False)

def save_training_data(df, output_filename="training_data.csv"):
    '''Saves simplified training data extracted from the "propagated" column.'''

    # Expand the 'propagated' column into separate rows
    expanded_df = df.explode('propagated').reset_index(drop=True)

    # Expand the dictionary inside 'propagated' into columns
    propagated_df_expanded = pd.json_normalize(expanded_df['propagated'])

    # Convert timestamp column to datetime
    propagated_df_expanded['timestamp'] = pd.to_datetime(propagated_df_expanded['timestamp'], utc=True)
    reference_time = propagated_df_expanded['timestamp'].min()

    # Create a new column "time" (float)
    propagated_df_expanded['time'] = (propagated_df_expanded['timestamp'] - reference_time).dt.total_seconds()

    # Select required columns
    training_df = propagated_df_expanded[['time', 'position_x', 'position_y', 'position_z',
                                          'velocity_x', 'velocity_y', 'velocity_z']]

    # Save the selected columns to CSV
    training_df.to_csv(output_filename, index=False)


# Get the current UTC time using datetime.now() with timezone set to UTC
start_time = datetime.now(timezone.utc)

df["propagated"] = df.apply(lambda row: propagate_row(row), axis=1)

data_formatting(df)

print(df.columns)
save_training_data(df, "../../src/Models/new_training_data.csv")











