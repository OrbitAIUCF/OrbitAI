'''
This program will propagate orbits of different space objects. Both satellites and debris.

It requires data to be stored in TLE format, ideally a pandas dataframe with columns labeled with TLE Line 1 and TLE Line 2 or the like.

Orbital Propagation is done by using SGP4 (Simplified General Perturbations), a NASA/NORAD algorithm for calculating future positions and velocities using keplerian elements
from the TLE data set.

Propagation just means predicting the future position of an object.
'''
import json

from sgp4.api import Satrec, jday
from sgp4.api import WGS72
import pandas as pd
from datetime import datetime, timedelta, timezone
import numpy as np
import matplotlib.pyplot as plt

#Load the dataframe
df = pd.read_csv("tle_data.csv")



def propagate_orbit(tle1, tle2, start_time, duration=3600, step=60):
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
    for i in range(0, duration, step):
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


    return results

def propagate_row(row):
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

def data_formatting(df):
    # Process each row into a list of sequences
    sequences = [parse_data(row["propagated"]) for i, row in df.iterrows()]
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

    # Save the properly formatted propagated data to a CSV!
    propagated_df.to_csv("propagated_orbits.csv", index=False)


# Get the current UTC time using datetime.now() with timezone set to UTC
start_time = datetime.now(timezone.utc)

#Applies the propagate_orbit function to each row in the dataframe, which is each satellite.
#We apply it using the wrapper function
df["propagated"] = df.apply(propagate_row, axis = 1)

#Graphing!
#graph_positions(df)
#graph_velocities(df)

data_formatting(df)










