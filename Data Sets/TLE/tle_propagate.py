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

#Load the dataframe
df = pd.read_csv("tle_data.csv")

def propagate_orbit(tle1, tle2, start_time, duration=60, step=10):
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
            results.append([t, position, velocity])

def propagate_row(row):
    return propagate_orbit(row["tle_line1"], row["tle_line2"], start_time)

# Get the current UTC time using datetime.now() with timezone set to UTC
start_time = datetime.now(timezone.utc)

#Applies the propagate_orbit function to each row in the dataframe, which is each satellite.
#We apply it using the wrapper function
df["propagated"] = df.apply(propagate_row, axis = 1)

df.to_csv("tle_data_propagated", index=False)

