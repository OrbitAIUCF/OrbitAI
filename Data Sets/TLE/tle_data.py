import requests
import pandas as pd

class GetSatelliteData:
    def retrieve_data(self):
        # Space-track credentials
        username = ''
        password = ''

        # Number of TLE elements to pull
        num_elements = 3

        login_url = "https://www.space-track.org/ajaxauth/login"
        tle_url = f"https://www.space-track.org/basicspacedata/query/class/tle/EPOCH/>now-14/orderby/EPOCH desc/limit/{num_elements}/format/tle"

        # Start a session to manage authentication
        session = requests.Session()

        # Login to Space-Track
        response = session.post(login_url, data={"identity": username, "password": password})

        if response.status_code != 200:
            raise Exception("Failed to authenticate")

        # Fetch data
        response = session.get(tle_url)
        if response.status_code != 200:
            raise Exception("Failed to fetch data")

        tle_data = response.text

        # Split the TLE data by lines (each TLE is in two lines)
        tle_lines = tle_data.splitlines()

        # Create a DataFrame with the TLE lines, every two lines correspond to a satellite
        tle_list = []

        #Iterate over the tle_lines in steps of 2 since each element is stored in 2 lines
        for i in range(0, len(tle_lines), 2):
            #Group every two lines into a sublist
            tle_pair = [tle_lines[i], tle_lines[i+1]]

            #Append it to the tle_list
            tle_list.append(tle_pair)

        df = pd.DataFrame(tle_list, columns=["tle_line1", "tle_line2"])

        return df


Satellite_Data = GetSatelliteData()
df = Satellite_Data.retrieve_data()

#Data preprocessing! Reshape the data frame by grouping every 2 rows into TLE_LINE1 and TLE_LINE2
print(df.head())

df.to_csv("tle_data.csv", index=False)