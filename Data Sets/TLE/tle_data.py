import requests
import pandas as pd

class GetSatelliteData:
    def retrieve_data(self):
        # Enter your Space-track credentials
        username = 'adam.mouedden@gmail.com'
        password = 'Adammouedden-2002'

        # Number of TLE elements to pull
        num_elements = 50

        login_url = "https://www.space-track.org/ajaxauth/login"
        tle_url = (
            "https://www.space-track.org/basicspacedata/query/class/gp/"
            "DECAY_DATE/null-val/"
            "EPOCH/>now-30/"
            "PERIOD/<128/"
            "orderby/EPOCH desc/"
            "limit/50/"
            "format/json"
        )

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

        # Split the TLE data by lines (each TLE is in two lines)
        tle_data = response.json()

        # Create a DataFrame with the TLE lines, every two lines correspond to a satellite
        tle_list = []

        #Iterate over the tle_lines in steps of 2 since each element is stored in 2 lines
        for sat in tle_data:
            tle_pair = [sat['TLE_LINE1'], sat['TLE_LINE2']]
            tle_list.append(tle_pair)

        df = pd.DataFrame(tle_list, columns=["tle_line1", "tle_line2"])

        return df


Satellite_Data = GetSatelliteData()
df = Satellite_Data.retrieve_data()

#Data preprocessing! Reshape the data frame by grouping every 2 rows into TLE_LINE1 and TLE_LINE2
print(df.head())

df.to_csv("tle_data.csv", index=False)
