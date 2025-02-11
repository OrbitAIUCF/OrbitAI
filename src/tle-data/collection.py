#pre-reqs
import requests
import json
#fill in with credentials (not listed because it is a public repo)
username = "" 
password = ""



#fetches from past 2 weeks with a limit of 1000 objects - can be altered, this parameter is just for simplicity of testing
login_url = "https://www.space-track.org/ajaxauth/login"
tle_url = "https://www.space-track.org/basicspacedata/query/class/tle/EPOCH/>now-14/orderby/EPOCH desc/limit/1000/format/tle"  


#session start
session = requests.Session()

#most of this error handling is left-over from when I was having issues with grabbing the data
#probably not needed anymore
try:
    response = session.post(login_url, data={"identity": username, "password": password})

    #error handling, used for initial debug     
    if response.status_code != 200:
        print("login failed:", response.status_code)
        print("response:", response.text)
        exit()
   

    response = session.get(tle_url)

    #server error handling
    if response.status_code != 200:
        print(f"api request failed: {response.status_code}")
        print("response:", response.text)
        exit()

    try:
        error_check = response.json()  # If response is JSON, it's probably an error
        print("api error:", json.dumps(error_check, indent=2))
        exit()
    except ValueError:
        pass

    tle_data = response.text.strip()
    if not tle_data:
        print("query failed")
        exit()

    with open("tle_data.txt", "w") as file:
        file.write(tle_data)

    print("data grab worked")

except requests.exceptions.RequestException as e:
    print("network error:", e)
    exit()
