import requests
import json

# Define the URL and payload
url = "http://192.168.1.149:8000/generate/"
headers = {"Content-Type": "application/json"}
payload = {"prompt": "An air handling unit does what to the building?"}

# Send POST request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response
if response.status_code == 200:
    print("Generated Text:", response.json().get("generated_text"))
else:
    print(f"Error {response.status_code}: {response.text}")
