

import requests
import json

API_URL = "https://campaignx.inxiteout.ai/api/v1/get_customer_cohort"

headers = {
    "X-API-Key": "xi6nFrz5T30S1xPYkH_j8YYoMNBPCIHhf5623eZYlGo",
    "Content-Type": "application/json"
}

response = requests.get(API_URL, headers=headers, timeout=30)

if response.status_code == 200:
    customer_data = response.json()

    with open("customer_cohort.json", "w", encoding="utf-8") as f:
        json.dump(customer_data, f, indent=4)

    print("Success! Data saved to 'customer_cohort.json'")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")
    print(response.text)