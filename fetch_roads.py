
import json
import requests
from pathlib import Path

# OpenStreetMap Overpass API
URL = "https://overpass-api.de/api/interpreter"

# Bengaluru-area bounding box
QUERY = """
[out:json][timeout:60];
way["highway"](12.90,77.50,13.05,77.70);
out tags center;
"""

# Identify our research script
HEADERS = {
    "User-Agent": "RoadRiskRadar/1.0 (student research project)"
}

def main():
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "bengaluru_roads.json"

    print("Requesting Bengaluru road data from OpenStreetMap...")

    try:
        response = requests.post(
            URL,
            data={"data": QUERY},
            headers=HEADERS,
            timeout=90
        )

        response.raise_for_status()
        road_data = response.json()

        elements = road_data.get("elements", [])

        if not elements:
            print("No road data returned. Try again later.")
            return

        # Save only after a successful response
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(road_data, file, indent=2)

        print(f"Road features downloaded: {len(elements)}")
        print(f"Data saved to: {output_file}")
        print("Source: OpenStreetMap contributors")

    except requests.exceptions.HTTPError as error:
        print(f"Server returned an HTTP error: {error}")
        print("The service may be busy or rejecting the request.")
        print("Wait a while before trying again.")

    except requests.exceptions.RequestException as error:
        print(f"Network request failed: {error}")
        print("Check the server's internet connection.")

    except json.JSONDecodeError:
        print("The server response was not valid JSON.")

if __name__ == "__main__":
    main()
