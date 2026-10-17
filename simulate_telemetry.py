import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

TELEMETRY_PATH = 'data/vehicle_telemetry.csv'

def generate_telemetry_batch():
    # Simulate a batch of vehicle sensor events from multiple trips
    # We will cluster some events around a specific high-risk coordinate (e.g., a blind spot near Indiranagar/MG Road)
    hotspots = [
        {'lat': 12.9716, 'lon': 77.5946, 'weight': 'high'}, # Crashpoint candidate
        {'lat': 12.9352, 'lon': 77.6245, 'weight': 'moderate'}
    ]
    
    events = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(50): # Generate 50 telemetry events
        # 70% chance to hit a known hotspot, 30% random road event
        if random.random() < 0.7:
            spot = random.choice(hotspots)
            lat = spot['lat'] + random.uniform(-0.0005, 0.0005)
            lon = spot['lon'] + random.uniform(-0.0005, 0.0005)
        else:
            lat = 12.9 + random.uniform(0.05, 0.15)
            lon = 77.5 + random.uniform(0.05, 0.20)
            
        trip_id = f"TRIP-{random.randint(1000, 9999)}"
        timestamp = base_time + timedelta(hours=random.randint(0, 168), minutes=random.randint(0, 59))
        speed = random.uniform(35.0, 75.0)
        
        # Hard braking trigger if speed is high or near hotspot
        event_type = 'hard_braking' if speed > 50 else 'sudden_deceleration'
        
        events.append({
            'trip_id': trip_id,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'speed_mph': round(speed, 1),
            'event_type': event_type
        })
        
    new_df = pd.DataFrame(events)
    
    if os.path.exists(TELEMETRY_PATH):
        existing_df = pd.read_csv(TELEMETRY_PATH)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
        
    os.makedirs('data', exist_ok=True)
    combined_df.to_csv(TELEMETRY_PATH, index=False)
    print(f"Successfully ingested {len(events)} automated telemetry records into {TELEMETRY_PATH}!")

if __name__ == '__main__':
    generate_telemetry_batch()

