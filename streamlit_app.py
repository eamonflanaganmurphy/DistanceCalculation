import streamlit as st
import pandas as pd
import requests
from geopy.distance import geodesic
import searoute as sr  # For sea distance
import numpy as np
from scipy.spatial import cKDTree
import re
import concurrent.futures
from io import BytesIO
import csv

# Google Maps API configurations
DISTANCE_MATRIX_API_ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"
GEOCODING_API_ENDPOINT = "https://maps.googleapis.com/maps/api/geocode/json"
API_KEY = st.secrets["google"]["api_key"]

# Caches
distance_cache = {}      # {(origin_norm, destination_norm): (distance, source)}
coordinate_cache = {}    # {location_name_lowercase_or_canonical_coord: (lat, lng)}

# ---------- NEW: coordinate parsing helpers ----------
_COORD_PATTERN_NE = re.compile(
    r'^\s*([+-]?\d+(?:\.\d+)?)\s*([NSns])\s*[,;]\s*([+-]?\d+(?:\.\d+)?)\s*([EWew])\s*$'
)
_COORD_PATTERN_PLAIN = re.compile(
    r'^\s*([+-]?\d+(?:\.\d+)?)\s*[,;]\s*([+-]?\d+(?:\.\d+)?)\s*$'
)

def try_parse_coordinates(s):
    if not isinstance(s, str):
        return None
    text = s.strip()

    m = _COORD_PATTERN_NE.match(text)
    if m:
        lat_val, lat_hem, lng_val, lng_hem = m.groups()
        lat = float(lat_val)
        lng = float(lng_val)
        if lat_hem.upper() == 'S':
            lat = -abs(lat)
        else:
            lat = abs(lat)
        if lng_hem.upper() == 'W':
            lng = -abs(lng)
        else:
            lng = abs(lng)
        return (lat, lng)

    m = _COORD_PATTERN_PLAIN.match(text)
    if m:
        lat, lng = map(float, m.groups())
        return (lat, lng)

    return None

def normalise_location_key(s):
    coords = try_parse_coordinates(s)
    if coords:
        lat, lng = coords
        return f"{lat:.8f},{lng:.8f}"
    return str(s).strip().lower()

def to_api_location_param(s):
    coords = try_parse_coordinates(s)
    if coords:
        lat, lng = coords
        return f"{lat:.8f},{lng:.8f}"
    return str(s).strip()

def coords_for_location(s, api_key):
    key = normalise_location_key(s)
    if key in coordinate_cache:
        return coordinate_cache[key]

    parsed = try_parse_coordinates(s)
    if parsed:
        coordinate_cache[key] = parsed
        return parsed

    params = {'address': str(s).strip(), 'key': api_key}
    response = requests.get(GEOCODING_API_ENDPOINT, params=params)
    data = response.json()

    coords = None
    if data.get('status') == 'OK' and data.get('results'):
        loc = data['results'][0]['geometry']['location']
        coords = (loc['lat'], loc['lng'])

    coordinate_cache[key] = coords
    return coords

def load_international_airports(filename="airports.csv"):
    airports = []
    with open(filename, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            airport_type = row["type"].strip().lower()
            if airport_type not in ("large_airport", "medium_airport"):
                continue
            try:
                name = row["name"].strip()
                lat = float(row["latitude_deg"].strip())
                lng = float(row["longitude_deg"].strip())
                if not name:
                    continue
            except (ValueError, KeyError):
                continue
            airports.append((name, (lat, lng)))
    return airports

def get_distance_matrix(origin, destination, api_key):
    origin_key = normalise_location_key(origin)
    destination_key = normalise_location_key(destination)
    key = (origin_key, destination_key)
    if key in distance_cache:
        return distance_cache[key]

    params = {
        'origins': to_api_location_param(origin),
        'destinations': to_api_location_param(destination),
        'key': api_key
    }
    response = requests.get(DISTANCE_MATRIX_API_ENDPOINT, params=params)
    data = response.json()

    distance, source = None, None
    try:
        el = data['rows'][0]['elements'][0]
        if el.get('status') == 'OK':
            distance_text = el['distance']['text']
            if ' km' in distance_text:
                distance = float(distance_text.replace(' km', '').replace(',', ''))
            elif ' m' in distance_text:
                distance = float(distance_text.replace(' m', '').replace(',', '')) / 1000.0
            source = "Road Distance"
    except (IndexError, KeyError, TypeError):
        pass

    distance_cache[key] = (distance, source)
    return distance, source

# Processing data
def process_file(input_file, output_file):
    df = pd.read_excel(input_file)
    international_airports = load_international_airports("airports.csv")

    for col in [
        'Distance (km)', 'Source', 'RoadToAirport (km)', 'Flight (km)',
        'RoadFromAirport (km)', 'Origin Airport', 'Destination Airport'
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    rows_to_process = df[df['Distance (km)'].isna()]
    if rows_to_process.empty:
        print("No new rows to process. Exiting.")
        return df

    unique_addresses = set()
    for _, row in rows_to_process.iterrows():
        unique_addresses.add(str(row['From']).strip())
        unique_addresses.add(str(row['To']).strip())

    def _warmup(addr):
        _ = coords_for_location(addr, API_KEY)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(_warmup, unique_addresses))

    rows_to_process['group_key'] = rows_to_process.apply(
        lambda x: (normalise_location_key(x['From']),
                   normalise_location_key(x['To']),
                   str(x.get('Mode', '')).strip().lower()),
        axis=1
    )

    unique_groups = rows_to_process['group_key'].unique()
    results_cache = {}

    def process_group(group):
        origin_key, destination_key, mode_lower = group
        origin_raw = origin_key
        destination_raw = destination_key

        distance = None
        source = None
        road_to_airport = pd.NA
        flight_dist = pd.NA
        road_from_airport = pd.NA
        origin_airport_name = pd.NA
        destination_airport_name = pd.NA

        origin_coords = coordinate_cache.get(origin_raw)
        destination_coords = coordinate_cache.get(destination_raw)

        origin_api = origin_raw if ',' in origin_raw else to_api_location_param(origin_raw)
        dest_api = destination_raw if ',' in destination_raw else to_api_location_param(destination_raw)

        try:
            if mode_lower in ["truck (road)", "trucking", "courier", "van", "truck", "car"]:
                dist, src = get_distance_matrix(origin_api, dest_api, API_KEY)
                if dist is None and origin_coords and destination_coords:
                    dist = geodesic(origin_coords, destination_coords).kilometers
                    src = "Geodesic Distance"
                distance, source = dist, src

            elif mode_lower in ["air", "airplane (air)", "flight"]:
                if origin_coords and destination_coords:
                    flight_dist = geodesic(origin_coords, destination_coords).kilometers
                    distance = flight_dist
                    source = "Geodesic Air Distance"
                else:
                    distance = geodesic(origin_coords, destination_coords).kilometers if origin_coords and destination_coords else None
                    source = "Geodesic Distance"

            elif mode_lower in ["cargo ship (sea)", "sea", "ocean", "vessel (sea)"]:
                if origin_coords and destination_coords:
                    origin_lnglat = [origin_coords[1], origin_coords[0]]
                    dest_lnglat = [destination_coords[1], destination_coords[0]]
                    route = sr.searoute(origin_lnglat, dest_lnglat, units="naut")
                    if route and getattr(route, "properties", None) and "length" in route.properties:
                        distance_nautical = route.properties["length"]
                        distance = distance_nautical * 1.852
                        source = "Sea Distance"
                    else:
                        distance = geodesic(origin_coords, destination_coords).kilometers
                        source = "Geodesic Distance"
                else:
                    dist, src = get_distance_matrix(origin_api, dest_api, API_KEY)
                    if dist is not None:
                        distance, source = dist, src
                    elif origin_coords and destination_coords:
                        distance = geodesic(origin_coords, destination_coords).kilometers
                        source = "Geodesic Distance"

            else:
                dist, src = get_distance_matrix(origin_api, dest_api, API_KEY)
                if dist is not None:
                    distance, source = dist, src
                elif origin_coords and destination_coords:
                    distance = geodesic(origin_coords, destination_coords).kilometers
                    source = "Geodesic Distance"

        except Exception as e:
            print(f"Error in group {group}: {e}")

        return group, {
            'Distance (km)': distance,
            'Source': source,
            'RoadToAirport (km)': road_to_airport,
            'Flight (km)': flight_dist,
            'RoadFromAirport (km)': road_from_airport,
            'Origin Airport': origin_airport_name,
            'Destination Airport': destination_airport_name
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_group = {executor.submit(process_group, group): group for group in unique_groups}
        for future in concurrent.futures.as_completed(future_to_group):
            group, result = future.result()
            results_cache[group] = result

    for index, row in rows_to_process.iterrows():
        group_key = row['group_key']
        res = results_cache.get(group_key, {})
        for k, v in res.items():
            df.at[index, k] = v

    df.drop(columns=['group_key'], inplace=True, errors='ignore')
    return df

# Web app
def main():
    st.title("Distance Calculation and File Processing App")

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file:
        input_file = BytesIO(uploaded_file.read())
        df = pd.read_excel(input_file)

        st.write("Input Data:")
        st.dataframe(df.head())

        st.write("Processing file... Please wait.")

        processed_df = process_file(input_file, None)

        st.write("Processed Data:")
        st.dataframe(processed_df.head())

        # Allow the user to download the output file
        output_file = BytesIO()
        processed_df.to_excel(output_file, index=False)
        output_file.seek(0)
        st.download_button(
            label="Download Processed File",
            data=output_file,
            file_name="processed_file.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()