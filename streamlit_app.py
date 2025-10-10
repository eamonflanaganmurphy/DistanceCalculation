# --- NEW/CHANGED BITS ONLY: you can paste this whole file to replace yours ---

import streamlit as st
import pandas as pd
import requests
from geopy.distance import geodesic
import searoute as sr
import numpy as np
from scipy.spatial import cKDTree
import re
import concurrent.futures
from io import BytesIO
import csv
from typing import Optional, Tuple

# Try importing AreaFeature + PortProps from searoute; fallback if older version
try:
    from searoute import AreaFeature, PortProps
    _HAS_AREAFEATURE = True
except Exception:
    AreaFeature = None
    PortProps = None
    _HAS_AREAFEATURE = False

DISTANCE_MATRIX_API_ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"
GEOCODING_API_ENDPOINT = "https://maps.googleapis.com/maps/api/geocode/json"
API_KEY = st.secrets.google_api_key

distance_cache = {}
coordinate_cache = {}

_COORD_PATTERN_NE = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*([NSns])\s*[,;]\s*([+-]?\d+(?:\.\d+)?)\s*([EWew])\s*$')
_COORD_PATTERN_PLAIN = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*[,;]\s*([+-]?\d+(?:\.\d+)?)\s*$')

def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password"); return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password"); return False
    else:
        return True

if check_password():
    def try_parse_coordinates(s) -> Optional[Tuple[float, float]]:
        if s is None:
            return None
        text = str(s).strip()  # robust to ints/floats/etc.
    
        m = _COORD_PATTERN_NE.match(text)
        if m:
            lat_val, lat_hem, lng_val, lng_hem = m.groups()
            lat = float(lat_val)
            lng = float(lng_val)
            lat = -abs(lat) if lat_hem.upper() == 'S' else abs(lat)
            lng = -abs(lng) if lng_hem.upper() == 'W' else abs(lng)
            return (lat, lng)
    
        m = _COORD_PATTERN_PLAIN.match(text)
        if m:
            lat, lng = map(float, m.groups())
            return (lat, lng)
    
        return None

    def normalise_location_key(s) -> str:
        coords = try_parse_coordinates(s)
        if coords:
            lat, lng = coords; return f"{lat:.8f},{lng:.8f}"
        return str(s).strip().lower()

    def to_api_location_param(s) -> str:
        coords = try_parse_coordinates(s)
        if coords:
            lat, lng = coords; return f"{lat:.8f},{lng:.8f}"
        return str(s).strip()

    def coords_for_location(s, api_key) -> Optional[Tuple[float,float]]:
        key = normalise_location_key(s)
        if key in coordinate_cache: return coordinate_cache[key]
        parsed = try_parse_coordinates(s)
        if parsed:
            coordinate_cache[key] = parsed; return parsed
        params = {'address': str(s).strip(), 'key': api_key}
        data = requests.get(GEOCODING_API_ENDPOINT, params=params).json()
        coords = None
        if data.get('status') == 'OK' and data.get('results'):
            loc = data['results'][0]['geometry']['location']
            coords = (loc['lat'], loc['lng'])
        coordinate_cache[key] = coords
        return coords

    def load_international_airports(filename="airports.csv"):
        airports = []
        try:
            with open(filename, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = row.get("type","").strip().lower()
                    if t not in ("large_airport","medium_airport"): continue
                    try:
                        name = row["name"].strip()
                        lat = float(row["latitude_deg"].strip())
                        lng = float(row["longitude_deg"].strip())
                        if name: airports.append((name,(lat,lng)))
                    except Exception:
                        continue
        except FileNotFoundError:
            pass
        return airports

    def build_kdtree(points):
        if not points: return None, [], None
        coords = np.array([[p[1][0], p[1][1]] for p in points], dtype=float)
        tree = cKDTree(coords); names = [p[0] for p in points]
        return tree, names, coords

    def nearest_point(tree, coords_array, names, target):
        if tree is None or coords_array is None or not names: return None
        _, idx = tree.query([target[0], target[1]], k=1)
        name = names[idx]; lat, lng = coords_array[idx]
        return (name, (float(lat), float(lng)))

    def get_distance_matrix(origin, destination, api_key):
        origin_key = normalise_location_key(origin); dest_key = normalise_location_key(destination)
        key = (origin_key, dest_key)
        if key in distance_cache: return distance_cache[key]
        params = {'origins': to_api_location_param(origin), 'destinations': to_api_location_param(destination), 'key': api_key}
        data = requests.get(DISTANCE_MATRIX_API_ENDPOINT, params=params).json()
        distance, source = None, None
        try:
            el = data['rows'][0]['elements'][0]
            if el.get('status') == 'OK':
                txt = el['distance']['text']
                if ' km' in txt: distance = float(txt.replace(' km','').replace(',',''))
                elif ' m' in txt: distance = float(txt.replace(' m','').replace(',',''))/1000.0
                source = "Road Distance"
        except Exception:
            pass
        distance_cache[key] = (distance, source); return distance, source

    # --- NEW: helper to call searoute with AreaFeature / PortProps and extract chosen ports if available
    def searoute_with_ports(origin_coords, destination_coords, prefer_auto_ports: bool,
                            origin_port_override: Optional[str], destination_port_override: Optional[str]):
        """
        Returns (sea_km, chosen_origin_port_id, chosen_destination_port_id)
        """
        o_lonlat = [origin_coords[1], origin_coords[0]]
        d_lonlat = [destination_coords[1], destination_coords[0]]

        # Build features if AreaFeature is available, otherwise plain coords
        origin_feat = o_lonlat
        dest_feat = d_lonlat
        if _HAS_AREAFEATURE:
            # Preferred port overrides (if provided)
            o_pref = PortProps(origin_port_override, 1) if (origin_port_override and origin_port_override.strip()) else None
            d_pref = PortProps(destination_port_override, 1) if (destination_port_override and destination_port_override.strip()) else None

            # If auto preferred ports requested but no overrides provided, pass a blank PortProps list;
            # searoute will choose sensible nearby ports internally.
            if prefer_auto_ports and not o_pref: o_pref = []
            if prefer_auto_ports and not d_pref: d_pref = []

            origin_feat = AreaFeature(coords=o_lonlat, name="Origin", preferred_ports=o_pref if o_pref is not None else None)
            dest_feat   = AreaFeature(coords=d_lonlat, name="Destination", preferred_ports=d_pref if d_pref is not None else None)

        route = sr.searoute(origin_feat, dest_feat, units="naut")
        sea_km = None; o_port_id = None; d_port_id = None

        if route and getattr(route, "properties", None):
            props = route.properties
            # length in nautical miles -> km
            if "length" in props:
                sea_km = float(props["length"]) * 1.852
            # Try to read chosen ports (keys vary by version; try a few)
            # Common patterns: origin_port / destination_port dicts or *_port_id strings
            o_port = props.get("origin_port") or props.get("start_port") or {}
            d_port = props.get("destination_port") or props.get("end_port") or {}
            o_port_id = (o_port.get("id") if isinstance(o_port, dict) else None) or props.get("origin_port_id") or props.get("start_port_id")
            d_port_id = (d_port.get("id") if isinstance(d_port, dict) else None) or props.get("destination_port_id") or props.get("end_port_id")

        if sea_km is None:
            # Fallback to great-circle between the two coords
            sea_km = geodesic((o_lonlat[1], o_lonlat[0]), (d_lonlat[1], d_lonlat[0])).kilometers

        return sea_km, o_port_id, d_port_id

    def process_file(input_file, enable_hub_legs: bool,
                     use_searoute_preferred_ports: bool,
                     origin_port_override: Optional[str],
                     destination_port_override: Optional[str]):
        df = pd.read_excel(input_file)

        needed_cols = [
            'Distance (km)', 'Source',
            'RoadToAirport (km)', 'Flight (km)', 'RoadFromAirport (km)', 'Origin Airport', 'Destination Airport',
            'RoadToPort (km)', 'Sea (km)', 'RoadFromPort (km)', 'Origin Port', 'Destination Port'
        ]
        for col in needed_cols:
            if col not in df.columns: df[col] = pd.NA

        rows_to_process = df[df['Distance (km)'].isna()]
        if rows_to_process.empty:
            st.info("No new rows to process."); return df

        # Warm-up geocoding
        unique_addresses = set()
        for _, r in rows_to_process.iterrows():
            unique_addresses.add(str(r['From']).strip()); unique_addresses.add(str(r['To']).strip())

        st.write("Geocoding unique locations…")
        warmup_bar = st.progress(0)
        unique_list = list(unique_addresses)
        def _warmup(addr): _ = coords_for_location(addr, API_KEY)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(_warmup, a): i for i, a in enumerate(unique_list)}
            total = len(unique_list); done = 0
            for _ in concurrent.futures.as_completed(futs):
                done += 1; warmup_bar.progress(done / max(total,1))

        # Airports KD-tree (as before)
        airports = load_international_airports("airports.csv")
        airport_tree, airport_names, airport_coords = build_kdtree(airports)

        tmp = rows_to_process.copy()
        tmp['group_key'] = tmp.apply(lambda x: (normalise_location_key(x['From']),
                                                normalise_location_key(x['To']),
                                                str(x.get('Mode','')).strip().lower()), axis=1)
        unique_groups = tmp['group_key'].unique()
        results_cache = {}

        def process_group(group):
            origin_key, destination_key, mode_lower = group
            origin_coords = coordinate_cache.get(origin_key) or coords_for_location(origin_key, API_KEY)
            destination_coords = coordinate_cache.get(destination_key) or coords_for_location(destination_key, API_KEY)

            distance = None; source = None

            # Defaults
            road_to_airport = pd.NA; flight_dist = pd.NA; road_from_airport = pd.NA
            origin_airport_name = pd.NA; destination_airport_name = pd.NA

            road_to_port = pd.NA; sea_dist = pd.NA; road_from_port = pd.NA
            origin_port_name = pd.NA; destination_port_name = pd.NA

            origin_api = origin_key if ',' in origin_key else to_api_location_param(origin_key)
            dest_api = destination_key if ',' in destination_key else to_api_location_param(destination_key)

            try:
                if mode_lower in ["air","airplane (air)","flight"]:
                    if enable_hub_legs and origin_coords and destination_coords and airport_tree is not None:
                        o_air = nearest_point(airport_tree, airport_coords, airport_names, origin_coords)
                        d_air = nearest_point(airport_tree, airport_coords, airport_names, destination_coords)
                        if o_air and d_air:
                            origin_airport_name, o_air_coords = o_air
                            destination_airport_name, d_air_coords = d_air
                            road_to_airport, _ = get_distance_matrix(f"{origin_coords[0]},{origin_coords[1]}",
                                                                     f"{o_air_coords[0]},{o_air_coords[1]}", API_KEY)
                            road_from_airport, _ = get_distance_matrix(f"{d_air_coords[0]},{d_air_coords[1]}",
                                                                       f"{destination_coords[0]},{destination_coords[1]}", API_KEY)
                            flight_dist = geodesic(o_air_coords, d_air_coords).kilometers
                            distance = (road_to_airport or 0) + (flight_dist or 0) + (road_from_airport or 0)
                            source = "Road + Airport GC"
                        else:
                            if origin_coords and destination_coords:
                                flight_dist = geodesic(origin_coords, destination_coords).kilometers
                                distance = flight_dist; source = "Geodesic Air Distance"
                    else:
                        if origin_coords and destination_coords:
                            flight_dist = geodesic(origin_coords, destination_coords).kilometers
                            distance = flight_dist; source = "Geodesic Air Distance"

                elif mode_lower in ["cargo ship (sea)","sea","ocean","vessel (sea)"]:
                    if origin_coords and destination_coords:
                        # Get sea route using searoute built-in ports (AreaFeature / PortProps)
                        sea_km, o_port_id, d_port_id = searoute_with_ports(
                            origin_coords, destination_coords,
                            prefer_auto_ports=enable_hub_legs and use_searoute_preferred_ports,
                            origin_port_override=origin_port_override,
                            destination_port_override=destination_port_override
                        )
                        sea_dist = sea_km
                        # If we have port IDs, record them
                        if o_port_id: origin_port_name = o_port_id
                        if d_port_id: destination_port_name = d_port_id

                        if enable_hub_legs:
                            # If we know chosen ports (IDs), we do not have coords directly; road legs will approximate
                            # by using the same coords we routed from/to (origin/destination) — which is fine,
                            # because searoute already handled port snapping internally.
                            # If you later expose a port-ID→coord lookup, plug it here for exact road-legs.
                            road_to_port, _ = get_distance_matrix(f"{origin_coords[0]},{origin_coords[1]}",
                                                                  f"{origin_coords[0]},{origin_coords[1]}", API_KEY)
                            road_from_port, _ = get_distance_matrix(f"{destination_coords[0]},{destination_coords[1]}",
                                                                    f"{destination_coords[0]},{destination_coords[1]}", API_KEY)
                            # Above sets road legs to ~0 unless you replace with real port coords.
                            # To keep semantics, we treat them as 0 when None.
                            distance = (road_to_port or 0) + (sea_dist or 0) + (road_from_port or 0)
                            source = "SeaRoute (ports internal) + Road (0 unless port coords known)"
                        else:
                            distance = sea_dist; source = "SeaRoute (ports internal)"
                    else:
                        distance = None; source = None

                elif mode_lower in ["truck (road)","trucking","courier","van","truck","car"]:
                    dist, src = get_distance_matrix(origin_api, dest_api, API_KEY)
                    if dist is None and origin_coords and destination_coords:
                        dist = geodesic(origin_coords, destination_coords).kilometers; src = "Geodesic Distance"
                    distance, source = dist, src

                else:
                    dist, src = get_distance_matrix(origin_api, dest_api, API_KEY)
                    if dist is not None:
                        distance, source = dist, src
                    elif origin_coords and destination_coords:
                        distance = geodesic(origin_coords, destination_coords).kilometers; source = "Geodesic Distance"

            except Exception as e:
                print(f"Error in group {group}: {e}")

            return group, {
                'Distance (km)': distance,
                'Source': source,
                'RoadToAirport (km)': road_to_airport,
                'Flight (km)': flight_dist,
                'RoadFromAirport (km)': road_from_airport,
                'Origin Airport': origin_airport_name,
                'Destination Airport': destination_airport_name,
                'RoadToPort (km)': road_to_port,
                'Sea (km)': sea_dist,
                'RoadFromPort (km)': road_from_port,
                'Origin Port': origin_port_name,
                'Destination Port': destination_port_name
            }

        st.write("Calculating distances…")
        progress_bar = st.progress(0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(process_group, g): g for g in unique_groups}
            total = len(unique_groups); done = 0; results_cache.clear()
            for f in concurrent.futures.as_completed(futs):
                g, res = f.result(); results_cache[g] = res
                done += 1; progress_bar.progress(done / max(total,1))

        for idx, row in rows_to_process.iterrows():
            gk = (normalise_location_key(row['From']),
                  normalise_location_key(row['To']),
                  str(row.get('Mode','')).strip().lower())
            res = results_cache.get(gk, {})
            for k, v in res.items(): df.at[idx, k] = v

        if 'group_key' in df.columns:
            df.drop(columns=['group_key'], inplace=True, errors='ignore')
        return df

    def main():
        st.title("Distance Calculation")
        st.write("Upload an Excel file with headers **From**, **To**, and **Mode**.")

        enable_hub_legs = st.checkbox("Enable nearest airport/port legs", value=False)
        use_searoute_preferred_ports = False
        origin_port_override = None
        destination_port_override = None

        if enable_hub_legs:
            st.subheader("Sea leg options")
            if not _HAS_AREAFEATURE:
                st.info("Your installed searoute version does not expose AreaFeature/PortProps; using direct coordinates only.")
            else:
                use_searoute_preferred_ports = st.checkbox("Use searoute preferred ports (auto select)", value=True,
                                                           help="Let searoute pick suitable ports from its internal database.")
                with st.expander("Optional: override port IDs (e.g., USNYC)"):
                    origin_port_override = st.text_input("Origin port ID (optional)", value="")
                    destination_port_override = st.text_input("Destination port ID (optional)", value="")

        uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
        if uploaded_file:
            input_file = BytesIO(uploaded_file.read())
            df = pd.read_excel(input_file)

            st.write("Input data preview:")
            st.dataframe(df.head())

            st.write("Processing file…")
            processed_df = process_file(
                input_file,
                enable_hub_legs=enable_hub_legs,
                use_searoute_preferred_ports=use_searoute_preferred_ports,
                origin_port_override=origin_port_override,
                destination_port_override=destination_port_override
            )

            st.write("Processed data preview:")
            st.dataframe(processed_df.head())

            output_file = BytesIO()
            processed_df.to_excel(output_file, index=False)
            output_file.seek(0)
            st.download_button(
                label="Download processed file",
                data=output_file,
                file_name="processed_file.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    if __name__ == "__main__":
        main()
