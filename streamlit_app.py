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
import json
import os
from typing import Optional, Tuple, List, Dict

# Try importing AreaFeature + PortProps from searoute (newer builds)
try:
    from searoute import AreaFeature, PortProps
    _HAS_AREAFEATURE = True
except Exception:
    AreaFeature = None
    PortProps = None
    _HAS_AREAFEATURE = False

# Google Maps API
DISTANCE_MATRIX_API_ENDPOINT = "https://maps.googleapis.com/maps/api/distancematrix/json"
GEOCODING_API_ENDPOINT = "https://maps.googleapis.com/maps/api/geocode/json"
API_KEY = st.secrets.google_api_key

# Caches
distance_cache: Dict[Tuple[str, str], Tuple[Optional[float], Optional[str]]] = {}
coordinate_cache: Dict[str, Optional[Tuple[float, float]]] = {}

# Coordinate regex
_COORD_PATTERN_NE = re.compile(
    r'^\s*([+-]?\d+(?:\.\d+)?)\s*([NSns])\s*[,;]\s*([+-]?\d+(?:\.\d+)?)\s*([EWew])\s*$'
)
_COORD_PATTERN_PLAIN = re.compile(
    r'^\s*([+-]?\d+(?:\.\d+)?)\s*[,;]\s*([+-]?\d+(?:\.\d+)?)\s*$'
)


def check_password():
    """Simple password gate backed by st.secrets['password']"""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False
    else:
        return True


def try_parse_coordinates(s) -> Optional[Tuple[float, float]]:
    """Parse 'lat,lon' or 'lat N/S, lon E/W' strings; robust to non-strings."""
    if s is None:
        return None
    text = str(s).strip()

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
        lat, lng = coords
        return f"{lat:.8f},{lng:.8f}"
    return str(s).strip().lower()


def to_api_location_param(s) -> str:
    coords = try_parse_coordinates(s)
    if coords:
        lat, lng = coords
        return f"{lat:.8f},{lng:.8f}"
    return str(s).strip()


def coords_for_location(s, api_key) -> Optional[Tuple[float, float]]:
    key = normalise_location_key(s)
    if key in coordinate_cache:
        return coordinate_cache[key]

    parsed = try_parse_coordinates(s)
    if parsed:
        coordinate_cache[key] = parsed
        return parsed

    params = {'address': str(s).strip(), 'key': api_key}
    data = requests.get(GEOCODING_API_ENDPOINT, params=params).json()

    coords = None
    if data.get('status') == 'OK' and data.get('results'):
        loc = data['results'][0]['geometry']['location']
        coords = (loc['lat'], loc['lng'])

    coordinate_cache[key] = coords
    return coords


# ---------- Airports (with codes in display name) ----------
def load_international_airports(filename="airports.csv") -> List[Tuple[str, Tuple[float, float]]]:
    """
    Read airports (expects columns such as: name, latitude_deg, longitude_deg, type, iata_code or ident/gps_code).
    Uses only large/medium airports.
    Returns list of (display_name_with_code, (lat, lng)).
    """
    airports: List[Tuple[str, Tuple[float, float]]] = []
    try:
        with open(filename, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = (row.get("type") or "").strip().lower()
                if t not in ("large_airport", "medium_airport"):
                    continue
                try:
                    name = (row.get("name") or "").strip()
                    lat = float((row.get("latitude_deg") or "").strip())
                    lng = float((row.get("longitude_deg") or "").strip())
                    code_raw = (row.get("iata_code") or "").strip() or (row.get("ident") or "").strip() or (row.get("gps_code") or "").strip()
                    code = code_raw.upper() if code_raw else ""
                    display = f"{name} ({code})" if code else name
                    if name:
                        airports.append((display, (lat, lng)))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return airports


def build_kdtree(points: List[Tuple[str, Tuple[float, float]]]):
    """KD-tree over (lat, lng). Returns (tree, names, coords_array) or (None, [], None) if empty."""
    if not points:
        return None, [], None
    coords = np.array([[p[1][0], p[1][1]] for p in points], dtype=float)
    tree = cKDTree(coords)
    names = [p[0] for p in points]  # already includes codes in display
    return tree, names, coords


def nearest_point(tree, coords_array, names, target: Tuple[float, float]) -> Optional[Tuple[str, Tuple[float, float]]]:
    if tree is None or coords_array is None or not names:
        return None
    _, idx = tree.query([target[0], target[1]], k=1)
    name = names[idx]
    lat, lng = coords_array[idx]
    return (name, (float(lat), float(lng)))


# ---------- Ports from local GeoJSON (with codes in display name) ----------
def load_ports_from_geojson_local(path: str = "ports.geojson") -> List[Tuple[str, Tuple[float, float]]]:
    """
    Load ports from a local GeoJSON file at 'ports.geojson'.
    Returns list of (display_name_with_code, (lat, lng)) like 'Piombino (TPIO)'.
    Tries several property keys to find a port code.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local ports file '{path}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    ports: List[Tuple[str, Tuple[float, float]]] = []

    def _first_str(*vals) -> Optional[str]:
        for v in vals:
            if isinstance(v, str) and v.strip():
                return v.strip()
            # some datasets put a single code in a 1-item list
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], str) and v[0].strip():
                return v[0].strip()
        return None

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")  # [lon, lat]
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            continue

        lon, lat = coords
        name_raw = _first_str(props.get("name"), props.get("port_name"), props.get("Name"), "Port") or "Port"

        # Try multiple possible keys for the port code
        code_raw = _first_str(
            props.get("port"),
            props.get("id"),
            props.get("locode"),
            props.get("LOCODE"),
            props.get("code"),
            props.get("unlocs"),        # sometimes used in other datasets
        )

        code = code_raw.upper().replace(" ", "") if code_raw else ""

        # Always append code if present and not already present at the end
        if code and not name_raw.rstrip().endswith(f"({code})"):
            display = f"{name_raw} ({code})"
        else:
            display = name_raw

        ports.append((display, (float(lat), float(lon))))

    if not ports:
        raise ValueError(f"No valid port points found in '{path}'.")
    return ports


# ---------- Distance helpers ----------
def get_distance_matrix(origin, destination, api_key):
    """Road distance via Google Distance Matrix. Accepts address strings or 'lat,lng' strings."""
    origin_key = normalise_location_key(origin)
    dest_key = normalise_location_key(destination)
    key = (origin_key, dest_key)
    if key in distance_cache:
        return distance_cache[key]

    params = {
        'origins': to_api_location_param(origin),
        'destinations': to_api_location_param(destination),
        'key': api_key
    }
    data = requests.get(DISTANCE_MATRIX_API_ENDPOINT, params=params).json()

    distance, source = None, None
    try:
        el = data['rows'][0]['elements'][0]
        if el.get('status') == 'OK':
            txt = el['distance']['text']
            if ' km' in txt:
                distance = float(txt.replace(' km', '').replace(',', ''))
            elif ' m' in txt:
                distance = float(txt.replace(' m', '').replace(',', '')) / 1000.0
            source = "Google Maps API Shortest Road Distance"
    except Exception:
        pass

    distance_cache[key] = (distance, source)
    return distance, source


def road_km_between_latlng(a_latlng: Tuple[float, float], b_latlng: Tuple[float, float]) -> Optional[float]:
    """Road km between two (lat,lng) pairs using Distance Matrix; fallback to geodesic if needed."""
    if not a_latlng or not b_latlng:
        return None
    a = f"{a_latlng[0]},{a_latlng[1]}"
    b = f"{b_latlng[0]},{b_latlng[1]}"
    dist, _ = get_distance_matrix(a, b, API_KEY)
    if dist is None:
        try:
            return geodesic(a_latlng, b_latlng).kilometers
        except Exception:
            return None
    return dist


# ---------- Searoute wrappers ----------
def searoute_sea_km_between_ports(o_port_latlng: Tuple[float, float], d_port_latlng: Tuple[float, float]) -> float:
    """
    Use searoute between two known port coordinates.
    Returns sea distance in km (falls back to GC if length not present).
    """
    o_lonlat = [o_port_latlng[1], o_port_latlng[0]]
    d_lonlat = [d_port_latlng[1], d_port_latlng[0]]
    route = sr.searoute(o_lonlat, d_lonlat, units="naut")
    sea_km = None
    if route and getattr(route, "properties", None):
        props = route.properties
        if "length" in props:
            sea_km = float(props["length"]) * 1.852
    if sea_km is None:
        sea_km = geodesic((o_port_latlng[0], o_port_latlng[1]), (d_port_latlng[0], d_port_latlng[1])).kilometers
    return sea_km


# ---------- Arrow-safe preview ----------
def coerce_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make object-typed text columns Arrow-friendly by converting to Pandas 'string' dtype
    for UI display only (does not mutate the working df).
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]):
            try:
                pd.to_numeric(out[col])
            except Exception:
                out[col] = out[col].astype("string")
    return out


# ---------- Main processing ----------
def process_file(
    input_file,
    enable_hub_legs: bool,
) -> pd.DataFrame:

    df = pd.read_excel(input_file)

    # Decide output columns based on the hub toggle
    base_cols = ['Distance (km)', 'Source', 'Flight (km)', 'Sea (km)']
    hub_cols = [
        'Distance to hub (km)', 'Origin hub', 'Destination hub',
        'Distance from hub (km)', 'Distance to/from hub (km)'
    ]
    needed_cols = base_cols + (hub_cols if enable_hub_legs else [])
    for col in needed_cols:
        if col not in df.columns:
            df[col] = pd.NA

    # If hub legs are disabled, also drop any lingering hub columns from prior runs
    if not enable_hub_legs:
        for col in hub_cols:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    rows_to_process = df[df['Distance (km)'].isna()]
    if rows_to_process.empty:
        st.info("No new rows to process.")
        return df

    # Geocode warm-up
    unique_addresses = set()
    for _, r in rows_to_process.iterrows():
        unique_addresses.add(str(r['From']).strip())
        unique_addresses.add(str(r['To']).strip())

    st.write("Geocoding unique locations…")
    warmup_bar = st.progress(0)
    unique_list = list(unique_addresses)

    def _warmup(addr):
        _ = coords_for_location(addr, API_KEY)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(_warmup, a): i for i, a in enumerate(unique_list)}
        total = len(unique_list)
        done = 0
        for _ in concurrent.futures.as_completed(futs):
            done += 1
            warmup_bar.progress(done / max(total, 1))

    # Airports KD-tree (for air hubs)
    airports = load_international_airports("airports.csv")
    airport_tree, airport_names, airport_coords = build_kdtree(airports)

    # Ports KD-tree from local GeoJSON (for sea hubs) — strictly enforced when hub legs ON
    ports_tree = None
    port_names: List[str] = []
    port_coords_arr = None
    if enable_hub_legs:
        try:
            ports_list = load_ports_from_geojson_local("ports.geojson")
            ports_tree, port_names, port_coords_arr = build_kdtree(ports_list)
        except Exception as e:
            st.error(f"Required local 'ports.geojson' not available/valid: {e}")
            st.stop()  # Enforce: no processing without the local file when hub legs are enabled

    # Grouping
    tmp = rows_to_process.copy()
    tmp['group_key'] = tmp.apply(
        lambda x: (
            normalise_location_key(x.get('From')),
            normalise_location_key(x.get('To')),
            str(x.get('Mode', '')).strip().lower()
        ),
        axis=1
    )
    unique_groups = tmp['group_key'].unique()
    results_cache = {}

    # Core per-group computation
    def process_group(group):
        origin_key, destination_key, mode_lower = group
        origin_coords = coordinate_cache.get(origin_key) or coords_for_location(origin_key, API_KEY)
        destination_coords = coordinate_cache.get(destination_key) or coords_for_location(destination_key, API_KEY)

        # Outputs (initialise)
        main_distance = None    # Distance (km): main-mode only
        source = None

        # hub outputs (only populated if enable_hub_legs)
        road_to_hub = pd.NA
        origin_hub_label = pd.NA
        destination_hub_label = pd.NA
        road_from_hub = pd.NA
        road_sum_hub = pd.NA  # combined

        # QA legs
        flight_dist = pd.NA
        sea_dist = pd.NA

        try:
            if mode_lower in ["air", "airplane (air)", "flight"]:
                if origin_coords and destination_coords:
                    if enable_hub_legs and airport_tree is not None:
                        o_air = nearest_point(airport_tree, airport_coords, airport_names, origin_coords)
                        d_air = nearest_point(airport_tree, airport_coords, airport_names, destination_coords)
                        if o_air and d_air:
                            o_air_name, o_air_coords = o_air   # name already contains code
                            d_air_name, d_air_coords = d_air
                            road_to_hub = road_km_between_latlng(origin_coords, o_air_coords)
                            road_from_hub = road_km_between_latlng(d_air_coords, destination_coords)
                            flight_dist = geodesic(o_air_coords, d_air_coords).kilometers
                            origin_hub_label = o_air_name
                            destination_hub_label = d_air_name
                        else:
                            flight_dist = geodesic(origin_coords, destination_coords).kilometers
                    else:
                        flight_dist = geodesic(origin_coords, destination_coords).kilometers

                    main_distance = flight_dist
                    source = "Great Circle Distance"

            elif mode_lower in ["cargo ship (sea)", "sea", "ocean", "vessel (sea)"]:
                if origin_coords and destination_coords:
                    if enable_hub_legs:
                        # Enforced: ports_tree must exist (we stopped earlier if not)
                        o_port = nearest_point(ports_tree, port_coords_arr, port_names, origin_coords)
                        d_port = nearest_point(ports_tree, port_coords_arr, port_names, destination_coords)
                        if o_port and d_port:
                            o_port_name, o_port_latlng = o_port  # name already includes (CODE)
                            d_port_name, d_port_latlng = d_port
                            road_to_hub = road_km_between_latlng(origin_coords, o_port_latlng)
                            road_from_hub = road_km_between_latlng(d_port_latlng, destination_coords)
                            sea_dist = searoute_sea_km_between_ports(o_port_latlng, d_port_latlng)
                            origin_hub_label = o_port_name
                            destination_hub_label = d_port_name
                            main_distance = sea_dist
                            source = "Shortest sea route between two points on Earth."
                        else:
                            # If we can't find two ports, still compute main sea distance without hubs
                            sea_dist = geodesic(origin_coords, destination_coords).kilometers
                            main_distance = sea_dist
                            source = "Shortest sea route between two points on Earth."
                    else:
                        # Hub legs disabled → just compute sea main distance with searoute between points
                        route = sr.searoute([origin_coords[1], origin_coords[0]], [destination_coords[1], destination_coords[0]], units="naut")
                        if route and getattr(route, "properties", None) and "length" in route.properties:
                            sea_dist = float(route.properties["length"]) * 1.852
                        else:
                            sea_dist = geodesic(origin_coords, destination_coords).kilometers
                        main_distance = sea_dist
                        source = "Shortest sea route between two points on Earth."
                else:
                    main_distance = None
                    source = None

            elif mode_lower in ["truck (road)", "trucking", "courier", "van", "truck", "car"]:
                origin_api = origin_key if ',' in origin_key else to_api_location_param(origin_key)
                dest_api = destination_key if ',' in destination_key else to_api_location_param(destination_key)
                dist, src = get_distance_matrix(origin_api, dest_api, API_KEY)
                if dist is None and origin_coords and destination_coords:
                    dist = geodesic(origin_coords, destination_coords).kilometers
                    src = "Geodesic Distance"
                main_distance = dist
                source = "Google Maps API Shortest Road Distance" if src == "Google Maps API Shortest Road Distance" else src

            else:
                # default: treat like road
                origin_api = origin_key if ',' in origin_key else to_api_location_param(origin_key)
                dest_api = destination_key if ',' in destination_key else to_api_location_param(destination_key)
                dist, src = get_distance_matrix(origin_api, dest_api, API_KEY)
                if dist is not None:
                    main_distance = dist
                    source = "Google Maps API Shortest Road Distance"
                elif origin_coords and destination_coords:
                    main_distance = geodesic(origin_coords, destination_coords).kilometers
                    source = "Geodesic Distance"

            # Sum the two road legs into the combined column (only when hubs enabled and at least one leg present)
            if enable_hub_legs:
                to_val = float(road_to_hub) if isinstance(road_to_hub, (int, float)) and road_to_hub is not pd.NA else None
                from_val = float(road_from_hub) if isinstance(road_from_hub, (int, float)) and road_from_hub is not pd.NA else None
                if to_val is not None or from_val is not None:
                    road_sum_hub = (to_val or 0.0) + (from_val or 0.0)

        except Exception as e:
            print(f"Error in group {group}: {e}")

        # Build row dict conditioned on hub toggle
        row_out = {
            'Distance (km)': main_distance,
            'Source': source,
            'Flight (km)': flight_dist,
            'Sea (km)': sea_dist
        }
        if enable_hub_legs:
            row_out.update({
                'Distance to hub (km)': road_to_hub,
                'Origin hub': origin_hub_label,
                'Destination hub': destination_hub_label,
                'Distance from hub (km)': road_from_hub,
                'Distance to/from hub (km)': road_sum_hub
            })
        return group, row_out

    # Run with progress bar
    st.write("Calculating distances…")
    progress_bar = st.progress(0)
    results_cache.clear()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(process_group, g): g for g in unique_groups}
        total = len(unique_groups)
        done = 0
        for f in concurrent.futures.as_completed(futs):
            g, res = f.result()
            results_cache[g] = res
            done += 1
            progress_bar.progress(done / max(total, 1))

    # Write results back using original row indices
    for idx, row in rows_to_process.iterrows():
        gk = (
            normalise_location_key(row.get('From')),
            normalise_location_key(row.get('To')),
            str(row.get('Mode', '')).strip().lower()
        )
        res = results_cache.get(gk, {})
        for k, v in res.items():
            df.at[idx, k] = v

    # Clean helper col if present
    if 'group_key' in df.columns:
        df.drop(columns=['group_key'], inplace=True, errors='ignore')

    return df


def main():
    st.title("Distance Calculation")
    st.caption("Automatically calculates road, air, and sea transport distances between locations for GHG emissions inventory calculations.")

    if not check_password():
        return
        
    with st.expander("📘 How to use this tool", expanded=False):
        st.markdown("""
        **1️⃣ Upload your file**
    
        - Upload an Excel file containing at least three columns:  
          `From` (origin), `To` (destination), and `Mode` (e.g. *Road*, *Air*, *Sea*).  
        - Each row represents one transport route.
    
        **2️⃣ Choose options**
    
        - Optionally enable *“nearest airport/port legs”* to include the road distance
          to and from the nearest hubs.  
        - When enabled, additional columns will appear for:
          - `Origin hub`, `Destination hub`
          - `Distance to hub (km)`, `Distance from hub (km)`
          - `Distance to/from hub (km)` (sum of both)
    
        **3️⃣ Run the calculations**
    
        - Click “Process file” (this happens automatically after upload).  
        - Watch the progress bars for geocoding and distance computation.
    
        **4️⃣ Review and download**
    
        - A preview of your processed data will appear below.  
        - Use the **Download Processed File** button to export an `.xlsx` file.
    
        **🗺️ Notes**
        - Road distances use **Google Maps API** (shortest route).
        - Air distances use **Great Circle** calculations between airports.
        - Sea distances use **Searoute shortest sea paths** between ports.
        - All distances are in **kilometres (km)**.
        - You may still want to modify these distances - e.g., trucking may not always take the shortest route so you may want to increase distances by 10% to compensate for this.
    """)
        
    with st.expander("🧭 Valid 'Mode' entries", expanded=False):
        st.markdown("""
        The **`Mode`** column determines how each route is calculated.  
        Accepted values are case-insensitive.
    
        **🚛 Road transport**
        ```
        road
        truck
        truck (road)
        trucking
        courier
        van
        car
        ```
        → Uses **Google Maps Distance Matrix** (shortest road route).  
        → Falls back to straight-line (geodesic) distance if API fails.  
        **Source label:** *Google Maps API Shortest Road Distance*
    
        **✈️ Air transport**
        ```
        air
        airplane (air)
        flight
        ```
        → Uses nearest airports and computes **Great Circle distance**.  
        → Road legs to/from airports added if hub option enabled.  
        **Source label:** *Great Circle Distance*
    
        **🚢 Sea transport**
        ```
        sea
        cargo ship (sea)
        ocean
        vessel (sea)
        ```
        → Uses nearest seaports from local `ports.geojson`.  
        → Calculates **shortest sea route** using Searoute.  
        → Road legs to/from ports added if hub option enabled.  
        **Source label:** *Shortest sea route between two points on Earth.*
    
        **⚙️ Fallback**
        - If left blank or unrecognised, defaults to *road* mode.
        """)
    
    # Toggle for hub legs
    enable_hub_legs = st.checkbox(
        "Enable nearest airport/port legs (adds road distances to closest airport/seaport in seperate columns)", value=False
    )

    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    if uploaded_file:
        input_file = BytesIO(uploaded_file.read())
        df = pd.read_excel(input_file)

        # UI-safe preview
        st.write("Input data preview:")
        st.dataframe(coerce_arrow_safe(df.head()))

        st.write("Processing file…")
        processed_df = process_file(
            input_file,
            enable_hub_legs=enable_hub_legs
        )

        if enable_hub_legs:
            st.caption("Ports are selected from local 'ports.geojson' (codes included in hub names). No fallback is used.")

        st.write("Processed data preview:")
        st.dataframe(coerce_arrow_safe(processed_df.head()))

        # Download
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
