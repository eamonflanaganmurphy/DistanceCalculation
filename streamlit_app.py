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
                    code = (row.get("iata_code") or "").strip() or (row.get("ident") or "").strip() or (row.get("gps_code") or "").strip()
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


# ---------- Ports from GeoJSON (with codes in display name) ----------
@st.cache_data(show_spinner=False)
def load_ports_from_geojson(source: str) -> List[Tuple[str, Tuple[float, float]]]:
    """
    Load ports from a GeoJSON file (URL or local path).
    Returns list of (display_name_with_code, (lat, lng)) like 'Prague (CZPRA)'.
    """
    text = None
    if source.startswith("http://") or source.startswith("https://"):
        r = requests.get(source, timeout=30)
        r.raise_for_status()
        text = r.text
    else:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()

    data = json.loads(text)
    features = data.get("features", [])
    ports: List[Tuple[str, Tuple[float, float]]] = []
    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")  # [lon, lat]
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            continue
        lon, lat = coords
        name = (props.get("name") or props.get("port_name") or "Port").strip()
        code = (props.get("port_id") or props.get("id") or "").strip()
        display = f"{name} ({code})" if code else name
        ports.append((display, (float(lat), float(lon))))
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


def searoute_with_ports(origin_coords: Tuple[float, float],
                        destination_coords: Tuple[float, float],
                        prefer_auto_ports: bool,
                        origin_port_override: Optional[str],
                        destination_port_override: Optional[str]):
    """
    Use searoute to compute sea path and extract chosen ports + their coordinates.
    Returns:
      sea_km: float
      o_port_id: Optional[str]
      d_port_id: Optional[str]
      o_port_latlng: Optional[Tuple[float,float]]  # (lat, lng)
      d_port_latlng: Optional[Tuple[float,float]]
    """
    o_lonlat = [origin_coords[1], origin_coords[0]]  # [lon, lat]
    d_lonlat = [destination_coords[1], destination_coords[0]]

    origin_feat = o_lonlat
    dest_feat = d_lonlat
    if _HAS_AREAFEATURE:
        o_pref = PortProps(origin_port_override, 1) if (origin_port_override and origin_port_override.strip()) else None
        d_pref = PortProps(destination_port_override, 1) if (destination_port_override and destination_port_override.strip()) else None
        if prefer_auto_ports and not o_pref:
            o_pref = []
        if prefer_auto_ports and not d_pref:
            d_pref = []
        origin_feat = AreaFeature(coords=o_lonlat, name="Origin", preferred_ports=o_pref if o_pref is not None else None)
        dest_feat = AreaFeature(coords=d_lonlat, name="Destination", preferred_ports=d_pref if d_pref is not None else None)

    route = sr.searoute(origin_feat, dest_feat, units="naut")

    sea_km = None
    o_port_id = None
    d_port_id = None
    o_port_latlng = None
    d_port_latlng = None

    # Distance / metadata
    if route and getattr(route, "properties", None):
        props = route.properties
        if "length" in props:
            sea_km = float(props["length"]) * 1.852  # nmi -> km

        # Port IDs (try several shapes)
        o_port = props.get("origin_port") or props.get("start_port") or {}
        d_port = props.get("destination_port") or props.get("end_port") or {}
        o_port_id = (o_port.get("id") if isinstance(o_port, dict) else None) or props.get("origin_port_id") or props.get("start_port_id")
        d_port_id = (d_port.get("id") if isinstance(d_port, dict) else None) or props.get("destination_port_id") or props.get("end_port_id")

    # Extract port coordinates from geometry (LineString first/last point)
    try:
        geom = getattr(route, "geometry", None)
        if isinstance(geom, dict):
            coords = geom.get("coordinates")
        else:
            coords = getattr(geom, "coordinates", None)
        if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            o_lon, o_lat = coords[0]
            d_lon, d_lat = coords[-1]
            o_port_latlng = (float(o_lat), float(o_lon))
            d_port_latlng = (float(d_lat), float(d_lon))
    except Exception:
        pass

    # Fallback for distance if needed
    if sea_km is None:
        sea_km = geodesic((o_lonlat[1], o_lonlat[0]), (d_lonlat[1], d_lonlat[0])).kilometers

    return sea_km, o_port_id, d_port_id, o_port_latlng, d_port_latlng


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
    use_geojson_ports: bool,
    ports_geojson_source: Optional[str],
    use_searoute_preferred_ports: bool,
    origin_port_override: Optional[str],
    destination_port_override: Optional[str],
) -> pd.DataFrame:

    df = pd.read_excel(input_file)

    # Ensure output columns
    needed_cols = [
        'Distance (km)', 'Source',
        'Distance to hub (km)', 'Origin hub', 'Destination hub', 'Distance from hub (km)',
        'Flight (km)', 'Sea (km)'
    ]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = pd.NA

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

    # Airports KD-tree
    airports = load_international_airports("airports.csv")
    airport_tree, airport_names, airport_coords = build_kdtree(airports)

    # Ports KD-tree from GeoJSON (optional)
    ports_tree = None
    port_names: List[str] = []
    port_coords_arr = None
    if enable_hub_legs and use_geojson_ports and ports_geojson_source:
        try:
            ports_list = load_ports_from_geojson(ports_geojson_source)
            ports_tree, port_names, port_coords_arr = build_kdtree(ports_list)
        except Exception as e:
            st.warning(f"Could not load ports from GeoJSON source ({e}). Falling back to searoute’s internal ports.")
            ports_tree, port_names, port_coords_arr = None, [], None

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

        # unified hub outputs
        road_to_hub = pd.NA
        origin_hub_label = pd.NA
        destination_hub_label = pd.NA
        road_from_hub = pd.NA

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
                    if enable_hub_legs and ports_tree is not None:
                        # Use GeoJSON ports to pick nearest ports; then route sea between them
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
                            # fallback to searoute’s internal port selection
                            sea_km, o_id, d_id, o_pll, d_pll = searoute_with_ports(
                                origin_coords,
                                destination_coords,
                                prefer_auto_ports=True,
                                origin_port_override=None,
                                destination_port_override=None
                            )
                            sea_dist = sea_km
                            if o_pll and d_pll:
                                road_to_hub = road_km_between_latlng(origin_coords, o_pll)
                                road_from_hub = road_km_between_latlng(d_pll, destination_coords)
                            origin_hub_label = f"{o_id}" if o_id else pd.NA
                            destination_hub_label = f"{d_id}" if d_id else pd.NA
                            main_distance = sea_dist
                            # keep generic wording for internal fallback
                            source = "Sea (searoute)"
                    else:
                        # No GeoJSON port KD-tree → use searoute internal ports or just searoute
                        sea_km, o_id, d_id, o_pll, d_pll = searoute_with_ports(
                            origin_coords,
                            destination_coords,
                            prefer_auto_ports=enable_hub_legs and use_searoute_preferred_ports,
                            origin_port_override=origin_port_override,
                            destination_port_override=destination_port_override
                        )
                        sea_dist = sea_km
                        if enable_hub_legs and o_pll and d_pll:
                            road_to_hub = road_km_between_latlng(origin_coords, o_pll)
                            road_from_hub = road_km_between_latlng(d_pll, destination_coords)
                        origin_hub_label = f"{o_id}" if o_id else pd.NA
                        destination_hub_label = f"{d_id}" if d_id else pd.NA
                        main_distance = sea_dist
                        source = "Sea (searoute)"
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
                # replace label if API worked
                source = src if src != "Google Maps API Shortest Road Distance" else "Google Maps API Shortest Road Distance"

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

        except Exception as e:
            print(f"Error in group {group}: {e}")

        return group, {
            'Distance (km)': main_distance,
            'Source': source,
            'Distance to hub (km)': road_to_hub,
            'Origin hub': origin_hub_label,
            'Destination hub': destination_hub_label,
            'Distance from hub (km)': road_from_hub,
            'Flight (km)': flight_dist,
            'Sea (km)': sea_dist
        }

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
    st.write("Upload an Excel file with headers **From**, **To**, and **Mode**.")

    if not check_password():
        return

    # Options
    enable_hub_legs = st.checkbox(
        "Enable nearest airport/port legs (adds hub columns; NOT counted in Distance (km))", value=False
    )

    use_geojson_ports = False
    ports_geojson_source = None
    use_searoute_preferred_ports = False
    origin_port_override = None
    destination_port_override = None

    if enable_hub_legs:
        st.caption("Hub columns show road legs to/from the selected hub and the hub names with codes (e.g., 'Prague (CZPRA)').")

        # Option 1: Use ports.geojson
        use_geojson_ports = st.checkbox(
            "Use ports.geojson to pick nearest seaports",
            value=True,
            help="Loads ports from a GeoJSON (URL or local path) and chooses the nearest ports to origin/destination."
        )
        if use_geojson_ports:
            ports_geojson_source = st.text_input(
                "ports.geojson URL or local path",
                value="https://raw.githubusercontent.com/genthalili/searoute-py/main/searoute/data/ports.geojson"
            ).strip()

        # Option 2: fall back to searoute internal ports
        if not use_geojson_ports:
            if not _HAS_AREAFEATURE:
                st.info("Your installed searoute version does not expose AreaFeature/PortProps; using direct coordinates only.")
            else:
                use_searoute_preferred_ports = st.checkbox(
                    "Use searoute preferred ports (auto select)",
                    value=True,
                    help="Let searoute pick suitable ports from its internal database."
                )
                with st.expander("Optional: override port IDs (e.g., USNYC)"):
                    origin_port_override = st.text_input("Origin port ID (optional)", value="").strip() or None
                    destination_port_override = st.text_input("Destination port ID (optional)", value="").strip() or None

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
            enable_hub_legs=enable_hub_legs,
            use_geojson_ports=use_geojson_ports,
            ports_geojson_source=ports_geojson_source,
            use_searoute_preferred_ports=use_searoute_preferred_ports,
            origin_port_override=origin_port_override,
            destination_port_override=destination_port_override
        )

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
