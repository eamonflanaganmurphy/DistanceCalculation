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
from typing import Optional, Tuple, List

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
distance_cache = {}      # {(origin_norm, destination_norm): (distance_km, source)}
coordinate_cache = {}    # {location_key: (lat, lng)}

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


def load_international_airports(filename="airports.csv") -> List[Tuple[str, Tuple[float, float]]]:
    """Read airports (expects OpenFlights-like columns). Uses large/medium airports only."""
    airports = []
    try:
        with open(filename, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("type", "").strip().lower()
                if t not in ("large_airport", "medium_airport"):
                    continue
                try:
                    name = row["name"].strip()
                    lat = float(row["latitude_deg"].strip())
                    lng = float(row["longitude_deg"].strip())
                    if name:
                        airports.append((name, (lat, lng)))
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
    names = [p[0] for p in points]
    return tree, names, coords


def nearest_point(tree, coords_array, names, target: Tuple[float, float]) -> Optional[Tuple[str, Tuple[float, float]]]:
    if tree is None or coords_array is None or not names:
        return None
    _, idx = tree.query([target[0], target[1]], k=1)
    name = names[idx]
    lat, lng = coords_array[idx]
    return (name, (float(lat), float(lng)))


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
            source = "Road Distance"
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


def coerce_arrow_safe(df: pd.DataFrame) -> pd
