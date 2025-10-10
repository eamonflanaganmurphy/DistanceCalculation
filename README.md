# 🌍 Distance Calculation Tool (Streamlit App)

This repository hosts a **Streamlit-based distance calculation tool** developed to automate transport distance estimation across road, air, and sea routes. It integrates multiple data sources (Google Maps API, Searoute, airports and ports datasets) to support **Scope 3 logistics calculations** and other sustainability assessments.

---

## ✨ Key Features

- **Automatic distance calculation** between any two locations.
- Supports **multiple transport modes**:
  - 🛣️ Road — via **Google Maps Distance Matrix API**
  - ✈️ Air — via **Great Circle distance** between nearest airports
  - 🚢 Sea — via **Searoute shortest sea path** between nearest seaports
- Optional calculation of **hub distances** (road legs to/from airports or seaports).
- Reads and writes standard **Excel (.xlsx)** files.
- Includes **progress bars** for large data processing.
- Fully secured with **password access** (set in `secrets.toml`).

---

## 🗂️ Repository Structure

```
distance-calculation-app/
│
├── streamlit_app.py           # Main Streamlit app
├── airports.csv               # Reference airport dataset (IATA/ICAO)
├── ports.geojson              # Local seaport database (required)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .streamlit/
    └── secrets.toml           # Stores API key and password (not in Git)
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/distance-calculation-app.git
cd distance-calculation-app
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # on macOS/Linux
venv\Scripts\activate      # on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Required Secrets

Create a `.streamlit/secrets.toml` file with the following structure:

```toml
google_api_key = "YOUR_GOOGLE_MAPS_API_KEY"
password = "YOUR_APP_PASSWORD"
```

> **Note:** You must enable the **Distance Matrix API** and **Geocoding API** in your Google Cloud project and restrict the key to your organisation’s domain if applicable.

---

## 🚀 Running the App

From the project directory, run:
```bash
streamlit run streamlit_app.py
```

Then open the URL displayed in your terminal (typically [http://localhost:8501](http://localhost:8501)).

---

## 🧭 How to Use

1. **Login:** Enter the app password (from `secrets.toml`).
2. **Upload an Excel file** containing at least these columns:
   - `From` (origin address or coordinates)
   - `To` (destination address or coordinates)
   - `Mode` (e.g. “Road”, “Air”, “Sea”)
3. **Optionally enable “nearest hub legs”**:
   - Adds road legs to/from the nearest airport or seaport.
   - Displays origin/destination hubs (e.g. “Piombino (TPIO)”).
4. Wait for progress bars to complete processing.
5. **Preview results** in the app or **download** as an `.xlsx` file.

---

## 📊 Output Columns

| Column | Description |
|---------|-------------|
| Distance (km) | Total distance for the selected mode only (excludes road legs) |
| Source | Method/source used (e.g. Google Maps, Great Circle, Searoute) |
| Distance to hub (km) | Road distance from origin to nearest hub |
| Origin hub | Name and code of nearest airport/port (e.g. “Piombino (TPIO)”) |
| Destination hub | Name and code of nearest destination hub |
| Distance from hub (km) | Road distance from destination hub to endpoint |
| Distance to/from hub (km) | Sum of the two road leg distances |
| Flight (km) | Great Circle distance between airports |
| Sea (km) | Sea route distance between ports |

---

## 🔒 Data Handling and Privacy

- No user data is stored; all processing occurs locally.
- API requests are sent directly to Google Maps and Searoute.
- Sensitive credentials (API key, passwords) must remain in `secrets.toml` and never be committed to Git.

---

## 🧠 Notes and Limitations

- Google API quotas apply (Distance Matrix and Geocoding).
- Sea distance computation relies on **port-to-port shortest route**; accuracy depends on `ports.geojson` resolution.
- The app is optimised for **corporate sustainability applications** (e.g., Scope 3.9 “Downstream Transportation”).

---

## 👥 Authors

Developed by Eamon Murphy

---

## 📜 Licence

This software is intended for internal project use only.
