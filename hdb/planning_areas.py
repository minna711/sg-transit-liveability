from __future__ import annotations
import sqlite3
from pathlib import Path

DB_PATH = Path("data/transport.db")

PLANNING_AREAS = {
    "ANG MO KIO":              (103.820, 103.880, 1.355, 1.395),
    "BEDOK":                   (103.890, 103.960, 1.300, 1.345),
    "BISHAN":                  (103.820, 103.865, 1.335, 1.365),
    "BOON LAY":                (103.690, 103.730, 1.330, 1.360),
    "BUKIT BATOK":             (103.730, 103.780, 1.340, 1.380),
    "BUKIT MERAH":             (103.790, 103.840, 1.265, 1.300),
    "BUKIT PANJANG":           (103.750, 103.800, 1.360, 1.400),
    "BUKIT TIMAH":             (103.770, 103.830, 1.310, 1.355),
    "CENTRAL WATER CATCHMENT": (103.790, 103.850, 1.370, 1.420),
    "CHANGI":                  (103.960, 104.010, 1.340, 1.390),
    "CHANGI BAY":              (103.970, 104.030, 1.300, 1.340),
    "CHOA CHU KANG":           (103.730, 103.780, 1.375, 1.415),
    "CLEMENTI":                (103.740, 103.800, 1.295, 1.335),
    "DOWNTOWN CORE":           (103.840, 103.870, 1.270, 1.295),
    "GEYLANG":                 (103.870, 103.920, 1.305, 1.335),
    "HOUGANG":                 (103.860, 103.910, 1.355, 1.395),
    "JURONG EAST":             (103.720, 103.760, 1.315, 1.355),
    "JURONG WEST":             (103.680, 103.730, 1.330, 1.370),
    "KALLANG":                 (103.850, 103.900, 1.295, 1.325),
    "LIM CHU KANG":            (103.690, 103.760, 1.410, 1.460),
    "MANDAI":                  (103.780, 103.840, 1.400, 1.440),
    "MARINA EAST":             (103.860, 103.900, 1.275, 1.300),
    "MARINA SOUTH":            (103.850, 103.890, 1.255, 1.280),
    "MARINE PARADE":           (103.893, 103.935, 1.295, 1.316),
    "MUSEUM":                  (103.845, 103.865, 1.290, 1.305),
    "NEWTON":                  (103.820, 103.850, 1.305, 1.325),
    "NORTH-EASTERN ISLANDS":   (103.960, 104.030, 1.390, 1.450),
    "NOVENA":                  (103.820, 103.850, 1.320, 1.345),
    "ORCHARD":                 (103.815, 103.845, 1.295, 1.315),
    "OUTRAM":                  (103.830, 103.860, 1.270, 1.295),
    "PASIR RIS":               (103.930, 103.980, 1.355, 1.395),
    "PAYA LEBAR":              (103.890, 103.930, 1.310, 1.340),
    "PIONEER":                 (103.680, 103.720, 1.295, 1.335),
    "PUNGGOL":                 (103.890, 103.940, 1.385, 1.420),
    "QUEENSTOWN":              (103.780, 103.830, 1.280, 1.315),
    "RIVER VALLEY":            (103.820, 103.850, 1.285, 1.310),
    "ROCHOR":                  (103.845, 103.870, 1.295, 1.315),
    "SELETAR":                 (103.860, 103.910, 1.395, 1.430),
    "SEMBAWANG":               (103.790, 103.850, 1.430, 1.470),
    "SENGKANG":                (103.870, 103.920, 1.375, 1.410),
    "SERANGOON":               (103.855, 103.895, 1.340, 1.375),
    "SIMPANG":                 (103.860, 103.920, 1.420, 1.460),
    "SINGAPORE RIVER":         (103.835, 103.865, 1.278, 1.300),
    "SOUTHERN ISLANDS":        (103.790, 103.870, 1.190, 1.240),
    "STRAITS VIEW":            (103.820, 103.860, 1.255, 1.278),
    "SUNGEI KADUT":            (103.740, 103.800, 1.400, 1.440),
    "TAMPINES":                (103.920, 103.970, 1.340, 1.380),
    "TANGLIN":                 (103.795, 103.830, 1.295, 1.325),
    "TENGAH":                  (103.720, 103.760, 1.360, 1.390),
    "TOA PAYOH":               (103.835, 103.875, 1.325, 1.355),
    "TUAS":                    (103.610, 103.670, 1.270, 1.340),
    "WESTERN ISLANDS":         (103.700, 103.780, 1.180, 1.230),
    "WESTERN WATER CATCHMENT": (103.680, 103.760, 1.370, 1.430),
    "WOODLANDS":               (103.770, 103.830, 1.420, 1.460),
    "YISHUN":                  (103.820, 103.870, 1.405, 1.445),
}

def init_table():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS planning_areas (name TEXT PRIMARY KEY, min_lon REAL, max_lon REAL, min_lat REAL, max_lat REAL)""")
    conn.commit(); conn.close()

def seed_planning_areas():
    init_table()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM planning_areas")
    for name, (a,b,c,d) in PLANNING_AREAS.items():
        conn.execute("INSERT OR REPLACE INTO planning_areas VALUES (?,?,?,?,?)", (name,a,b,c,d))
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM planning_areas").fetchone()[0]
    conn.close()
    print(f"Seeded {count} planning areas!")

def load_all_planning_areas():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT name,min_lon,max_lon,min_lat,max_lat FROM planning_areas ORDER BY name").fetchall()
    conn.close()
    return [{"name":r[0],"min_lon":r[1],"max_lon":r[2],"min_lat":r[3],"max_lat":r[4]} for r in rows]

def get_bbox_for_area(name):
    conn = sqlite3.connect(str(DB_PATH))
    row = conn.execute("SELECT min_lon,max_lon,min_lat,max_lat FROM planning_areas WHERE name=?", (name.upper(),)).fetchone()
    conn.close()
    return tuple(row) if row else None

if __name__ == "__main__":
    seed_planning_areas()
    for a in load_all_planning_areas():
        print(f"  {a['name']:35s} ({a['min_lon']:.3f}-{a['max_lon']:.3f}, {a['min_lat']:.3f}-{a['max_lat']:.3f})")
