from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import math
import asyncio
import os
from datetime import datetime, timezone, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

SYNOPTIC_TOKEN = os.environ.get("SYNOPTIC_TOKEN", "")

# Simple in-memory cache — keyed by rounded lat/lon, expires after 30 min
_cache = {}
CACHE_TTL = timedelta(minutes=30)

def cache_key(lat, lon):
    return f"{round(lat,2)},{round(lon,2)}"

def cache_get(key):
    entry = _cache.get(key)
    if entry and datetime.now(timezone.utc) - entry["ts"] < CACHE_TTL:
        return entry["data"]
    return None

def cache_set(key, data):
    _cache[key] = {"ts": datetime.now(timezone.utc), "data": data}

def wind_uv(spd, direction):
    r = math.radians(direction)
    return -spd * math.sin(r), -spd * math.cos(r)

def calc_lcl(temp_c, dew_c):
    return max(0, 125 * (temp_c - dew_c))

def calc_shear(spd_lo, dir_lo, spd_hi, dir_hi):
    u1, v1 = wind_uv(spd_lo, dir_lo)
    u2, v2 = wind_uv(spd_hi, dir_hi)
    return math.sqrt((u2 - u1)**2 + (v2 - v1)**2)

def calc_srh(spd_sfc, dir_sfc, spd_850, dir_850):
    u0, v0 = wind_uv(spd_sfc, dir_sfc)
    u8, v8 = wind_uv(spd_850, dir_850)
    um, vm = (u0 + u8) / 2, (v0 + v8) / 2
    mag = math.sqrt((u8 - u0)**2 + (v8 - v0)**2) or 0.001
    cx = um + 7.5 * (v8 - v0) / mag
    cy = vm - 7.5 * (u8 - u0) / mag
    return abs((u0 - cx) * (v8 - cy) - (u8 - cx) * (v0 - cy))

def calc_convergence(stations, lat, lon):
    near = [s for s in stations if math.sqrt((s["lat"] - lat)**2 + (s["lon"] - lon)**2) < 1.5]
    if len(near) < 3:
        return None
    dudx, dvdy, n = 0, 0, 0
    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            dx = (near[j]["lon"] - near[i]["lon"]) * 111 * math.cos(math.radians(lat))
            dy = (near[j]["lat"] - near[i]["lat"]) * 111
            if abs(dx) > 0.1:
                dudx += (near[j]["u"] - near[i]["u"]) / dx
                n += 1
            if abs(dy) > 0.1:
                dvdy += (near[j]["v"] - near[i]["v"]) / dy
    return -(dudx / n + dvdy / n) * 1000 if n > 0 else None

async def fetch_hrrr(lat, lon):
    # Single API call — combine all fields into one request to minimize quota usage
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,dewpoint_2m,cape,cin,"
        f"windspeed_10m,winddirection_10m,"
        f"windspeed_80m,winddirection_80m,"
        f"windspeed_180m,winddirection_180m"
        f"&wind_speed_unit=kn&temperature_unit=celsius&forecast_days=1&timezone=auto"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(500, f"Surface API error: {r.text[:200]}")
        return r.json()

async def fetch_synoptic(lat, lon):
    if not SYNOPTIC_TOKEN:
        return []
    url = (
        f"https://api.synopticdata.com/v2/stations/nearesttime"
        f"?token={SYNOPTIC_TOKEN}&within=60&radius={lat},{lon}"
        f"&vars=wind_speed,wind_direction,dew_point_temperature&units=speed|kts&recent=60"
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url)
            if r.status_code != 200:
                return []
            data = r.json()
            stations = []
            for s in data.get("STATION", []):
                obs = s.get("OBSERVATIONS", {})
                spd = obs.get("wind_speed_value_1", {}).get("value", 0) or 0
                direction = obs.get("wind_direction_value_1", {}).get("value", 0) or 0
                dew = obs.get("dew_point_temperature_value_1", {}).get("value")
                u, v = wind_uv(spd, direction)
                stations.append({
                    "lat": float(s["LATITUDE"]),
                    "lon": float(s["LONGITUDE"]),
                    "u": u, "v": v, "dew": dew
                })
            return stations
    except Exception:
        return []

def get_current_hour_idx(hourly_times):
    now = datetime.now(timezone.utc)
    best = 0
    for i, t in enumerate(hourly_times):
        try:
            dt = datetime.fromisoformat(t.replace("Z", "+00:00"))
            if dt <= now:
                best = i
        except Exception:
            pass
    return best

@app.get("/score")
async def score(lat: float, lon: float):
    key = cache_key(lat, lon)
    cached = cache_get(key)
    if cached:
        cached["cached"] = True
        return cached

    wx, syn = await asyncio.gather(fetch_hrrr(lat, lon), fetch_synoptic(lat, lon))

    h = wx["hourly"]
    idx = get_current_hour_idx(h["time"])

    temp_c = (h.get("temperature_2m") or [0])[idx] or 0
    dew_c  = (h.get("dewpoint_2m")    or [0])[idx] or 0
    cape   = (h.get("cape")           or [0])[idx] or 0
    cin    = (h.get("cin")            or [0])[idx] or 0
    spd10  = (h.get("windspeed_10m")  or [0])[idx] or 0
    dir10  = (h.get("winddirection_10m") or [0])[idx] or 0
    spd80  = (h.get("windspeed_80m")  or [0])[idx] or 0
    dir80  = (h.get("winddirection_80m") or [0])[idx] or 0
    spd180 = (h.get("windspeed_180m") or [0])[idx] or 0
    dir180 = (h.get("winddirection_180m") or [0])[idx] or 0

    lcl   = calc_lcl(temp_c, dew_c)
    shear = calc_shear(spd10, dir10, spd180, dir180)
    srh   = calc_srh(spd10, dir10, spd80, dir80)

    conv_val = None
    conv_src = "estimated"
    sta_dew  = None

    if len(syn) >= 3:
        conv_val = calc_convergence(syn, lat, lon)
        if conv_val is not None:
            conv_src = "live"
        dew_vals = [s["dew"] for s in syn if s["dew"] is not None]
        if dew_vals:
            sta_dew = sum(dew_vals) / len(dew_vals)

    if conv_val is None:
        u, v = wind_uv(spd10, dir10)
        conv_val = (abs(u) + abs(v)) * 0.3

    final_dew_c = sta_dew if sta_dew is not None else dew_c
    final_dew_f = final_dew_c * 9/5 + 32

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cached": False,
        "params": {
            "cape":       {"value": round(cape),           "unit": "J/kg",   "src": "live"},
            "cin":        {"value": round(cin),            "unit": "J/kg",   "src": "live"},
            "lcl":        {"value": round(lcl),            "unit": "m",      "src": "live"},
            "shear_06km": {"value": round(shear, 1),       "unit": "kts",    "src": "est"},
            "srh_01km":   {"value": round(srh, 1),         "unit": "m2/s2",  "src": "est"},
            "dewpoint":   {"value": round(final_dew_f, 1), "unit": "F",      "src": "live" if sta_dew else "live"},
            "convergence":{"value": round(conv_val, 2),    "unit": "",       "src": conv_src},
            "boundary":   {"value": "likely" if cape > 300 and cin > -150 else "unlikely", "unit": "", "src": "live"},
        }
    }
    cache_set(key, result)
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "cache_entries": len(_cache)}
