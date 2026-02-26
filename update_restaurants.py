\
import re
import time
from urllib.parse import urlparse

import httpx
import pandas as pd

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"

def norm_url(u: str) -> str | None:
    if not u:
        return None
    u = u.strip()
    if not u:
        return None
    if u.startswith("www."):
        u = "https://" + u
    if not re.match(r"^https?://", u, flags=re.I):
        return None
    p = urlparse(u)
    if not p.netloc:
        return None
    return f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")

def norm_domain(u: str | None) -> str | None:
    if not u:
        return None
    return urlparse(u).netloc.lower()

def fetch_wikidata_moscow_food() -> pd.DataFrame:
    """
    Restaurants/cafes/fast food in Moscow with official website and (if present) opening hours.
    Note: Wikidata opening hours coverage is limited; OSM is usually richer.
    """
    sparql = """
    SELECT
      ?item ?itemLabel ?website ?coord
      (GROUP_CONCAT(DISTINCT ?catLabel; separator="; ") AS ?category)
      (GROUP_CONCAT(DISTINCT ?ohRow; separator=" | ") AS ?opening_hours)
    WHERE {
      VALUES ?cls { wd:Q11707 wd:Q30022 wd:Q274332 wd:Q1778821 }
      ?item wdt:P31/wdt:P279* ?cls .
      ?item wdt:P131* wd:Q649 .

      OPTIONAL { ?item wdt:P856 ?website . }
      OPTIONAL { ?item wdt:P625 ?coord . }

      OPTIONAL {
        ?item wdt:P31 ?cat .
        ?cat rdfs:label ?catLabel FILTER(LANG(?catLabel) IN ("ru","en"))
      }

      OPTIONAL {
        ?item p:P3025 ?st .
        ?st ps:P3025 ?daysItem .
        OPTIONAL { ?st pq:P8626 ?openTime . }
        OPTIONAL { ?st pq:P8627 ?closeTime . }

        ?daysItem rdfs:label ?daysLabel FILTER(LANG(?daysLabel) IN ("ru","en"))

        BIND(
          CONCAT(
            ?daysLabel,
            IF(BOUND(?openTime), CONCAT(" ", STR(?openTime)), ""),
            IF(BOUND(?closeTime), CONCAT("-", STR(?closeTime)), "")
          ) AS ?ohRow
        )
      }

      SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
    }
    GROUP BY ?item ?itemLabel ?website ?coord
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; moscow-places-csv/1.0)",
        "Accept": "application/sparql-results+json",
    }
    with httpx.Client(timeout=60, headers=headers) as c:
        r = c.get(WIKIDATA_ENDPOINT, params={"query": sparql, "format": "json"})
        r.raise_for_status()
        data = r.json()

    rows = []
    for b in data["results"]["bindings"]:
        name = b.get("itemLabel", {}).get("value")
        website_raw = b.get("website", {}).get("value")
        website = norm_url(website_raw) if website_raw else None
        coord = b.get("coord", {}).get("value")  # "Point(lon lat)"
        lat = lon = None
        if coord and coord.startswith("Point(") and coord.endswith(")"):
            parts = coord[6:-1].split()
            if len(parts) == 2:
                lon, lat = parts[0], parts[1]

        category = b.get("category", {}).get("value") or None
        opening_hours = b.get("opening_hours", {}).get("value") or None

        rows.append(
            {
                "source": "wikidata",
                "name": name,
                "website": website,
                "domain": norm_domain(website),
                "category": category,
                "opening_hours": opening_hours,
                "lat": float(lat) if lat else None,
                "lon": float(lon) if lon else None,
                "osm_id": None,
                "wikidata_id": b.get("item", {}).get("value"),
            }
        )
    return pd.DataFrame(rows)

def fetch_overpass_moscow_food() -> pd.DataFrame:
    """
    OSM Overpass query for Moscow amenities: restaurant|cafe|fast_food
    plus website/contact:website and opening_hours and cuisine.
    """
    query = r"""
    [out:json][timeout:180];
    area["name"="Москва"]["boundary"="administrative"]->.a;
    (
      node["amenity"~"^(restaurant|cafe|fast_food)$"](area.a);
      way["amenity"~"^(restaurant|cafe|fast_food)$"](area.a);
      relation["amenity"~"^(restaurant|cafe|fast_food)$"](area.a);
    );
    out center tags;
    """

    headers = {"User-Agent": "Mozilla/5.0 (compatible; moscow-places-csv/1.0)"}
    with httpx.Client(timeout=240, headers=headers) as c:
        r = c.post(OVERPASS_ENDPOINT, content=query.encode("utf-8"))
        r.raise_for_status()
        data = r.json()

    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or tags.get("name:ru") or tags.get("brand") or None

        website = tags.get("website") or tags.get("contact:website") or None
        website = norm_url(website)

        amenity = (tags.get("amenity") or "").strip()
        cuisine = (tags.get("cuisine") or "").strip()
        category = amenity if amenity else None
        if category and cuisine:
            category = f"{category}; cuisine={cuisine}"
        elif (not category) and cuisine:
            category = f"cuisine={cuisine}"

        opening_hours = (tags.get("opening_hours") or tags.get("opening_hours:url") or "").strip() or None

        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")

        rows.append(
            {
                "source": "openstreetmap",
                "name": name,
                "website": website,
                "domain": norm_domain(website),
                "category": category,
                "opening_hours": opening_hours,
                "lat": float(lat) if lat is not None else None,
                "lon": float(lon) if lon is not None else None,
                "osm_id": f"{el.get('type')}:{el.get('id')}",
                "wikidata_id": None,
            }
        )
    return pd.DataFrame(rows)

def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name"] = df["name"].fillna("").str.strip()
    df = df[df["name"] != ""]

    df["name_l"] = df["name"].str.lower()
    df["domain"] = df["domain"].fillna("")
    df["lat_r"] = df["lat"].round(4)
    df["lon_r"] = df["lon"].round(4)

    keys = []
    for _, r in df.iterrows():
        if r["domain"]:
            keys.append(f"d|{r['domain']}|{r['name_l']}")
        else:
            keys.append(f"g|{r['name_l']}|{r['lat_r']}|{r['lon_r']}")
    df["dedupe_key"] = keys

    df = df.sort_values(by=["domain", "name_l", "source"])
    df = df.drop_duplicates(subset=["dedupe_key"], keep="first")
    return df.drop(columns=["name_l", "lat_r", "lon_r", "dedupe_key"])

def main():
    print("Downloading Wikidata…")
    wd = fetch_wikidata_moscow_food()
    time.sleep(1.0)

    print("Downloading OpenStreetMap (Overpass)…")
    osm = fetch_overpass_moscow_food()

    df = pd.concat([osm, wd], ignore_index=True)
    df = dedupe(df)

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/restaurants_moscow.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved: {out} | rows={len(df)}")

if __name__ == "__main__":
    import os
    main()
