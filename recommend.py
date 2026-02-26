\
import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

ALCO_KEYWORDS = [
    "вино", "wine", "шампан", "игрист", "просек", "prosecco",
    "пиво", "beer", "сидр", "cider",
    "коктейл", "cocktail", "мохито", "negroni", "аперол", "aperol",
    "виски", "whisky", "whiskey", "ром", "rum", "джин", "gin",
    "водка", "vodka", "текил", "tequila", "коньяк", "cognac",
    "ликер", "liqueur", "мартини", "martini",
    "бурбон", "bourbon", "вермут", "vermouth",
]

def is_alcohol_item(name: str) -> bool:
    s = (name or "").lower()
    return any(k in s for k in ALCO_KEYWORDS)

DOW_MAP = {"пн":0,"вт":1,"ср":2,"чт":3,"пт":4,"сб":5,"вс":6,"mon":0,"tue":1,"wed":2,"thu":3,"fri":4,"sat":5,"sun":6,"mo":0,"tu":1,"we":2,"th":3,"fr":4,"sa":5,"su":6}

def parse_time_hhmm(s: str) -> Optional[int]:
    m = re.match(r"^\\s*(\\d{1,2}):(\\d{2})\\s*$", s or "")
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return hh * 60 + mm
    return None

def likely_open_now(opening_hours: Optional[str], user_dow: int, user_minutes: int) -> Optional[bool]:
    if not opening_hours:
        return None
    oh = opening_hours.strip().lower()
    if "24/7" in oh or "24x7" in oh:
        return True

    parts = [p.strip() for p in re.split(r"[;|]", oh) if p.strip()]
    rules = []
    for p in parts:
        m = re.search(r"([a-zа-я]{2,3})\\s*-\\s*([a-zа-я]{2,3}).*?(\\d{1,2}:\\d{2})\\s*-\\s*(\\d{1,2}:\\d{2})", p)
        if m:
            d1 = DOW_MAP.get(m.group(1)[:3], DOW_MAP.get(m.group(1), None))
            d2 = DOW_MAP.get(m.group(2)[:3], DOW_MAP.get(m.group(2), None))
            t1 = parse_time_hhmm(m.group(3))
            t2 = parse_time_hhmm(m.group(4))
            if None not in (d1, d2, t1, t2):
                rules.append((d1, d2, t1, t2))
            continue

    if not rules:
        return None

    def dow_in_range(d, d1, d2):
        if d1 <= d2:
            return d1 <= d <= d2
        return d >= d1 or d <= d2

    for d1, d2, t1, t2 in rules:
        if not dow_in_range(user_dow, d1, d2):
            continue
        if t1 <= t2:
            if t1 <= user_minutes <= t2:
                return True
        else:
            if user_minutes >= t1 or user_minutes <= t2:
                return True
    return False

def ask_preferences() -> dict:
    print("\\n--- Подбор бара/ресторана ---\\n")
    mood = input("Какое настроение? (спокойно / весело / романтика / без разницы): ").strip().lower()
    companions = input("С кем идёте? (один / пара / друзья / коллеги): ").strip().lower()
    day = input("День недели? (Пн/Вт/Ср/Чт/Пт/Сб/Вс): ").strip().lower()
    time_now = input("Текущее время (HH:MM): ").strip()
    try:
        budget = float(input("Бюджет на человека (₽): ").strip())
    except:
        budget = 0.0
    alcohol_focus = input("Алкоголь? (вино / пиво / коктейли / крепкое / без разницы): ").strip().lower()
    food_needed = input("Нужна ли еда? (да/нет): ").strip().lower() == "да"
    return {
        "mood": mood,
        "companions": companions,
        "day": day,
        "time_now": time_now,
        "budget": budget,
        "alcohol_focus": alcohol_focus,
        "food_needed": food_needed,
    }

@dataclass
class Candidate:
    name: str
    website: str
    score: float
    reasons: List[str]

def build_features(restaurants_df: pd.DataFrame, menu_df: pd.DataFrame) -> pd.DataFrame:
    menu = menu_df.copy()
    menu["restaurant_name"] = menu["restaurant_name"].fillna("").astype(str)
    menu["name"] = menu["name"].fillna("").astype(str)
    menu["price_value"] = pd.to_numeric(menu.get("price_value"), errors="coerce")
    menu["is_alcohol"] = menu["name"].apply(is_alcohol_item)

    agg_all = menu.groupby("restaurant_name", as_index=False).agg(
        items_count=("name", "count"),
        median_price=("price_value", "median"),
    )
    agg_alco = menu[menu["is_alcohol"]].groupby("restaurant_name", as_index=False).agg(
        alco_count=("name", "count"),
        alco_median_price=("price_value", "median"),
    )

    df = restaurants_df.copy()
    df["name"] = df["name"].fillna("").astype(str)
    df["website"] = df["website"].fillna("").astype(str)
    df["category"] = df.get("category", "").fillna("").astype(str)
    df["opening_hours"] = df.get("opening_hours", "").fillna("").astype(str)

    df = df.merge(agg_all, left_on="name", right_on="restaurant_name", how="left").drop(columns=["restaurant_name"])
    df = df.merge(agg_alco, left_on="name", right_on="restaurant_name", how="left").drop(columns=["restaurant_name"])
    df["items_count"] = df["items_count"].fillna(0).astype(int)
    df["alco_count"] = df["alco_count"].fillna(0).astype(int)
    return df

def score_row(row: pd.Series, pref: dict) -> Candidate:
    score = 0.0
    reasons: List[str] = []

    name = row["name"]
    website = row.get("website", "")

    if row.get("alco_count", 0) > 0:
        score += 30
        reasons.append("есть алкогольные позиции в меню")
    else:
        score -= 50
        reasons.append("не найден алкоголь в меню")

    budget = float(pref.get("budget") or 0)
    ref_price = row.get("alco_median_price")
    if pd.isna(ref_price):
        ref_price = row.get("median_price")

    if budget > 0 and pd.notna(ref_price):
        target = float(ref_price) * 3
        if budget >= target:
            score += 20
            reasons.append(f"по бюджету ок (медиана ~{int(ref_price)} ₽)")
        else:
            score -= 15
            reasons.append(f"может быть дороговато (медиана ~{int(ref_price)} ₽)")

    mood = pref.get("mood", "без разницы")
    cat = (row.get("category") or "").lower()
    if mood == "весело":
        if "bar" in cat or "бар" in cat or "pub" in cat or "паб" in cat:
            score += 8
            reasons.append("категория похожа на бар/паб")
    if mood == "романтика":
        if "wine" in cat or "вин" in cat:
            score += 8
            reasons.append("категория похожа на винный формат")

    # Open now (best-effort)
    day = (pref.get("day") or "").lower()
    tnow = (pref.get("time_now") or "").strip()
    user_dow = DOW_MAP.get(day[:3], None)
    user_minutes = parse_time_hhmm(tnow)
    if user_dow is not None and user_minutes is not None:
        open_flag = likely_open_now(row.get("opening_hours"), user_dow, user_minutes)
        if open_flag is True:
            score += 10
            reasons.append("скорее всего сейчас открыто")
        elif open_flag is False:
            score -= 20
            reasons.append("скорее всего сейчас закрыто")

    if pref.get("food_needed", True):
        if row.get("items_count", 0) > 20:
            score += 6
            reasons.append("меню большое — вероятно есть еда")
        else:
            score -= 4
            reasons.append("меню небольшое — еды может быть мало")

    return Candidate(name=name, website=website, score=score, reasons=reasons)

def main():
    restaurants = pd.read_csv("data/restaurants_moscow.csv")
    menu = pd.read_csv("data/menus_moscow_clean.csv")

    pref = ask_preferences()
    features = build_features(restaurants, menu)

    cands: List[Candidate] = []
    for _, row in features.iterrows():
        if not row.get("website"):
            continue
        cands.append(score_row(row, pref))

    cands.sort(key=lambda x: x.score, reverse=True)
    top = cands[:5]

    print("\\n--- ТОП рекомендации ---\\n")
    for i, c in enumerate(top, 1):
        print(f"{i}. {c.name}")
        if c.website:
            print(f"   Сайт: {c.website}")
        print(f"   Score: {c.score:.1f}")
        for r in c.reasons[:5]:
            print(f"   - {r}")
        print()

if __name__ == "__main__":
    main()
