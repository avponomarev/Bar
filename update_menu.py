\
import asyncio
import csv
import json
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
import pandas as pd
import fitz  # pymupdf
from bs4 import BeautifulSoup
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential

@dataclass
class MenuItem:
    site: str
    restaurant_name: str
    restaurant_category: Optional[str]
    restaurant_hours: Optional[str]
    menu_url: str
    source_type: str  # "html" | "pdf"
    name: str
    price_raw: Optional[str] = None
    price_value: Optional[float] = None
    currency: Optional[str] = None
    description: Optional[str] = None
    item_category: Optional[str] = None

MENU_KEYWORDS = [
    "меню", "menu", "еда", "кухня", "доставка", "delivery", "заказать",
    "бар", "винная", "wine", "cocktail", "напитки", "drinks",
    "breakfast", "lunch", "dinner", "обеды", "завтраки",
]
BAD_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".mp4", ".mov", ".avi", ".zip", ".rar")

STOP_ITEM_PATTERNS = [
    r"\bконтакты\b", r"\bcontact\b",
    r"\bо\s+нас\b", r"\babout\b",
    r"\bваканс", r"\bcareer\b",
    r"\bдоставка\b", r"\bdelivery\b",
    r"\bоплата\b", r"\bpayment\b",
    r"\bуслов", r"\bterms\b",
    r"\bполитик", r"\bprivacy\b",
    r"\bкорзина\b", r"\bcart\b",
    r"\bвойти\b", r"\blogin\b",
    r"\bрегистрац", r"\bsign\s*up\b",
]

AGE_GATE_PATTERNS = [
    r"\b18\+\b",
    r"\bвам\s+есть\s+18\b",
    r"\bмне\s+есть\s+18\b",
    r"\bподтверд(ите|ить)\s+возраст\b",
    r"\bage\s*verification\b",
    r"\bover\s*18\b",
]
AGE_GATE_URL_HINTS = ["/age", "age-gate", "adult", "restricted", "18plus", "18+"]

PDF_LINE_PRICE_RE = re.compile(
    r"""
    ^\s*
    (?P<name>.{3,120}?)
    (?:\s{1,6}|[\.\·\-\–]{2,})
    (?P<price>\d[\d\s]{0,6}(?:[.,]\d{1,2})?)
    \s*(?P<cur>₽|р\.?|руб\.?|RUB|€|EUR|\$|USD)?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE
)

def norm_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s

def same_domain(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()

def is_probably_menu_url(url: str) -> int:
    u = url.lower()
    score = 0
    for kw in MENU_KEYWORDS:
        if kw in u:
            score += 2
    if "pdf" in u or u.endswith(".pdf"):
        score += 1
    if any(u.endswith(ext) for ext in BAD_EXT):
        score -= 10
    return score

def score_anchor(text: str, href: str) -> int:
    t = (text or "").lower()
    h = (href or "").lower()
    score = 0
    for kw in MENU_KEYWORDS:
        if kw in t:
            score += 4
        if kw in h:
            score += 2
    if 1 <= len(t) <= 30:
        score += 1
    return score

def extract_currency(s: str) -> Optional[str]:
    if not s:
        return None
    s_l = s.lower()
    if "₽" in s or "руб" in s_l or "rub" in s_l or re.search(r"(\\s|^)(р|р\\.)(\\s|$)", s_l):
        return "RUB"
    if "€" in s or "eur" in s_l:
        return "EUR"
    if "$" in s or "usd" in s_l:
        return "USD"
    return None

def parse_price_value(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = s.replace("\\u00a0", " ").strip()
    s = re.sub(r"[^\\d,.\\-– ]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    if not s:
        return None
    s = s.replace("–", "-")
    parts = s.split("-")
    candidate = parts[0].strip().replace(" ", "")
    if candidate.count(",") == 1 and candidate.count(".") == 0:
        candidate = candidate.replace(",", ".")
    m = re.match(r"^\\d+(\\.\\d+)?$", candidate)
    if not m:
        m2 = re.search(r"\\d+(\\.\\d+)?", candidate)
        if not m2:
            return None
        candidate = m2.group(0)
    try:
        return float(candidate)
    except Exception:
        return None

def looks_like_garbage_item(name: str) -> bool:
    n = (name or "").strip().lower()
    if len(n) < 3 or len(n) > 120:
        return True
    if re.fullmatch(r"[\\d\\W_]+", n):
        return True
    for pat in STOP_ITEM_PATTERNS:
        if re.search(pat, n):
            return True
    return False

def detect_age_gate(html: str, url: str) -> bool:
    u = (url or "").lower()
    if any(h in u for h in AGE_GATE_URL_HINTS):
        return True
    text = re.sub(r"\\s+", " ", (html or "")).lower()
    return any(re.search(p, text) for p in AGE_GATE_PATTERNS)

def normalize_site_key(website: str) -> str:
    p = urlparse(website)
    return (p.netloc or website).lower()

class Fetcher:
    def __init__(self, timeout: float = 20.0, concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
                ),
                "Accept-Language": "ru,en;q=0.8",
            },
            follow_redirects=True,
        )

    async def close(self):
        await self.client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=0.8, max=6))
    async def get_text(self, url: str) -> Tuple[str, str]:
        async with self.semaphore:
            r = await self.client.get(url)
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").lower()
            return r.text, ctype

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.8, min=0.8, max=6))
    async def get_bytes(self, url: str) -> Tuple[bytes, str]:
        async with self.semaphore:
            r = await self.client.get(url)
            r.raise_for_status()
            ctype = (r.headers.get("content-type") or "").lower()
            return r.content, ctype

class MenuFinder:
    def __init__(self, max_pages: int = 60):
        self.max_pages = max_pages

    def extract_links(self, html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        out = []
        for a in soup.select("a[href]"):
            href = (a.get("href") or "").strip()
            if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                continue
            abs_url = urljoin(base_url, href)
            if abs_url.lower().endswith(BAD_EXT):
                continue
            out.append(abs_url)
        return out

    def find_menu_candidates(self, html: str, base_url: str) -> List[Tuple[str, int]]:
        soup = BeautifulSoup(html, "lxml")
        candidates: Dict[str, int] = {}
        for a in soup.select("a[href]"):
            href = a.get("href") or ""
            txt = norm_text(a.get_text(" "))
            abs_url = urljoin(base_url, href)
            if abs_url.lower().endswith(BAD_EXT):
                continue
            if not same_domain(abs_url, base_url):
                continue
            sc = score_anchor(txt, abs_url) + is_probably_menu_url(abs_url)
            if sc <= 0:
                continue
            candidates[abs_url] = max(candidates.get(abs_url, 0), sc)
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    async def crawl_for_menu_pages(self, fetcher: Fetcher, site_url: str) -> List[str]:
        visited: Set[str] = set()
        queue: List[str] = [site_url]
        found: Dict[str, int] = {}

        while queue and len(visited) < self.max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            try:
                html, ctype = await fetcher.get_text(url)
            except Exception:
                continue

            if "application/pdf" in ctype or url.lower().endswith(".pdf"):
                found[url] = max(found.get(url, 0), 6)
                continue

            for cand_url, score in self.find_menu_candidates(html, url):
                found[cand_url] = max(found.get(cand_url, 0), score)

            links = self.extract_links(html, url)
            links_sorted = sorted(
                [l for l in links if same_domain(l, site_url)],
                key=lambda l: is_probably_menu_url(l),
                reverse=True,
            )
            for l in links_sorted[:30]:
                if l not in visited and l not in queue:
                    queue.append(l)

        best = sorted(found.items(), key=lambda x: x[1], reverse=True)
        return [u for u, _ in best[:10]]

class MenuExtractor:
    def extract_from_jsonld(
        self,
        html: str,
        site: str,
        page_url: str,
        restaurant_name: str,
        restaurant_category: Optional[str],
        restaurant_hours: Optional[str],
    ) -> List[MenuItem]:
        soup = BeautifulSoup(html, "lxml")
        items: List[MenuItem] = []
        scripts = soup.select('script[type="application/ld+json"]')
        for s in scripts:
            raw = (s.string or s.get_text() or "").strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue

            nodes = []
            if isinstance(data, list):
                nodes = data
            elif isinstance(data, dict):
                if "@graph" in data and isinstance(data["@graph"], list):
                    nodes = data["@graph"]
                else:
                    nodes = [data]

            for node in nodes:
                self._jsonld_walk(node, site, page_url, restaurant_name, restaurant_category, restaurant_hours, items)

        return items

    def _jsonld_walk(
        self, node, site: str, page_url: str, restaurant_name: str,
        restaurant_category: Optional[str], restaurant_hours: Optional[str], out: List[MenuItem]
    ) -> None:
        if isinstance(node, dict):
            t = node.get("@type")
            if isinstance(t, list):
                t = " ".join(map(str, t))
            t_l = str(t).lower() if t else ""

            if t_l and any(x in t_l for x in ["product", "menuitem", "offer"]):
                name = norm_text(str(node.get("name") or ""))
                desc = norm_text(str(node.get("description") or "")) or None

                price_raw = None
                currency = None
                offers = node.get("offers")
                if isinstance(offers, dict):
                    price_raw = offers.get("price") or offers.get("lowPrice") or offers.get("highPrice")
                    currency = offers.get("priceCurrency")
                elif isinstance(offers, list) and offers:
                    o0 = offers[0]
                    if isinstance(o0, dict):
                        price_raw = o0.get("price") or o0.get("lowPrice") or o0.get("highPrice")
                        currency = o0.get("priceCurrency")

                if name:
                    pr = str(price_raw) if price_raw is not None else None
                    out.append(
                        MenuItem(
                            site=site,
                            restaurant_name=restaurant_name,
                            restaurant_category=restaurant_category,
                            restaurant_hours=restaurant_hours,
                            menu_url=page_url,
                            source_type="html",
                            name=name,
                            price_raw=pr,
                            price_value=parse_price_value(pr),
                            currency=str(currency) if currency is not None else extract_currency(pr or ""),
                            description=desc,
                            item_category=None,
                        )
                    )

            for v in node.values():
                self._jsonld_walk(v, site, page_url, restaurant_name, restaurant_category, restaurant_hours, out)

        elif isinstance(node, list):
            for x in node:
                self._jsonld_walk(x, site, page_url, restaurant_name, restaurant_category, restaurant_hours, out)

    def extract_from_html_heuristics(
        self,
        html: str,
        site: str,
        page_url: str,
        restaurant_name: str,
        restaurant_category: Optional[str],
        restaurant_hours: Optional[str],
    ) -> List[MenuItem]:
        soup = BeautifulSoup(html, "lxml")
        items: List[MenuItem] = []

        cards = soup.select(".menu-item, .menu__item, .product, .product-card, .card, article, li, .item")
        for c in cards:
            name_el = c.select_one("h3, h2, h4, .title, .name, .product-title, .card-title")
            if not name_el:
                continue
            name = norm_text(name_el.get_text(" "))
            if not name:
                continue

            price_el = c.select_one(".price, .cost, [class*='price'], [data-price]")
            price_raw = norm_text(price_el.get_text(" ")) if price_el else None

            desc_el = c.select_one(".desc, .description, .text, p")
            desc = norm_text(desc_el.get_text(" ")) if desc_el else None
            if desc and len(desc) > 300:
                desc = desc[:300]

            items.append(
                MenuItem(
                    site=site,
                    restaurant_name=restaurant_name,
                    restaurant_category=restaurant_category,
                    restaurant_hours=restaurant_hours,
                    menu_url=page_url,
                    source_type="html",
                    name=name,
                    price_raw=price_raw,
                    price_value=parse_price_value(price_raw),
                    currency=extract_currency(price_raw or ""),
                    description=desc,
                    item_category=None,
                )
            )

        uniq = {}
        for it in items:
            key = (it.name.lower(), (it.price_raw or "").lower())
            if key not in uniq:
                uniq[key] = it
        return list(uniq.values())

def pdf_to_text(pdf_bytes: bytes, max_pages: int = 25) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    pages = min(len(doc), max_pages)
    for i in range(pages):
        t = doc[i].get_text("text")
        if t:
            parts.append(t)
    doc.close()
    return "\\n".join(parts)

def parse_pdf_menu_lines_to_items(
    text: str,
    site: str,
    menu_url: str,
    restaurant_name: str,
    restaurant_category: Optional[str],
    restaurant_hours: Optional[str],
) -> List[MenuItem]:
    items: List[MenuItem] = []
    if not text:
        return items

    text = text.replace("\\u00a0", " ")
    lines = [re.sub(r"\\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) <= 200]

    def is_noise_line(ln: str) -> bool:
        l = ln.lower()
        if len(l) < 3:
            return True
        if re.fullmatch(r"\\d+", l):
            return True
        if "www." in l or "http" in l:
            return True
        if re.search(r"\\b(тел|phone|instagram|@)\\b", l):
            return True
        return False

    for ln in lines:
        if is_noise_line(ln):
            continue
        m = PDF_LINE_PRICE_RE.match(ln)
        if not m:
            continue

        name = norm_text(m.group("name"))
        price_part = norm_text(m.group("price"))
        cur_part = norm_text(m.group("cur") or "")
        price_raw = (price_part + (" " + cur_part if cur_part else "")).strip()

        pv = parse_price_value(price_raw)
        cur = extract_currency(price_raw)

        if looks_like_garbage_item(name):
            continue

        items.append(
            MenuItem(
                site=site,
                restaurant_name=restaurant_name,
                restaurant_category=restaurant_category,
                restaurant_hours=restaurant_hours,
                menu_url=menu_url,
                source_type="pdf",
                name=name,
                price_raw=price_raw or None,
                price_value=pv,
                currency=cur,
                description=None,
                item_category=None,
            )
        )

    uniq = {}
    for it in items:
        key = (it.name.lower(), it.price_value)
        if key not in uniq:
            uniq[key] = it
    return list(uniq.values())

def clean_menu_items(items: List[MenuItem]) -> List[MenuItem]:
    cleaned: List[MenuItem] = []
    for it in items:
        it.name = norm_text(it.name)
        it.description = norm_text(it.description or "") or None
        it.price_raw = norm_text(it.price_raw or "") or None

        if looks_like_garbage_item(it.name):
            continue

        if it.price_value is None and it.price_raw:
            it.price_value = parse_price_value(it.price_raw)

        if it.currency is None and it.price_raw:
            it.currency = extract_currency(it.price_raw)

        if it.price_value is not None:
            if it.price_value <= 0 or it.price_value < 30 or it.price_value > 50000:
                continue

        if fuzz.partial_ratio(it.name.lower(), "меню") > 95 and len(it.name) <= 6:
            continue

        cleaned.append(it)

    df = pd.DataFrame([asdict(x) for x in cleaned])
    if df.empty:
        return []

    priced = df[df["price_value"].notna()].copy()
    if not priced.empty:
        def iqr_filter(group: pd.DataFrame) -> pd.DataFrame:
            g = group.copy()
            q1 = g["price_value"].quantile(0.25)
            q3 = g["price_value"].quantile(0.75)
            iqr = q3 - q1
            if iqr <= 1e-9:
                return g
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            return g[(g["price_value"] >= low) & (g["price_value"] <= high)]

        priced_f = priced.groupby("site", group_keys=False).apply(iqr_filter)
        no_price = df[df["price_value"].isna()]
        df2 = pd.concat([priced_f, no_price], ignore_index=True)
    else:
        df2 = df

    df2["name_l"] = df2["name"].str.lower()
    df2["menu_url_l"] = df2["menu_url"].str.lower()
    df2 = df2.drop_duplicates(subset=["site", "name_l", "price_value", "menu_url_l"], keep="first")
    df2 = df2.drop(columns=["name_l", "menu_url_l"])

    out: List[MenuItem] = []
    for _, r in df2.iterrows():
        out.append(MenuItem(**r.to_dict()))
    return out

class UniversalMenuPipeline:
    def __init__(self, concurrency: int = 10, max_pages_per_site: int = 60):
        self.fetcher = Fetcher(concurrency=concurrency)
        self.finder = MenuFinder(max_pages=max_pages_per_site)
        self.extractor = MenuExtractor()

    async def close(self):
        await self.fetcher.close()

    async def process_restaurant_row(self, row: dict, blocked: List[dict]) -> List[MenuItem]:
        website = (row.get("website") or "").strip()
        if not website:
            return []
        website = website.rstrip("/")
        site_key = normalize_site_key(website)

        restaurant_name = row.get("name") or ""
        restaurant_category = row.get("category")
        restaurant_hours = row.get("opening_hours")

        candidates = await self.finder.crawl_for_menu_pages(self.fetcher, website)

        all_items: List[MenuItem] = []
        for menu_url in candidates:
            try:
                html_or_text, ctype = await self.fetcher.get_text(menu_url)
            except Exception as e:
                blocked.append({"site": site_key, "restaurant_name": restaurant_name, "menu_url": menu_url, "reason": f"fetch_error:{type(e).__name__}"})
                continue

            is_pdf = ("application/pdf" in ctype) or menu_url.lower().endswith(".pdf")
            if is_pdf:
                try:
                    pdf_bytes, _ = await self.fetcher.get_bytes(menu_url)
                    pdf_text = pdf_to_text(pdf_bytes, max_pages=25)
                    pdf_items = parse_pdf_menu_lines_to_items(
                        text=pdf_text,
                        site=site_key,
                        menu_url=menu_url,
                        restaurant_name=restaurant_name,
                        restaurant_category=restaurant_category,
                        restaurant_hours=restaurant_hours,
                    )
                    if not pdf_items:
                        blocked.append({"site": site_key, "restaurant_name": restaurant_name, "menu_url": menu_url, "reason": "pdf_no_items"})
                    all_items.extend(pdf_items)
                except Exception as e:
                    blocked.append({"site": site_key, "restaurant_name": restaurant_name, "menu_url": menu_url, "reason": f"pdf_error:{type(e).__name__}"})
                continue

            html = html_or_text
            if detect_age_gate(html, menu_url):
                blocked.append({"site": site_key, "restaurant_name": restaurant_name, "menu_url": menu_url, "reason": "age_gate_detected"})
                continue

            items = self.extractor.extract_from_jsonld(html, site_key, menu_url, restaurant_name, restaurant_category, restaurant_hours)
            if len(items) < 5:
                items += self.extractor.extract_from_html_heuristics(html, site_key, menu_url, restaurant_name, restaurant_category, restaurant_hours)

            if not items:
                blocked.append({"site": site_key, "restaurant_name": restaurant_name, "menu_url": menu_url, "reason": "html_no_items"})
            all_items.extend(items)

        uniq = {}
        for it in all_items:
            key = (it.site, it.menu_url, it.source_type, it.name.lower(), it.price_value)
            if key not in uniq:
                uniq[key] = it
        return list(uniq.values())

    async def run_from_restaurants_csv(self, restaurants_csv: str) -> Tuple[List[MenuItem], List[dict]]:
        df = pd.read_csv(restaurants_csv)
        rows = df.to_dict(orient="records")
        blocked: List[dict] = []

        async def _one(r: dict) -> List[MenuItem]:
            try:
                return await self.process_restaurant_row(r, blocked)
            except Exception as e:
                website = (r.get("website") or "").strip()
                blocked.append({"site": normalize_site_key(website) if website else "", "restaurant_name": r.get("name") or "", "menu_url": website, "reason": f"row_error:{type(e).__name__}"})
                return []

        chunks = await asyncio.gather(*[_one(r) for r in rows])
        items: List[MenuItem] = []
        for ch in chunks:
            items.extend(ch)
        return items, blocked

def save_menu_csv(items: Iterable[MenuItem], path: str) -> None:
    fieldnames = [
        "site","restaurant_name","restaurant_category","restaurant_hours",
        "menu_url","source_type","name","price_raw","price_value","currency",
        "description","item_category"
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for it in items:
            w.writerow(asdict(it))

def save_blocked_csv(rows: List[dict], path: str) -> None:
    if not rows:
        return
    fieldnames = ["site", "restaurant_name", "menu_url", "reason"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

async def main():
    in_csv = "data/restaurants_moscow.csv"
    out_menu_csv = "data/menus_moscow_clean.csv"
    out_blocked_csv = "data/blocked_or_failed_pages.csv"

    pipe = UniversalMenuPipeline(concurrency=10, max_pages_per_site=60)
    try:
        raw_items, blocked = await pipe.run_from_restaurants_csv(in_csv)
        clean_items = clean_menu_items(raw_items)
        save_menu_csv(clean_items, out_menu_csv)
        save_blocked_csv(blocked, out_blocked_csv)

        print(f"Restaurants file: {in_csv}")
        print(f"Raw menu items: {len(raw_items)}")
        print(f"Clean menu items: {len(clean_items)}")
        print(f"Saved: {out_menu_csv}")
        print(f"Blocked/failed pages: {len(blocked)} -> {out_blocked_csv}")
    finally:
        await pipe.close()

if __name__ == "__main__":
    asyncio.run(main())
