"""Rule-based normalisation of business names / addresses.
Pure string processing - no external lookups (per the fair-play rules)."""
import re
import pandas as pd
from unidecode import unidecode

# Country is an OPEN set: known aliases are canonicalised, unknown labels
# (e.g. a new country in the test set) simply pass through lower-cased.
COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "united states": "us", "united states of america": "us",
    "america": "us", "in": "in", "ind": "in", "india": "in",
    "fr": "fr", "fra": "fr", "france": "fr",
}

NAME_ABBR = {
    "corp": "corporation", "inc": "incorporated", "ltd": "limited", "pvt": "private",
    "co": "company", "intl": "international", "mfg": "manufacturing", "svcs": "services",
    "svc": "service", "assoc": "associates", "bros": "brothers", "govt": "government",
    "dept": "department", "natl": "national", "univ": "university", "hosp": "hospital",
}
LEGAL = {
    "incorporated", "corporation", "company", "limited", "private", "llc", "llp", "plc",
    "gmbh", "sarl", "sas", "sa", "eurl", "sasu", "lp", "opc", "pllc",
}
NAME_DROP = {"the", "and", "dba", "aka"}

ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court",
    "pl": "place", "hwy": "highway", "pkwy": "parkway", "sq": "square", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "blk": "block", "apt": "apartment",
    "sec": "sector", "rte": "route", "chem": "chemin", "stn": "station", "mkt": "market",
    "cir": "circle", "ter": "terrace",
}
# transliteration / renaming variants of common city names
CITY = {
    "bengaluru": "bangalore", "bombay": "mumbai", "madras": "chennai", "calcutta": "kolkata",
    "gurugram": "gurgaon", "baroda": "vadodara", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "mysuru": "mysore", "puducherry": "pondicherry", "poona": "pune",
}

LANDMARK = re.compile(
    r"\b(?:near|nr|opposite|opp|behind|beside|next to|adjacent to|close to|landmark)\b[^,]*"
)
POSTAL = re.compile(r"(?<!\d)(\d{5,6})(?:-\d{4})?(?!\d)")  # US zip / IN PIN / FR code postal


def norm_country(c):
    s = re.sub(r"[^a-z ]", "", unidecode(str(c)).lower()).strip()
    return COUNTRY_ALIASES.get(s, s)


def _clean(s):
    s = unidecode(str(s)).lower().replace("&", " and ")
    s = re.sub(r"['`]", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_name(raw):
    """-> (core name without legal suffixes, legal tokens, full normalised name)"""
    toks = _clean(raw).split()
    out = []
    for k, t in enumerate(toks):
        if t in NAME_DROP:
            continue
        if t == "p" and k + 1 < len(toks) and toks[k + 1] in ("ltd", "limited"):
            t = "private"
        out.append(NAME_ABBR.get(t, t))
    core = [t for t in out if t not in LEGAL] or out
    legal = sorted({t for t in out if t in LEGAL})
    return " ".join(core), " ".join(legal), " ".join(out)


def _expand(text):
    return [CITY.get(t, ADDR_ABBR.get(t, t)) for t in _clean(text).split()]


def norm_address(raw):
    """-> (address core w/o landmark & postal, full address w/o postal, postal code, number tokens)"""
    raw = unidecode(str(raw)).lower()
    found = POSTAL.findall(raw)
    postal = found[-1] if found else ""
    no_postal = POSTAL.sub(" ", raw)
    core = _expand(LANDMARK.sub(" ", no_postal))
    full = _expand(no_postal)
    nums = " ".join(t for t in core if t.isdigit())
    return " ".join(core), " ".join(full), postal, nums


def prepare(df):
    df = df.copy()
    for c in ("business_name", "business_address", "country"):
        df[c] = df[c].fillna("").astype(str)
    df["name_core"] = df["business_name"].map(lambda value: norm_name(value)[0])
    df["name_legal"] = df["business_name"].map(lambda value: norm_name(value)[1])
    df["name_full"] = df["business_name"].map(lambda value: norm_name(value)[2])
    df["addr_core"] = df["business_address"].map(lambda value: norm_address(value)[0])
    df["addr_full"] = df["business_address"].map(lambda value: norm_address(value)[1])
    df["postal"] = df["business_address"].map(lambda value: norm_address(value)[2])
    df["nums"] = df["business_address"].map(lambda value: norm_address(value)[3])
    df["ctry"] = df["country"].map(norm_country)
    df["text_full"] = (df["name_full"] + " " + df["addr_core"]).str.strip()
    return df.reset_index(drop=True)