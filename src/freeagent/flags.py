"""Country name (as written by Transfermarkt) -> flag code used by flagcdn.com (ISO 3166-1 alpha-2, lower case,
plus gb-eng / gb-sct / gb-wls / gb-nir). Flags are public-domain images served by flagcdn.com."""
from __future__ import annotations

CODES = {
    "Afghanistan": "af", "Albania": "al", "Algeria": "dz", "Andorra": "ad", "Angola": "ao", "Argentina": "ar",
    "Armenia": "am", "Australia": "au", "Austria": "at", "Azerbaijan": "az", "Bahrain": "bh", "Belarus": "by",
    "Belgium": "be", "Benin": "bj", "Bolivia": "bo", "Bosnia-Herzegovina": "ba", "Bosnia and Herzegovina": "ba",
    "Brazil": "br", "Bulgaria": "bg", "Burkina Faso": "bf", "Burundi": "bi", "Cameroon": "cm", "Canada": "ca",
    "Cape Verde": "cv", "Central African Republic": "cf", "Chad": "td", "Chile": "cl", "China": "cn",
    "Colombia": "co", "Comoros": "km", "Congo": "cg", "DR Congo": "cd", "Costa Rica": "cr", "Cote d'Ivoire": "ci",
    "Côte d'Ivoire": "ci", "Ivory Coast": "ci", "Croatia": "hr", "Cuba": "cu", "Curacao": "cw", "Curaçao": "cw",
    "Cyprus": "cy", "Czech Republic": "cz", "Czechia": "cz", "Denmark": "dk", "Dominican Republic": "do",
    "Ecuador": "ec", "Egypt": "eg", "El Salvador": "sv", "England": "gb-eng", "Equatorial Guinea": "gq",
    "Eritrea": "er", "Estonia": "ee", "Ethiopia": "et", "Faroe Islands": "fo", "Finland": "fi", "France": "fr",
    "French Guiana": "gf", "Gabon": "ga", "Gambia": "gm", "The Gambia": "gm", "Georgia": "ge", "Germany": "de",
    "Ghana": "gh", "Gibraltar": "gi", "Greece": "gr", "Grenada": "gd", "Guadeloupe": "gp", "Guatemala": "gt",
    "Guinea": "gn", "Guinea-Bissau": "gw", "Guyana": "gy", "Haiti": "ht", "Honduras": "hn", "Hungary": "hu",
    "Iceland": "is", "India": "in", "Indonesia": "id", "Iran": "ir", "Iraq": "iq", "Ireland": "ie",
    "Republic of Ireland": "ie", "Israel": "il", "Italy": "it", "Jamaica": "jm", "Japan": "jp", "Jordan": "jo",
    "Kazakhstan": "kz", "Kenya": "ke", "Korea, South": "kr", "South Korea": "kr", "Korea, North": "kp",
    "Kosovo": "xk", "Kuwait": "kw", "Kyrgyzstan": "kg", "Latvia": "lv", "Lebanon": "lb", "Liberia": "lr",
    "Libya": "ly", "Liechtenstein": "li", "Lithuania": "lt", "Luxembourg": "lu", "Madagascar": "mg",
    "Malawi": "mw", "Malaysia": "my", "Mali": "ml", "Malta": "mt", "Martinique": "mq", "Mauritania": "mr",
    "Mauritius": "mu", "Mexico": "mx", "Moldova": "md", "Monaco": "mc", "Montenegro": "me", "Morocco": "ma",
    "Mozambique": "mz", "Namibia": "na", "Netherlands": "nl", "New Caledonia": "nc", "New Zealand": "nz",
    "Nicaragua": "ni", "Niger": "ne", "Nigeria": "ng", "North Macedonia": "mk", "Macedonia": "mk",
    "Northern Ireland": "gb-nir", "Norway": "no", "Oman": "om", "Palestine": "ps", "Panama": "pa",
    "Paraguay": "py", "Peru": "pe", "Philippines": "ph", "Poland": "pl", "Portugal": "pt", "Qatar": "qa",
    "Romania": "ro", "Russia": "ru", "Rwanda": "rw", "San Marino": "sm", "Saudi Arabia": "sa", "Scotland": "gb-sct",
    "Senegal": "sn", "Serbia": "rs", "Sierra Leone": "sl", "Slovakia": "sk", "Slovenia": "si", "Somalia": "so",
    "South Africa": "za", "Spain": "es", "Sri Lanka": "lk", "Sudan": "sd", "Suriname": "sr", "Sweden": "se",
    "Switzerland": "ch", "Syria": "sy", "Tahiti": "pf", "Tajikistan": "tj", "Tanzania": "tz", "Thailand": "th",
    "Togo": "tg", "Trinidad and Tobago": "tt", "Tunisia": "tn", "Turkey": "tr", "Türkiye": "tr",
    "Turkmenistan": "tm", "Uganda": "ug", "Ukraine": "ua", "United Arab Emirates": "ae", "United States": "us",
    "USA": "us", "Uruguay": "uy", "Uzbekistan": "uz", "Venezuela": "ve", "Vietnam": "vn", "Wales": "gb-wls",
    "Zambia": "zm", "Zimbabwe": "zw",
}
LEAGUE_FLAG = {"GB1": "gb-eng", "ES1": "es", "IT1": "it", "L1": "de", "FR1": "fr"}


def code(country) -> str | None:
    if not isinstance(country, str):
        return None
    return CODES.get(country.strip())


def url(c: str | None, width: int = 40) -> str | None:
    return f"https://flagcdn.com/w{width}/{c}.png" if c else None
