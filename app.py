"""
Journal Finder v3
─────────────────
Three parallel discovery engines + LLM annotation:

  1. OpenAlex  → group_by source on keyword search → ranked by article count
                 Provides: works_count, OA status, APC, is_in_doaj, publisher
  2. PubMed    → esearch with calibrated MeSH query → article count per journal
                 Provides: validated topic article counts (MEDLINE-indexed)
  3. Jane      → HTML parse of suggestions.php → confidence-scored journal list
                 Provides: article-similarity-based journal ranking from Medline

  Scimago CSV  → local lookup by ISSN → SJR, quartile, H-index, Scopus subject area
                 (bundle scimagojr_2024.csv in the app folder — see README)

  Groq LLM     → call 1: build calibrated search queries from user input
                 call 2: annotate merged journal list (scope fit, flags, tier)

APIs used (all free, no subscription keys needed except Groq):
  • Groq Cloud        https://console.groq.com  (free tier, llama-3.3-70b)
  • OpenAlex          https://api.openalex.org   (free, no key)
  • PubMed E-utils    https://eutils.ncbi.nlm.nih.gov (free, optional NCBI key)
  • Jane              https://jane.biosemantics.org   (free HTML endpoint)
  • Scimago           local CSV (download once from scimagojr.com/journalrank.php?out=xls)
"""

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"   # console.groq.com → free, no credit card
NCBI_API_KEY = ""                          # optional: ncbi.nlm.nih.gov/account/
SCIMAGO_CSV  = "scimagojr_2024.csv"       # path relative to app.py; see README
# ─────────────────────────────────────────────────────────────────────────────

import json, re, time, urllib.parse, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

GROQ_MODEL  = "llama-3.3-70b-versatile"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
OA_BASE     = "https://api.openalex.org"
PM_BASE     = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
JANE_URL    = "https://jane.biosemantics.org/suggestions.php"
OA_MAILTO   = "journal-finder@research.app"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Journal Finder v3", page_icon="🔬", layout="wide")
st.title("🔬 Journal Finder")
st.caption(
    "Discovers the most relevant journals using **OpenAlex + PubMed + Jane** in parallel, "
    "enriched with Scimago SJR data, ranked and annotated by an LLM."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 Manuscript details")
    keywords_raw = st.text_area(
        "Keywords *", height=130,
        placeholder="e.g.\nTracheostomy\nCOVID-19\nOtolaryngology\nSurgical volume",
        help="One per line. Required.",
    )
    title = st.text_input("Working title (optional)",
        placeholder="Tracheostomy Procedure Volume Trends…")
    abstract = st.text_area("Abstract (optional)", height=160,
        placeholder="Paste your abstract here for better journal matching…")

    st.divider()
    st.header("⚙️ Settings")
    years        = st.slider("Publication window (years)", 1, 10, 5)
    n_journals   = st.slider("Max journals to find", 10, 25, 15)
    article_type = st.selectbox("Article type",
        ["Original research", "Review", "Systematic review / meta-analysis",
         "Case report", "Methods / technical note"])

    st.divider()
    st.caption(
        "**Data sources**\n"
        "• [OpenAlex](https://openalex.org) — discovery & OA metadata\n"
        "• [PubMed E-utils](https://www.ncbi.nlm.nih.gov/books/NBK25500/) — MEDLINE counts\n"
        "• [Jane](https://jane.biosemantics.org) — similarity-based ranking\n"
        "• [Scimago SJR](https://www.scimagojr.com) — local CSV (Scopus quartiles)\n"
        "• [Groq](https://console.groq.com) — LLM annotation"
    )
    run_btn = st.button("▶ Find journals", type="primary", use_container_width=True)

# ── Scimago CSV loader ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_scimago(path: str) -> pd.DataFrame:
    """
    Load Scimago CSV (semicolon-delimited, downloaded from scimagojr.com).
    Returns DataFrame indexed by cleaned ISSN strings.
    Falls back gracefully if file missing.
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=";", dtype=str, on_bad_lines="skip")
        # Normalise ISSN column: strip spaces, quotes, keep first ISSN
        df["issn_clean"] = (
            df["Issn"].fillna("")
            .str.replace(r'["\s]', "", regex=True)
            .str.split(",").str[0]
        )
        df = df.set_index("issn_clean")
        return df
    except Exception:
        return pd.DataFrame()

SJR_DB = load_scimago(SCIMAGO_CSV)

def sjr_lookup(issn: str) -> dict:
    """Return Scimago metrics for a given ISSN. Returns {} if not found."""
    if SJR_DB.empty or not issn:
        return {}
    cleaned = issn.replace("-", "").strip()
    # Try hyphenated and non-hyphenated
    for key in [issn, cleaned, issn.replace("-", "")]:
        if key in SJR_DB.index:
            row = SJR_DB.loc[key]
            return {
                "sjr":         row.get("SJR", ""),
                "sjr_q":       row.get("SJR Best Quartile", ""),
                "h_index":     row.get("H index", ""),
                "categories":  row.get("Categories", ""),
                "areas":       row.get("Areas", ""),
                "publisher":   row.get("Publisher", ""),
                "is_oa_scimago": str(row.get("Open Access", "")).strip().lower() == "yes",
                "scimago_url": f"https://www.scimagojr.com/journalsearch.php?q={cleaned}&tip=issn",
            }
    return {}

# ── Groq helper ───────────────────────────────────────────────────────────────
def groq_call(messages: list, temperature: float = 0.15, max_tokens: int = 2048) -> str:
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    body    = {"model": GROQ_MODEL, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    r = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def parse_json(text: str) -> object:
    """Strip markdown fences, parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    return json.loads(text)

# ── Step 1: LLM builds search strategy ───────────────────────────────────────
def build_strategy(keywords: list, title: str, abstract: str,
                   years: int, article_type: str) -> dict:
    parts = [f"Keywords: {', '.join(keywords)}"]
    if title:    parts.append(f"Title: {title}")
    if abstract: parts.append(f"Abstract (first 500 chars): {abstract[:500]}")
    parts += [f"Article type: {article_type}",
              f"Publication window: last {years} years"]

    system = (
        "You are a biomedical informationist expert in PubMed and OpenAlex search strategy.\n"
        "Produce a calibrated search strategy for finding 15–20 relevant journals. "
        "NOT too narrow (< 50 results) and NOT too broad (> 5,000 results over 5 years).\n"
        "Output ONLY valid JSON with exactly these keys:\n"
        "  pubmed_query: Valid PubMed query. Rules:\n"
        "    - Use MeSH terms with [MeSH Terms] tag where appropriate\n"
        "    - Combine with free-text [Title/Abstract] terms using OR/AND\n"
        "    - Do NOT include [Journal] or [TA] tags — journal filtering happens separately\n"
        "    - Do NOT include date filters — year filtering happens separately\n"
        "    - Example good query: (tracheostomy[MeSH Terms] OR tracheostomy[Title/Abstract]) "
        "AND (COVID-19[MeSH Terms] OR SARS-CoV-2[Title/Abstract]) AND "
        "(surgical procedures, operative[MeSH Terms] OR surgical volume[Title/Abstract])\n"
        "  openalex_query: 4–6 word phrase for OpenAlex full-text search (no operators)\n"
        "  jane_text: 1–3 sentences of natural language describing the paper topic "
        "(used as input to Jane journal finder — should read like an abstract intro)\n"
        "  concepts: list of 4–6 core subject concept strings\n"
        "  rationale: 2 sentences explaining strategy choices"
    )
    raw = groq_call([{"role": "system", "content": system},
                     {"role": "user",   "content": "\n".join(parts)}])
    return parse_json(raw)

# ── Step 2a: OpenAlex discovery ───────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def oa_discover(query: str, years: int, n: int) -> list:
    """
    Search OpenAlex, group by source, return top-n journal metadata dicts.
    Uses both search= (keyword) and the /sources endpoint for metadata.
    """
    end_y, start_y = date.today().year, date.today().year - years
    params = {
        "search":   query,
        "filter":   f"type:article,publication_year:{start_y}-{end_y},"
                    "primary_location.source.type:journal",   # journals only
        "group_by": "primary_location.source.id",
        "per_page": "60",
        "mailto":   OA_MAILTO,
    }
    r = requests.get(f"{OA_BASE}/works", params=params, timeout=25)
    r.raise_for_status()
    groups = r.json().get("group_by", [])

    results = []
    for g in groups:
        sid = g.get("key", "")
        if not sid or sid in ("unknown", ""):
            continue
        results.append({"oa_id": sid, "oa_count": g.get("count", 0)})
        if len(results) >= n * 2:
            break

    # Enrich with source metadata
    enriched = []
    for item in results:
        raw_id = item["oa_id"]
        short  = raw_id.split("/")[-1]   # e.g. S123456
        meta   = _oa_source(short)
        if not meta or meta.get("type") != "journal":
            continue
        item.update(meta)
        enriched.append(item)
        if len(enriched) >= n:
            break
    return enriched

@st.cache_data(ttl=86400, show_spinner=False)
def _oa_source(short_id: str) -> dict:
    url = f"{OA_BASE}/sources/{short_id}?mailto={OA_MAILTO}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        s = r.json()
        issns  = s.get("issn") or []
        issn   = issns[0] if issns else ""
        stats  = s.get("summary_stats") or {}
        if2y   = stats.get("2yr_mean_citedness")
        return {
            "name":        s.get("display_name", ""),
            "issn":        issn,
            "type":        s.get("type", ""),
            "publisher":   s.get("host_organization_name") or "",
            "is_oa":       s.get("is_oa", False),
            "is_in_doaj":  s.get("is_in_doaj", False),
            "apc_usd":     s.get("apc_usd"),
            "if_oa_2y":    round(if2y, 2) if if2y else None,
            "works_count": s.get("works_count", 0),
            "homepage":    s.get("homepage_url") or "",
            "oa_url":      s.get("id", ""),
        }
    except Exception:
        return {}

# ── Step 2b: PubMed per-journal counts ───────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def pm_count(journal: str, pubmed_q: str, years: int) -> int:
    end_y, start_y = date.today().year, date.today().year - years
    full_q = f'({pubmed_q}) AND "{journal}"[Journal] AND {start_y}:{end_y}[pdat]'
    params = {"db": "pubmed", "term": full_q, "retmode": "json", "rettype": "count"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    try:
        r = requests.get(f"{PM_BASE}/esearch.fcgi", params=params, timeout=12)
        r.raise_for_status()
        return int(r.json()["esearchresult"]["count"])
    except Exception:
        return -1

def pm_url(journal: str, pubmed_q: str, years: int) -> str:
    end_y, start_y = date.today().year, date.today().year - years
    q = f'({pubmed_q}) AND "{journal}"[Journal] AND {start_y}:{end_y}[pdat]'
    return "https://pubmed.ncbi.nlm.nih.gov/?" + urllib.parse.urlencode({"term": q})

# ── Step 2c: Jane journal suggestions ────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def jane_suggest(text: str, n: int = 20) -> list:
    """
    Call Jane's suggestions.php endpoint, parse HTML table.
    Returns list of dicts: {name, confidence, article_count}
    Confidence is the % score Jane assigns (0–100).
    """
    params = {"findJournals": "", "text": text}
    headers = {"User-Agent": "Mozilla/5.0 (journal-finder research tool; "
               "mailto:research@university.edu)"}
    try:
        r = requests.get(JANE_URL, params=params, headers=headers, timeout=25)
        r.raise_for_status()
    except Exception as e:
        return [{"error": str(e)}]

    soup = BeautifulSoup(r.text, "html.parser")
    journals = []
    # Jane renders results in <tr> rows with class 'janeResult' or similar
    # Try multiple selectors as the page structure may vary
    rows = (soup.find_all("tr", class_="janeResult") or
            soup.find_all("tr", class_=re.compile(r"jane", re.I)) or
            soup.select("table.results tr") or
            soup.find_all("tr")[1:])   # skip header

    for row in rows[:n]:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        name_cell = cells[0]
        name = name_cell.get_text(strip=True)
        if not name or len(name) < 5:
            continue
        # Extract confidence score — Jane shows it as a percentage bar or text
        conf_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
        conf = 0
        m = re.search(r"(\d+(?:\.\d+)?)", conf_text)
        if m:
            conf = float(m.group(1))
        # Article count if available
        art_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
        art = 0
        m2 = re.search(r"(\d+)", art_text)
        if m2:
            art = int(m2.group(1))
        journals.append({"name": name, "jane_confidence": conf, "jane_articles": art})

    # If parsing failed, try a simple text scan for journal names
    if not journals:
        for tag in soup.find_all(["td", "div"], class_=re.compile(r"journal|title|name", re.I)):
            txt = tag.get_text(strip=True)
            if 10 < len(txt) < 120 and not txt[0].isdigit():
                journals.append({"name": txt, "jane_confidence": 0, "jane_articles": 0})
            if len(journals) >= n:
                break

    return journals

def jane_url(text: str) -> str:
    return JANE_URL + "?" + urllib.parse.urlencode({"findJournals": "", "text": text})

# ── Step 2d: merge sources ────────────────────────────────────────────────────
def merge_sources(oa_results: list, jane_results: list, n: int) -> list:
    """
    Merge OpenAlex and Jane journal lists by name similarity.
    Jane-only hits are added if they have ISSN in Scimago (quality gate).
    Returns deduplicated list up to n journals.
    """
    merged = {j["name"].lower(): j for j in oa_results}

    for j in jane_results:
        if "error" in j:
            continue
        key = j["name"].lower().strip()
        if key in merged:
            merged[key]["jane_confidence"] = j.get("jane_confidence", 0)
            merged[key]["jane_articles"]   = j.get("jane_articles", 0)
        else:
            # Only add Jane-only journals if they're found in Scimago (quality gate)
            # Try to find ISSN via OpenAlex by name
            merged[key] = {
                "name":             j["name"],
                "issn":             "",
                "publisher":        "",
                "is_oa":            False,
                "is_in_doaj":       False,
                "apc_usd":          None,
                "if_oa_2y":         None,
                "works_count":      0,
                "oa_count":         0,
                "homepage":         "",
                "oa_url":           "",
                "jane_confidence":  j.get("jane_confidence", 0),
                "jane_articles":    j.get("jane_articles", 0),
                "jane_only":        True,
            }

    result = list(merged.values())

    # Score for ordering: oa_count (60%) + jane_confidence (40%)
    max_oa   = max((r.get("oa_count", 0) for r in result), default=1) or 1
    max_jane = max((r.get("jane_confidence", 0) for r in result), default=1) or 1
    for r in result:
        r["_score"] = (
            0.6 * r.get("oa_count", 0) / max_oa +
            0.4 * r.get("jane_confidence", 0) / max_jane
        )
    result.sort(key=lambda x: x["_score"], reverse=True)
    return result[:n]

# ── Step 3: LLM annotation ────────────────────────────────────────────────────
def annotate(journals: list, concepts: list, article_type: str, title: str) -> list:
    summaries = []
    for j in journals:
        summaries.append({
            "name":            j.get("name", ""),
            "issn":            j.get("issn", ""),
            "publisher":       j.get("publisher", ""),
            "sjr":             j.get("sjr", ""),
            "sjr_quartile":    j.get("sjr_q", ""),
            "h_index":         j.get("h_index", ""),
            "is_oa":           j.get("is_oa", False),
            "is_in_doaj":      j.get("is_in_doaj", False),
            "apc_usd":         j.get("apc_usd"),
            "oa_article_count":j.get("oa_count", 0),
            "pubmed_count":    j.get("pubmed_count", -1),
            "jane_confidence": j.get("jane_confidence", 0),
            "categories":      j.get("categories", ""),
        })

    system = (
        "You are an expert academic publishing consultant. "
        "Given candidate journals and manuscript concepts, annotate and rank them. "
        "Output ONLY a JSON array. Each element has EXACTLY these keys:\n"
        "  name: string (exact journal name)\n"
        "  rank: integer (1 = best overall)\n"
        "  scope_fit: one of Excellent | Good | Moderate | Weak\n"
        "  scope_note: ≤20 words explaining scope fit\n"
        "  red_flag: brief warning if any (scope mismatch, very low SJR, predatory risk); "
        "    empty string if none\n"
        "  apc_note: ≤15 words on cost/OA situation\n"
        "  tier: one of Reach | Target | Safety\n\n"
        "Ranking criteria (in order):\n"
        "1. Topical fit to the manuscript concepts (most important)\n"
        "2. SJR quartile (Q1 > Q2 > Q3 > Q4)\n"
        "3. Evidence of publishing on this topic (pubmed_count, oa_article_count, jane_confidence)\n"
        "4. Accessibility (OA, low APC)\n"
        "Flag journals with SJR quartile Q4 or missing SJR as ⚠ low impact. "
        "Flag clear scope mismatches (e.g., pure oncology journal for a surgical volume paper)."
    )
    user = (
        f"Manuscript concepts: {', '.join(concepts)}\n"
        f"Article type: {article_type}\n"
        f"Working title: {title or 'not provided'}\n\n"
        f"Candidate journals:\n{json.dumps(summaries, indent=2)}"
    )
    raw = groq_call([{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
                    temperature=0.1, max_tokens=3500)
    annotations = parse_json(raw)
    ann_map = {a["name"]: a for a in annotations}
    enriched = []
    for j in journals:
        a = ann_map.get(j["name"], {})
        j["rank"]       = a.get("rank", 99)
        j["scope_fit"]  = a.get("scope_fit", "–")
        j["scope_note"] = a.get("scope_note", "")
        j["red_flag"]   = a.get("red_flag", "")
        j["apc_note"]   = a.get("apc_note", "")
        j["tier"]       = a.get("tier", "–")
        enriched.append(j)
    return sorted(enriched, key=lambda x: x.get("rank", 99))

# ── Utility: search links ─────────────────────────────────────────────────────
def wos_url(name: str, q: str) -> str:
    return ("https://www.webofscience.com/wos/woscc/full-search?" +
            urllib.parse.urlencode({"query": f'SO="{name}" AND TS=({q})'}))

def scopus_url(name: str, q: str) -> str:
    return ("https://www.scopus.com/search/form.uri#basic?" +
            urllib.parse.urlencode({"query": f'SRCTITLE("{name}") AND TITLE-ABS-KEY({q})'}))

# ── Main flow ─────────────────────────────────────────────────────────────────
if not run_btn:
    # Show instructions and example when idle
    st.info("Enter your manuscript details in the sidebar, then click **▶ Find journals**.")
    with st.expander("ℹ️ How this tool works"):
        st.markdown("""
**Three discovery engines run in parallel:**

| Engine | What it finds | How |
|---|---|---|
| **OpenAlex** | Journals with most articles matching your keywords | `group_by=source` on keyword search |
| **PubMed** | Validated MEDLINE article counts per journal | E-utilities `esearch` with MeSH query |
| **Jane** | Journals whose published articles most resemble your text | Medline semantic similarity |

**Scimago SJR** data (downloaded once, local CSV) provides Scopus quartile, SJR score, and H-index for any discovered journal.

**Groq LLM** (Llama 3.3 70B) does two things:
1. Builds a calibrated search query from your keywords/title/abstract
2. Reads the merged table and annotates each journal with scope fit, red flags, APC notes, and a recommended submission tier

**How to bundle Scimago data:**
1. Go to [scimagojr.com/journalrank.php](https://www.scimagojr.com/journalrank.php)
2. Click **Download** (CSV/Excel button at bottom of the page)
3. Save as `scimagojr_2024.csv` in the same folder as `app.py`
        """)
    st.stop()

keywords = [k.strip() for k in keywords_raw.splitlines() if k.strip()]
if not keywords:
    st.error("Please enter at least one keyword.")
    st.stop()

if SJR_DB.empty:
    st.warning(
        "⚠️ **Scimago CSV not found** — SJR, quartile and H-index columns will be empty.\n\n"
        "Download `scimagojr_2024.csv` from "
        "[scimagojr.com/journalrank.php](https://www.scimagojr.com/journalrank.php) "
        "and place it next to `app.py`. See README for details."
    )

# ── Step 1: search strategy ───────────────────────────────────────────────────
with st.status("🧠 Building search strategy…", expanded=True) as status:
    st.write("LLM is calibrating PubMed + OpenAlex + Jane queries…")
    try:
        strat = build_strategy(keywords, title, abstract, years, article_type)
    except Exception as e:
        st.error(f"Groq error building strategy: {e}")
        st.stop()
    pubmed_q  = strat["pubmed_query"]
    oa_q      = strat["openalex_query"]
    jane_text = strat.get("jane_text", " ".join(keywords))
    concepts  = strat["concepts"]
    rationale = strat["rationale"]
    st.write(f"**PubMed query:** `{pubmed_q}`")
    st.write(f"**OpenAlex:** `{oa_q}`  |  **Jane text:** _{jane_text[:120]}_")
    status.update(label="✅ Strategy ready", state="complete")

# ── Step 2: parallel discovery ────────────────────────────────────────────────
with st.status("📡 Discovering journals (3 engines in parallel)…", expanded=True) as status:
    oa_results, jane_results = [], []

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_oa   = ex.submit(oa_discover, oa_q, years, n_journals)
        fut_jane = ex.submit(jane_suggest, jane_text, 25)
        for fut in as_completed([fut_oa, fut_jane]):
            if fut is fut_oa:
                try:
                    oa_results = fut.result()
                    st.write(f"✅ OpenAlex: **{len(oa_results)}** journals")
                except Exception as e:
                    st.write(f"⚠️ OpenAlex error: {e}")
            else:
                try:
                    jane_results = fut.result()
                    if jane_results and "error" in jane_results[0]:
                        st.write(f"⚠️ Jane: {jane_results[0]['error']}")
                        jane_results = []
                    else:
                        st.write(f"✅ Jane: **{len(jane_results)}** suggestions")
                except Exception as e:
                    st.write(f"⚠️ Jane error: {e}")

    # Merge
    journals = merge_sources(oa_results, jane_results, n_journals)
    st.write(f"**Merged & deduplicated:** {len(journals)} unique journals")
    status.update(label=f"✅ {len(journals)} journals discovered", state="complete")

if not journals:
    st.warning("No journals found. Try broader or different keywords.")
    st.stop()

# ── Step 3: Scimago enrichment + PubMed counts ───────────────────────────────
with st.status("📊 Enriching with Scimago + PubMed counts…", expanded=True) as status:
    for j in journals:
        name = j.get("name", "")
        issn = j.get("issn", "")

        # Scimago
        sjr = sjr_lookup(issn)
        if sjr:
            j.update(sjr)
        else:
            j.setdefault("sjr", "")
            j.setdefault("sjr_q", "")
            j.setdefault("h_index", "")
            j.setdefault("categories", "")
            j.setdefault("scimago_url", f"https://www.scimagojr.com/journalsearch.php?q={urllib.parse.quote(name)}&tip=pub")

        # PubMed count
        st.write(f"  PubMed → {name[:55]}…")
        j["pubmed_count"] = pm_count(name, pubmed_q, years)
        j["pubmed_url"]   = pm_url(name, pubmed_q, years)
        j["wos_url"]      = wos_url(name, oa_q)
        j["scopus_url"]   = scopus_url(name, oa_q)
        time.sleep(0.15)   # respect 3 req/s NCBI limit

    status.update(label="✅ Enrichment done", state="complete")

# ── Step 4: LLM annotation ────────────────────────────────────────────────────
with st.status("✍️ LLM annotating journals…", expanded=True) as status:
    try:
        journals = annotate(journals, concepts, article_type, title)
    except Exception as e:
        st.warning(f"LLM annotation failed ({e}) — showing unannotated results.")
    status.update(label="✅ Annotation complete", state="complete")

# ── Step 5: Display ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Results")

# Strategy expander
with st.expander("🔎 Search strategy", expanded=False):
    c1, c2 = st.columns(2)
    c1.markdown("**PubMed query**"); c1.code(pubmed_q, language="text")
    c2.markdown("**OpenAlex query**"); c2.code(oa_q, language="text")
    st.markdown(f"**Jane input:** _{jane_text}_")
    st.markdown(f"**Core concepts:** {', '.join(concepts)}")
    st.markdown(f"**Rationale:** {rationale}")

st.caption(
    f"{len(journals)} journals  ·  keywords: `{', '.join(keywords)}`  ·  "
    f"{years}-year window  ·  {article_type}"
)
st.caption("Tier: 🔵 Reach  🟢 Target  🟡 Safety  |  Scope: 🟢 Excellent  🔵 Good  🟡 Moderate  🔴 Weak")

TIER_ICON  = {"Reach": "🔵", "Target": "🟢", "Safety": "🟡"}
SCOPE_ICON = {"Excellent": "🟢", "Good": "🔵", "Moderate": "🟡", "Weak": "🔴"}

rows = []
for j in journals:
    rows.append({
        "#":           j.get("rank", "–"),
        "Journal":     j.get("name", ""),
        "ISSN":        j.get("issn", ""),
        "SJR":         j.get("sjr", "–"),
        "Q (Scopus)":  j.get("sjr_q", "–"),
        "H-index":     j.get("h_index", "–"),
        "IF proxy":    j.get("if_oa_2y"),
        "OA":          "✓" if j.get("is_oa") else "–",
        "DOAJ":        "✓" if j.get("is_in_doaj") else "–",
        "APC (USD)":   j.get("apc_usd"),
        "OA hits (OA)":j.get("oa_count", 0),
        "PubMed hits": j.get("pubmed_count", "–"),
        "Jane conf.":  j.get("jane_confidence", "–"),
        "Scope fit":   f"{SCOPE_ICON.get(j.get('scope_fit',''), '')} {j.get('scope_fit','')}",
        "Tier":        f"{TIER_ICON.get(j.get('tier',''), '')} {j.get('tier','')}",
        "Red flag":    j.get("red_flag", ""),
        "APC note":    j.get("apc_note", ""),
        "Scope note":  j.get("scope_note", ""),
    })

df = pd.DataFrame(rows)
st.dataframe(
    df, use_container_width=True, height=560, hide_index=True,
    column_config={
        "#":            st.column_config.NumberColumn(width=40),
        "Journal":      st.column_config.TextColumn(width="large"),
        "ISSN":         st.column_config.TextColumn(width="small"),
        "SJR":          st.column_config.TextColumn(width="small"),
        "Q (Scopus)":   st.column_config.TextColumn(width="small"),
        "H-index":      st.column_config.TextColumn(width="small"),
        "IF proxy":     st.column_config.NumberColumn(format="%.2f", width="small"),
        "OA":           st.column_config.TextColumn(width="small"),
        "DOAJ":         st.column_config.TextColumn(width="small"),
        "APC (USD)":    st.column_config.NumberColumn(format="$%d", width="small"),
        "OA hits (OA)": st.column_config.NumberColumn(width="small"),
        "PubMed hits":  st.column_config.NumberColumn(width="small"),
        "Jane conf.":   st.column_config.NumberColumn(format="%.0f%%", width="small"),
        "Scope fit":    st.column_config.TextColumn(width="medium"),
        "Tier":         st.column_config.TextColumn(width="small"),
        "Red flag":     st.column_config.TextColumn(width="large"),
        "APC note":     st.column_config.TextColumn(width="large"),
        "Scope note":   st.column_config.TextColumn(width="large"),
    },
)

# Per-journal detail + links
st.subheader("🔗 Links & detailed notes")
for j in journals:
    flag  = f"  ⚠ {j['red_flag']}" if j.get("red_flag") else ""
    label = (f"**#{j.get('rank')} {j.get('name', '')}** — "
             f"{j.get('tier', '')} · {j.get('scope_fit', '')}{flag}")
    with st.expander(label):
        c1, c2, c3, c4, c5 = st.columns(5)
        if j.get("homepage"):
            c1.markdown(f"[🏠 Journal]({j['homepage']})")
        c2.markdown(f"[🔍 PubMed]({j.get('pubmed_url', '#')})")
        c3.markdown(f"[📚 WoS ↗]({j.get('wos_url', '#')})")
        c4.markdown(f"[📊 Scopus ↗]({j.get('scopus_url', '#')})")
        if j.get("scimago_url"):
            c5.markdown(f"[📈 Scimago]({j.get('scimago_url', '#')})")

        ca, cb = st.columns(2)
        ca.markdown(f"**Scope:** {j.get('scope_note', '–')}")
        cb.markdown(f"**APC:** {j.get('apc_note', '–')}")
        if j.get("categories"):
            st.caption(f"Scopus categories: {j.get('categories', '')}")

# Jane open in browser
st.markdown(
    f"🔗 [Open Jane directly]({jane_url(jane_text)}) "
    "— interactive Medline similarity search with up to 20 journal suggestions"
)

# CSV download
st.divider()
dl_cols = ["#", "Journal", "ISSN", "SJR", "Q (Scopus)", "H-index", "IF proxy",
           "OA", "DOAJ", "APC (USD)", "OA hits (OA)", "PubMed hits", "Jane conf.",
           "Scope fit", "Tier", "Scope note", "Red flag", "APC note"]
st.download_button(
    "⬇ Download results as CSV",
    df[dl_cols].to_csv(index=False),
    file_name="journal_finder_results.csv",
    mime="text/csv",
)
