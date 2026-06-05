"""
Journal Finder v4
─────────────────
Flow:
  1. User enters keywords + optional title/abstract
  2. Groq LLM → suggests 2-3 MeSH terms + editable PubMed search string
  3. PubMed (esearch + einfo journal facet) + Jane fire immediately in parallel
  4. User can edit the search string → "Re-run" button re-fires PubMed + Jane
  5. OpenAlex enriches the *combined journal name list* (metadata only: OA, APC, IF, DOAJ)
  6. Scimago CSV → SJR, quartile, H-index per journal
  7. Groq annotates the merged table (scope fit, tier, red flags)

Tabs: PubMed | Jane | OpenAlex metadata | Merged & Annotated
"""

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
GROQ_API_KEY = "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"   # console.groq.com — free tier
NCBI_API_KEY = "297913872cccbeaf7b2e626307a38ede7d09"                          # optional: ncbi.nlm.nih.gov/account/
SCIMAGO_CSV  = "scimagojr_2024.csv"       # place next to app.py — see README
# ─────────────────────────────────────────────────────────────────────────────

import json, re, time, os, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
OA_BASE    = "https://api.openalex.org"
PM_BASE    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
JANE_URL   = "https://jane.biosemantics.org/suggestions.php"
JANE_OPEN  = "https://jane.biosemantics.org/"

# ─── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Journal Finder", page_icon="🔬", layout="wide")
st.title("🔬 Journal Finder v4")
st.caption(
    "LLM builds a MeSH search string → **PubMed** + **Jane** discover journals in parallel "
    "→ **OpenAlex** enriches with metadata → **Scimago** adds SJR/quartile → **Groq** annotates."
)

# ─── Scimago CSV ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_scimago(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=";", dtype=str, on_bad_lines="skip")
        # Build multi-key ISSN index (Scimago stores multiple ISSNs comma-separated)
        records = {}
        for _, row in df.iterrows():
            raw_issn = str(row.get("Issn", ""))
            for issn in re.findall(r"\d{4}-?\d{3}[\dX]", raw_issn):
                issn_clean = issn.replace("-", "")
                records[issn_clean] = row.to_dict()
                records[issn]       = row.to_dict()   # keep hyphenated too
        return pd.DataFrame(records).T.drop_duplicates()
    except Exception:
        return pd.DataFrame()

SJR_DF = load_scimago(SCIMAGO_CSV)

def sjr_lookup(issn: str) -> dict:
    if SJR_DF.empty or not issn:
        return {}
    for key in [issn, issn.replace("-", "")]:
        if key in SJR_DF.index:
            r = SJR_DF.loc[key]
            return {
                "sjr":       str(r.get("SJR",                "")),
                "sjr_q":     str(r.get("SJR Best Quartile",  "")),
                "h_index":   str(r.get("H index",            "")),
                "categories":str(r.get("Categories",         "")),
                "oa_scimago":str(r.get("Open Access",        "")).strip().lower() == "yes",
                "sjr_url":   f"https://www.scimagojr.com/journalsearch.php"
                             f"?q={issn.replace('-','')}&tip=issn",
            }
    return {}

# ─── Groq helpers ─────────────────────────────────────────────────────────────
def groq_call(messages: list, temperature=0.15, max_tokens=2048) -> str:
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type":  "application/json"},
        json={"model": GROQ_MODEL, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def parse_json_response(text: str):
    text = re.sub(r"^```(?:json)?", "", text.strip()).rstrip("`").strip()
    return json.loads(text)

# ─── Step 1 : LLM → MeSH + search string ─────────────────────────────────────
def build_mesh_strategy(keywords: list, title: str, abstract: str,
                        years: int, article_type: str) -> dict:
    parts = [f"Keywords provided: {', '.join(keywords)}"]
    if title:    parts.append(f"Working title: {title}")
    if abstract: parts.append(f"Abstract (first 600 chars): {abstract[:600]}")
    parts += [f"Article type: {article_type}",
              f"Desired time window: last {years} years"]

    system = """You are a biomedical librarian specialising in PubMed search strategy.

Given manuscript details, produce a calibrated search string for finding 15-25 relevant journals.
The string must be suitable for PubMed esearch AND for Jane (jane.biosemantics.org).

Rules for the search string:
- Use 2-3 core MeSH terms with [MeSH Terms] tag
- Combine with 2-4 free-text synonyms using [Title/Abstract] tag
- Use OR between synonyms, AND between distinct concepts
- Keep it balanced: aim for 200-2000 PubMed results over the time window
- Do NOT include journal names, [TA], [Journal], or date filters

Output ONLY valid JSON with exactly these keys:
  mesh_terms: list of 2-3 MeSH term strings (without the [MeSH Terms] tag)
  search_string: the complete ready-to-use PubMed query string
  jane_text: 2-3 sentence natural language description of the paper (for Jane's text input)
  concepts: list of 4-6 core concept strings for LLM annotation
  rationale: 1-2 sentences explaining the strategy

Example search_string format:
(tracheostomy[MeSH Terms] OR tracheostomy[Title/Abstract] OR tracheotomy[Title/Abstract]) AND (COVID-19[MeSH Terms] OR SARS-CoV-2[Title/Abstract] OR pandemic[Title/Abstract]) AND (surgical procedures, operative[MeSH Terms] OR surgical volume[Title/Abstract] OR procedure volume[Title/Abstract])"""

    raw = groq_call(
        [{"role": "system", "content": system},
         {"role": "user",   "content": "\n".join(parts)}]
    )
    return parse_json_response(raw)

# ─── Step 2a : PubMed journal frequency via efetch/esearch ───────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def pubmed_journal_counts(search_str: str, years: int, max_results: int = 5000) -> list[dict]:
    """
    Run esearch with the search string, retrieve up to max_results PMIDs,
    then fetch the journal names via esummary in batches.
    Returns list of {journal, count} sorted descending.
    """
    end_y   = date.today().year
    start_y = end_y - years
    query   = f"({search_str}) AND {start_y}:{end_y}[pdat]"
    params  = {"db": "pubmed", "term": query, "retmax": str(max_results),
               "retmode": "json", "usehistory": "y"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    # Step 1: esearch
    r = requests.get(f"{PM_BASE}/esearch.fcgi", params=params, timeout=20)
    r.raise_for_status()
    result   = r.json()["esearchresult"]
    total    = int(result.get("count", 0))
    web_env  = result.get("webenv", "")
    query_key= result.get("querykey", "")
    ids      = result.get("idlist", [])

    if not ids:
        return []

    # Step 2: esummary in batches of 500 to get journal names
    journal_counts: dict[str, int] = {}
    batch = 500
    for start in range(0, min(len(ids), max_results), batch):
        chunk = ids[start:start + batch]
        sp = {"db": "pubmed", "id": ",".join(chunk), "retmode": "json"}
        if NCBI_API_KEY:
            sp["api_key"] = NCBI_API_KEY
        try:
            sr = requests.get(f"{PM_BASE}/esummary.fcgi", params=sp, timeout=30)
            sr.raise_for_status()
            summaries = sr.json().get("result", {})
            for uid, doc in summaries.items():
                if uid == "uids":
                    continue
                jname = doc.get("fulljournalname") or doc.get("source", "")
                if jname:
                    journal_counts[jname] = journal_counts.get(jname, 0) + 1
            time.sleep(0.12 if not NCBI_API_KEY else 0.05)
        except Exception:
            continue

    ranked = sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"journal": j, "pubmed_count": c} for j, c in ranked], total

@st.cache_data(ttl=1800, show_spinner=False)
def pubmed_total_count(search_str: str, years: int) -> int:
    end_y, start_y = date.today().year, date.today().year - years
    q = f"({search_str}) AND {start_y}:{end_y}[pdat]"
    p = {"db": "pubmed", "term": q, "retmode": "json", "rettype": "count"}
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    try:
        r = requests.get(f"{PM_BASE}/esearch.fcgi", params=p, timeout=12)
        r.raise_for_status()
        return int(r.json()["esearchresult"]["count"])
    except Exception:
        return -1

def pubmed_open_url(search_str: str, years: int) -> str:
    end_y, start_y = date.today().year, date.today().year - years
    q = f"({search_str}) AND {start_y}:{end_y}[pdat]"
    return "https://pubmed.ncbi.nlm.nih.gov/?" + urllib.parse.urlencode({"term": q})

# ─── Step 2b : Jane ───────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def jane_journals(text: str, n: int = 25) -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0 (journal-finder; mailto:research@university.edu)"}
    try:
        r = requests.get(JANE_URL, params={"findJournals": "", "text": text},
                         headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        return [{"error": str(e)}]

    soup = BeautifulSoup(r.text, "html.parser")
    results = []

    # Jane renders a table; rows have alternating classes or a consistent structure
    # Try the known structure: each journal result is a <tr> in the results table
    rows = (
        soup.select("tr.janeResult") or
        soup.select("table#suggestionsTable tr") or
        soup.select("div.jane-result") or
        soup.find_all("tr")[2:]   # skip header rows
    )

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # Journal name — first meaningful text cell
        name = cells[0].get_text(" ", strip=True)
        if not name or len(name) < 4 or name[0].isdigit():
            continue

        # Confidence — Jane shows a numeric score or a progress bar width
        conf = 0.0
        conf_cell = cells[1] if len(cells) > 1 else None
        if conf_cell:
            # Try to find a percentage in a style attribute (progress bar)
            style = conf_cell.get("style", "")
            m = re.search(r"width\s*:\s*(\d+(?:\.\d+)?)\s*%", style)
            if m:
                conf = float(m.group(1))
            else:
                # Or a plain number in the cell text
                m2 = re.search(r"(\d+(?:\.\d+)?)", conf_cell.get_text())
                if m2:
                    conf = float(m2.group(1))

        # Article count — third cell if present
        art = 0
        if len(cells) > 2:
            m3 = re.search(r"(\d+)", cells[2].get_text())
            if m3:
                art = int(m3.group(1))

        results.append({
            "journal":          name,
            "jane_confidence":  conf,
            "jane_articles":    art,
        })
        if len(results) >= n:
            break

    # Fallback: extract any text that looks like a journal name
    if not results:
        for tag in soup.find_all(["a", "td", "div"],
                                  string=re.compile(r"[A-Z][a-z]+ .{5,60}")):
            txt = tag.get_text(strip=True)
            if 8 < len(txt) < 120 and not txt[0].isdigit():
                results.append({"journal": txt, "jane_confidence": 0.0, "jane_articles": 0})
            if len(results) >= n:
                break

    return results

# ─── Step 3 : OpenAlex metadata enrichment ───────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def oa_metadata_by_name(journal_name: str) -> dict:
    """Look up a journal by name in OpenAlex /sources. Returns metadata dict."""
    params = {
        "filter":  f"display_name.search:{urllib.parse.quote(journal_name)}",
        "per_page": "3",
        "mailto":   "journal-finder@research.app",
    }
    try:
        r = requests.get(f"{OA_BASE}/sources", params=params, timeout=12)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return {}
        # pick best name match
        best = results[0]
        for s in results:
            if s.get("display_name", "").lower() == journal_name.lower():
                best = s
                break
        issns  = best.get("issn") or []
        issn   = issns[0] if issns else ""
        stats  = best.get("summary_stats") or {}
        if2y   = stats.get("2yr_mean_citedness")
        return {
            "issn":       issn,
            "publisher":  best.get("host_organization_name") or "",
            "is_oa":      best.get("is_oa", False),
            "is_in_doaj": best.get("is_in_doaj", False),
            "apc_usd":    best.get("apc_usd"),
            "if_oa_2y":   round(if2y, 2) if if2y else None,
            "homepage":   best.get("homepage_url") or "",
            "oa_id":      best.get("id", ""),
        }
    except Exception:
        return {}

# ─── Step 4 : Groq annotation ─────────────────────────────────────────────────
def annotate_merged(journals: list, concepts: list,
                    article_type: str, title: str) -> list:
    summaries = []
    for j in journals:
        summaries.append({
            "name":            j.get("journal", ""),
            "issn":            j.get("issn", ""),
            "publisher":       j.get("publisher", ""),
            "sjr":             j.get("sjr", ""),
            "sjr_quartile":    j.get("sjr_q", ""),
            "h_index":         j.get("h_index", ""),
            "is_oa":           j.get("is_oa", False),
            "is_in_doaj":      j.get("is_in_doaj", False),
            "apc_usd":         j.get("apc_usd"),
            "if_oa_2y":        j.get("if_oa_2y"),
            "pubmed_count":    j.get("pubmed_count", 0),
            "jane_confidence": j.get("jane_confidence", 0),
            "jane_articles":   j.get("jane_articles", 0),
            "scopus_categories": j.get("categories", ""),
        })

    system = """You are an expert academic publishing consultant.
Annotate and rank the following candidate journals for this manuscript.
Output ONLY a JSON array with no preamble. Each element has EXACTLY these keys:
  name: string — exact journal name as given
  rank: integer — 1 = best overall fit
  scope_fit: one of Excellent | Good | Moderate | Weak
  scope_note: ≤20 words on why this journal fits or doesn't
  red_flag: brief warning if any (wrong scope, low SJR, predatory risk); empty string if none
  apc_note: ≤15 words on cost/OA situation
  tier: one of Reach | Target | Safety

Ranking logic (in order of importance):
1. Topical fit to the manuscript concepts
2. SJR quartile if available (Q1 > Q2 > Q3 > Q4 > missing)
3. Evidence of publishing on this topic (pubmed_count + jane_confidence + jane_articles)
4. Accessibility (OA, low APC)
Flag Q4 or missing SJR with ⚠ in red_flag.
Flag clear scope mismatches even if the journal has a high SJR."""

    user = (
        f"Manuscript concepts: {', '.join(concepts)}\n"
        f"Article type: {article_type}\n"
        f"Working title: {title or 'not provided'}\n\n"
        f"Candidate journals:\n{json.dumps(summaries, indent=2)}"
    )

    raw = groq_call(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
        temperature=0.1, max_tokens=4000,
    )
    annotations = parse_json_response(raw)
    ann_map = {a["name"]: a for a in annotations}
    for j in journals:
        a = ann_map.get(j.get("journal", ""), {})
        j["rank"]       = a.get("rank",       99)
        j["scope_fit"]  = a.get("scope_fit",  "–")
        j["scope_note"] = a.get("scope_note", "")
        j["red_flag"]   = a.get("red_flag",   "")
        j["apc_note"]   = a.get("apc_note",   "")
        j["tier"]       = a.get("tier",       "–")
    return sorted(journals, key=lambda x: x.get("rank", 99))

# ─── Search link builders ─────────────────────────────────────────────────────
def wos_url(name: str, q: str) -> str:
    return ("https://www.webofscience.com/wos/woscc/full-search?" +
            urllib.parse.urlencode({"query": f'SO="{name}" AND TS=({q})'}))

def scopus_url(name: str, q: str) -> str:
    return ("https://www.scopus.com/search/form.uri#basic?" +
            urllib.parse.urlencode({"query": f'SRCTITLE("{name}") AND TITLE-ABS-KEY({q})'}))

def sjr_url_name(name: str) -> str:
    return ("https://www.scimagojr.com/journalsearch.php?" +
            urllib.parse.urlencode({"q": name, "tip": "pub"}))

# ─── SESSION STATE KEYS ───────────────────────────────────────────────────────
# strategy, search_string, jane_text, concepts, pm_results, pm_total,
# jane_results, merged, annotated

def init_state():
    for key in ["strategy", "search_string", "jane_text", "concepts",
                "pm_journals", "pm_total", "jane_results",
                "merged", "annotated", "oa_enriched"]:
        if key not in st.session_state:
            st.session_state[key] = None

init_state()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 Manuscript details")
    kw_raw  = st.text_area("Keywords *", height=130,
                            placeholder="One per line\ne.g.\nTracheostomy\nCOVID-19\nOtolaryngology\nSurgical volume")
    title   = st.text_input("Working title (optional)",
                             placeholder="Tracheostomy Procedure Volume Trends…")
    abstract= st.text_area("Abstract (optional)", height=150,
                             placeholder="Paste abstract here…")
    st.divider()
    st.header("⚙️ Settings")
    years        = st.slider("Publication window (years)", 1, 10, 5)
    n_journals   = st.slider("Max journals in final table", 10, 25, 15)
    article_type = st.selectbox("Article type",
                                ["Original research", "Review",
                                 "Systematic review / meta-analysis",
                                 "Case report", "Methods / technical note"])
    st.divider()
    build_btn = st.button("🧠 Build search string", type="primary",
                           use_container_width=True)
    st.caption(
        "**APIs**  \n"
        "[PubMed E-utils](https://www.ncbi.nlm.nih.gov/books/NBK25500/) · "
        "[Jane](https://jane.biosemantics.org) · "
        "[OpenAlex](https://openalex.org) · "
        "[Groq](https://console.groq.com)"
    )

# ─── STEP 1 : Build strategy + fire PubMed & Jane in parallel ─────────────────
keywords = [k.strip() for k in kw_raw.splitlines() if k.strip()]

if build_btn:
    if not keywords:
        st.error("Please enter at least one keyword.")
        st.stop()

    with st.spinner("🧠 Generating MeSH-based search string…"):
        try:
            strat = build_mesh_strategy(keywords, title, abstract,
                                        years, article_type)
            st.session_state.strategy      = strat
            st.session_state.search_string = strat["search_string"]
            st.session_state.jane_text     = strat.get("jane_text", " ".join(keywords))
            st.session_state.concepts      = strat.get("concepts", keywords)
            # Clear downstream results so tabs refresh
            for k in ["pm_journals", "pm_total", "jane_results",
                      "merged", "annotated", "oa_enriched"]:
                st.session_state[k] = None
        except Exception as e:
            st.error(f"Groq error: {e}")
            st.stop()

    # Fire PubMed + Jane immediately
    with st.spinner("🔍 Running PubMed and Jane searches in parallel…"):
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_pm   = ex.submit(pubmed_journal_counts,
                                 st.session_state.search_string, years, 3000)
            fut_jane = ex.submit(jane_journals,
                                 st.session_state.jane_text, 25)
            for fut in as_completed([fut_pm, fut_jane]):
                if fut is fut_pm:
                    try:
                        result = fut.result()
                        if isinstance(result, tuple):
                            st.session_state.pm_journals, st.session_state.pm_total = result
                        else:
                            st.session_state.pm_journals = result
                            st.session_state.pm_total = None
                    except Exception as e:
                        st.warning(f"PubMed error: {e}")
                        st.session_state.pm_journals = []
                else:
                    try:
                        jres = fut.result()
                        if jres and "error" in jres[0]:
                            st.warning(f"Jane: {jres[0]['error']}")
                            st.session_state.jane_results = []
                        else:
                            st.session_state.jane_results = jres
                    except Exception as e:
                        st.warning(f"Jane error: {e}")
                        st.session_state.jane_results = []

# ─── EDITABLE SEARCH STRING ───────────────────────────────────────────────────
if st.session_state.strategy:
    strat = st.session_state.strategy
    st.divider()

    with st.expander("🔎 Search strategy", expanded=True):
        st.markdown(
            f"**MeSH terms identified:** "
            f"`{'` · `'.join(strat.get('mesh_terms', []))}`"
        )
        st.caption(strat.get("rationale", ""))

        edited_query = st.text_area(
            "✏️ PubMed / Jane search string — edit and re-run if needed",
            value=st.session_state.search_string,
            height=120,
            key="editable_query",
            help="This query is used for both PubMed (with date filter added automatically) "
                 "and passed to Jane as structured text. Edit and click Re-run.",
        )

        col1, col2 = st.columns([1, 3])
        rerun_btn = col1.button("🔄 Re-run with edited query")
        col2.markdown(
            f"[Open in PubMed ↗]({pubmed_open_url(edited_query, years)})  ·  "
            f"[Open Jane ↗]({JANE_OPEN})",
            unsafe_allow_html=False,
        )

    if rerun_btn and edited_query != st.session_state.search_string:
        st.session_state.search_string = edited_query
        # Also update Jane text to reflect edited query intent
        st.session_state.jane_text = edited_query.replace("[MeSH Terms]", "")\
            .replace("[Title/Abstract]", "").replace("[pdat]", "")\
            .replace("AND", " ").replace("OR", " ")\
            .replace("(", " ").replace(")", " ")
        for k in ["pm_journals", "pm_total", "jane_results",
                  "merged", "annotated", "oa_enriched"]:
            st.session_state[k] = None

        with st.spinner("🔄 Re-running PubMed + Jane…"):
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_pm   = ex.submit(pubmed_journal_counts,
                                     st.session_state.search_string, years, 3000)
                fut_jane = ex.submit(jane_journals,
                                     st.session_state.jane_text, 25)
                for fut in as_completed([fut_pm, fut_jane]):
                    if fut is fut_pm:
                        try:
                            result = fut.result()
                            if isinstance(result, tuple):
                                st.session_state.pm_journals, st.session_state.pm_total = result
                            else:
                                st.session_state.pm_journals = result
                        except Exception as e:
                            st.warning(f"PubMed re-run error: {e}")
                    else:
                        try:
                            jres = fut.result()
                            st.session_state.jane_results = (
                                [] if (jres and "error" in jres[0]) else jres
                            )
                        except Exception as e:
                            st.warning(f"Jane re-run error: {e}")

# ─── TABS ─────────────────────────────────────────────────────────────────────
if (st.session_state.pm_journals is not None or
        st.session_state.jane_results is not None):

    tab_pm, tab_jane, tab_oa, tab_merged = st.tabs(
        ["🔍 PubMed", "📄 Jane", "🌐 OpenAlex metadata", "⭐ Merged & Annotated"]
    )

    # ── Tab 1: PubMed ─────────────────────────────────────────────────────────
    with tab_pm:
        pm_list = st.session_state.pm_journals or []
        pm_tot  = st.session_state.pm_total

        if pm_tot is not None:
            st.metric("Total PubMed results for query", f"{pm_tot:,}")
        else:
            st.caption("Showing top journals by publication count.")

        if pm_list:
            df_pm = pd.DataFrame(pm_list[:50])   # top 50 for display
            df_pm.index = range(1, len(df_pm) + 1)
            df_pm.columns = ["Journal", "PubMed count"]

            # Enrich with SJR if ISSN found via OpenAlex later;
            # for now show plain table with bar chart
            st.dataframe(
                df_pm,
                use_container_width=True,
                height=500,
                column_config={
                    "Journal":      st.column_config.TextColumn(width="large"),
                    "PubMed count": st.column_config.ProgressColumn(
                        format="%d", min_value=0,
                        max_value=df_pm["PubMed count"].max() or 1,
                        width="medium",
                    ),
                },
            )
            st.caption(
                f"Showing top {len(df_pm)} journals · full query period: "
                f"{date.today().year - years}–{date.today().year}"
            )
            st.markdown(
                f"[🔗 Open full query in PubMed]"
                f"({pubmed_open_url(st.session_state.search_string, years)})"
            )
        else:
            st.warning("No PubMed results. Try editing the search string above.")

    # ── Tab 2: Jane ───────────────────────────────────────────────────────────
    with tab_jane:
        jane_list = st.session_state.jane_results or []

        st.caption(
            f"Jane matches abstracts/titles in Medline to your text. "
            f"[Open Jane directly ↗]({JANE_OPEN})"
        )

        if jane_list and "error" not in jane_list[0]:
            df_jane = pd.DataFrame(jane_list)
            df_jane.index = range(1, len(df_jane) + 1)
            col_map = {"journal": "Journal",
                       "jane_confidence": "Confidence (%)",
                       "jane_articles": "Matching articles"}
            df_jane = df_jane.rename(columns=col_map)
            display_cols = [c for c in col_map.values() if c in df_jane.columns]

            max_conf = df_jane["Confidence (%)"].max() if "Confidence (%)" in df_jane.columns else 100
            st.dataframe(
                df_jane[display_cols],
                use_container_width=True,
                height=500,
                column_config={
                    "Journal":           st.column_config.TextColumn(width="large"),
                    "Confidence (%)":    st.column_config.ProgressColumn(
                        format="%.1f%%", min_value=0,
                        max_value=float(max_conf) or 100.0,
                        width="medium",
                    ),
                    "Matching articles": st.column_config.NumberColumn(width="small"),
                },
            )
            st.caption(
                "Confidence = how closely Jane's Medline similarity score matches "
                "your input text to articles published in each journal."
            )
        elif jane_list and "error" in jane_list[0]:
            st.error(f"Jane returned an error: {jane_list[0]['error']}")
            st.markdown(f"[Open Jane manually ↗]({JANE_OPEN})")
        else:
            st.warning("No Jane results yet.")

    # ── Tab 3: OpenAlex metadata ──────────────────────────────────────────────
    with tab_oa:
        pm_names   = [j["journal"]  for j in (st.session_state.pm_journals  or [])[:n_journals]]
        jane_names = [j["journal"]  for j in (st.session_state.jane_results or [])
                      if "error" not in j][:n_journals]
        all_names  = list(dict.fromkeys(pm_names + jane_names))[:n_journals]

        if not all_names:
            st.info("Run a search first to see OpenAlex metadata.")
        else:
            if st.session_state.oa_enriched is None:
                with st.spinner(f"Fetching OpenAlex metadata for {len(all_names)} journals…"):
                    enriched = {}
                    for name in all_names:
                        enriched[name] = oa_metadata_by_name(name)
                        time.sleep(0.1)
                    st.session_state.oa_enriched = enriched

            oa_data = st.session_state.oa_enriched or {}
            oa_rows = []
            for name in all_names:
                m = oa_data.get(name, {})
                oa_rows.append({
                    "Journal":    name,
                    "ISSN":       m.get("issn", "–"),
                    "Publisher":  m.get("publisher", "–"),
                    "OA":         "✓" if m.get("is_oa") else "–",
                    "DOAJ":       "✓" if m.get("is_in_doaj") else "–",
                    "APC (USD)":  m.get("apc_usd"),
                    "IF proxy":   m.get("if_oa_2y"),
                    "Homepage":   m.get("homepage", ""),
                })

            df_oa = pd.DataFrame(oa_rows)
            df_oa.index = range(1, len(df_oa) + 1)
            st.dataframe(
                df_oa,
                use_container_width=True,
                height=500,
                column_config={
                    "Journal":   st.column_config.TextColumn(width="large"),
                    "ISSN":      st.column_config.TextColumn(width="small"),
                    "Publisher": st.column_config.TextColumn(width="medium"),
                    "OA":        st.column_config.TextColumn(width="small"),
                    "DOAJ":      st.column_config.TextColumn(width="small"),
                    "APC (USD)": st.column_config.NumberColumn(format="$%d", width="small"),
                    "IF proxy":  st.column_config.NumberColumn(format="%.2f", width="small"),
                    "Homepage":  st.column_config.LinkColumn(width="medium"),
                },
            )
            st.caption(
                "OpenAlex metadata: OA status, DOAJ indexation, APC, and 2-year "
                "mean citedness (IF proxy). Used to enrich the Merged table."
            )

    # ── Tab 4: Merged & Annotated ─────────────────────────────────────────────
    with tab_merged:
        if st.button("⭐ Build merged & annotated table", type="primary"):
            pm_list   = st.session_state.pm_journals  or []
            jane_list = [j for j in (st.session_state.jane_results or [])
                         if "error" not in j]

            # Build unified dict keyed by lowercase journal name
            merged: dict[str, dict] = {}

            for j in pm_list[:n_journals]:
                key = j["journal"].lower().strip()
                merged[key] = {"journal": j["journal"],
                               "pubmed_count": j["pubmed_count"],
                               "jane_confidence": 0.0,
                               "jane_articles":   0}

            for j in jane_list[:n_journals]:
                key = j["journal"].lower().strip()
                if key in merged:
                    merged[key]["jane_confidence"] = j.get("jane_confidence", 0)
                    merged[key]["jane_articles"]   = j.get("jane_articles", 0)
                else:
                    merged[key] = {"journal": j["journal"],
                                   "pubmed_count":    0,
                                   "jane_confidence": j.get("jane_confidence", 0),
                                   "jane_articles":   j.get("jane_articles", 0)}

            journals = list(merged.values())

            # Trim to n_journals by combined score
            max_pm   = max((j["pubmed_count"]    for j in journals), default=1) or 1
            max_jane = max((j["jane_confidence"] for j in journals), default=1) or 1
            for j in journals:
                j["_score"] = (0.55 * j["pubmed_count"] / max_pm +
                               0.45 * j["jane_confidence"] / max_jane)
            journals = sorted(journals, key=lambda x: x["_score"], reverse=True)[:n_journals]

            # OpenAlex enrichment
            oa_data = st.session_state.oa_enriched or {}
            with st.spinner("Enriching with OpenAlex + Scimago…"):
                for j in journals:
                    name = j["journal"]
                    # OpenAlex
                    if name not in oa_data:
                        oa_data[name] = oa_metadata_by_name(name)
                        time.sleep(0.1)
                    m = oa_data.get(name, {})
                    j.update({
                        "issn":       m.get("issn", ""),
                        "publisher":  m.get("publisher", ""),
                        "is_oa":      m.get("is_oa", False),
                        "is_in_doaj": m.get("is_in_doaj", False),
                        "apc_usd":    m.get("apc_usd"),
                        "if_oa_2y":   m.get("if_oa_2y"),
                        "homepage":   m.get("homepage", ""),
                        "oa_id":      m.get("oa_id", ""),
                    })
                    # Scimago
                    sjr = sjr_lookup(j.get("issn", ""))
                    j.update(sjr if sjr else {
                        "sjr": "", "sjr_q": "", "h_index": "",
                        "categories": "", "oa_scimago": False,
                        "sjr_url": sjr_url_name(name),
                    })
                    j["wos_url"]    = wos_url(name, st.session_state.search_string)
                    j["scopus_url"] = scopus_url(name, st.session_state.search_string)

                st.session_state.oa_enriched = oa_data

            # Groq annotation
            with st.spinner("✍️ Groq annotating journals…"):
                try:
                    journals = annotate_merged(
                        journals,
                        st.session_state.concepts or keywords,
                        article_type, title,
                    )
                except Exception as e:
                    st.warning(f"Annotation error: {e}")

            st.session_state.merged    = journals
            st.session_state.annotated = True

        # ── Display merged table ──────────────────────────────────────────────
        if st.session_state.merged:
            journals = st.session_state.merged
            TIER  = {"Reach": "🔵", "Target": "🟢", "Safety": "🟡"}
            SCOPE = {"Excellent": "🟢", "Good": "🔵", "Moderate": "🟡", "Weak": "🔴"}

            st.caption(
                "🔵 Reach  🟢 Target  🟡 Safety   |   "
                "Scope: 🟢 Excellent  🔵 Good  🟡 Moderate  🔴 Weak"
            )
            if SJR_DF.empty:
                st.warning(
                    "⚠️ Scimago CSV not found — SJR/quartile columns are empty. "
                    "See README to add `scimagojr_2024.csv`."
                )

            rows = []
            for j in journals:
                rows.append({
                    "#":           j.get("rank", "–"),
                    "Journal":     j.get("journal", ""),
                    "ISSN":        j.get("issn", "–"),
                    "Publisher":   (j.get("publisher") or "–")[:28],
                    "SJR":         j.get("sjr", "–"),
                    "Q":           j.get("sjr_q", "–"),
                    "H":           j.get("h_index", "–"),
                    "IF proxy":    j.get("if_oa_2y"),
                    "OA":          "✓" if j.get("is_oa") else "–",
                    "APC (USD)":   j.get("apc_usd"),
                    "PubMed n":    j.get("pubmed_count", 0),
                    "Jane %":      j.get("jane_confidence", 0),
                    "Scope":       f"{SCOPE.get(j.get('scope_fit',''), '')} {j.get('scope_fit','')}",
                    "Tier":        f"{TIER.get(j.get('tier',''), '')} {j.get('tier','')}",
                    "⚠":           j.get("red_flag", ""),
                    "APC note":    j.get("apc_note", ""),
                    "Scope note":  j.get("scope_note", ""),
                })

            df_m = pd.DataFrame(rows)
            st.dataframe(
                df_m, use_container_width=True, height=560, hide_index=True,
                column_config={
                    "#":         st.column_config.NumberColumn(width=40),
                    "Journal":   st.column_config.TextColumn(width="large"),
                    "ISSN":      st.column_config.TextColumn(width="small"),
                    "Publisher": st.column_config.TextColumn(width="medium"),
                    "SJR":       st.column_config.TextColumn(width="small"),
                    "Q":         st.column_config.TextColumn(width="small"),
                    "H":         st.column_config.TextColumn(width="small"),
                    "IF proxy":  st.column_config.NumberColumn(format="%.2f", width="small"),
                    "OA":        st.column_config.TextColumn(width="small"),
                    "APC (USD)": st.column_config.NumberColumn(format="$%d", width="small"),
                    "PubMed n":  st.column_config.ProgressColumn(
                        format="%d", min_value=0,
                        max_value=max((r["PubMed n"] for r in rows), default=1) or 1,
                        width="medium",
                    ),
                    "Jane %":    st.column_config.ProgressColumn(
                        format="%.0f%%", min_value=0, max_value=100, width="medium",
                    ),
                    "Scope":     st.column_config.TextColumn(width="medium"),
                    "Tier":      st.column_config.TextColumn(width="small"),
                    "⚠":         st.column_config.TextColumn(width="medium"),
                    "APC note":  st.column_config.TextColumn(width="large"),
                    "Scope note":st.column_config.TextColumn(width="large"),
                },
            )

            # Per-journal links
            with st.expander("🔗 Per-journal links", expanded=False):
                for j in journals:
                    flag = f"  ⚠ {j['red_flag']}" if j.get("red_flag") else ""
                    with st.expander(
                        f"**#{j.get('rank')} {j.get('journal','')}** "
                        f"— {j.get('tier','')} · {j.get('scope_fit','')}{flag}"
                    ):
                        c = st.columns(5)
                        if j.get("homepage"):
                            c[0].markdown(f"[🏠 Journal]({j['homepage']})")
                        c[1].markdown(
                            f"[🔍 PubMed]"
                            f"({pubmed_open_url(st.session_state.search_string, years)})"
                        )
                        c[2].markdown(f"[📚 WoS ↗]({j.get('wos_url','#')})")
                        c[3].markdown(f"[📊 Scopus ↗]({j.get('scopus_url','#')})")
                        if j.get("sjr_url"):
                            c[4].markdown(f"[📈 Scimago]({j.get('sjr_url','#')})")
                        if j.get("categories"):
                            st.caption(f"Scopus categories: {j.get('categories')}")

            # CSV download
            st.divider()
            dl_cols = ["#", "Journal", "ISSN", "Publisher", "SJR", "Q", "H",
                       "IF proxy", "OA", "APC (USD)", "PubMed n", "Jane %",
                       "Scope", "Tier", "⚠", "APC note", "Scope note"]
            st.download_button(
                "⬇ Download merged table as CSV",
                df_m[dl_cols].to_csv(index=False),
                file_name="journal_finder_merged.csv",
                mime="text/csv",
            )
        else:
            st.info("Click **⭐ Build merged & annotated table** to run the full pipeline.")
