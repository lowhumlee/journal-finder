"""
Journal Finder v2
─────────────────
Pipeline:
  1. User enters keywords, optional title, optional abstract
  2. Groq LLM → builds calibrated PubMed search string + concept list
  3. OpenAlex  → group_by source → top-N journals by article count + metadata
  4. PubMed    → article counts per discovered journal (validation)
  5. Groq LLM → annotates each journal (scope fit, red flags, APC note)
  6. Render ranked table + CSV export

APIs used:
  • Groq Cloud  (free tier, llama-3.3-70b-versatile) — requires GROQ_API_KEY below
  • OpenAlex    (free, no key)
  • PubMed E-utilities (free, no key; optional NCBI key for faster rate)
"""

# ── CONFIGURATION — edit these two lines ──────────────────────────────────────
GROQ_API_KEY = "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"   # get free at console.groq.com
NCBI_API_KEY = "297913872cccbeaf7b2e626307a38ede7d09"                          # optional — speeds up PubMed queries
# ─────────────────────────────────────────────────────────────────────────────

import json
import time
import urllib.parse
import requests
import streamlit as st
import pandas as pd

GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
OA_BASE      = "https://api.openalex.org"
PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OA_MAILTO    = "journal-finder@research.app"

# ── page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Journal Finder", page_icon="🔬", layout="wide")

st.title("🔬 Journal Finder")
st.caption(
    "Discovers the most relevant journals for your manuscript using "
    "OpenAlex + PubMed data, ranked and annotated by an LLM."
)

# ── sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📝 Manuscript details")

    keywords_raw = st.text_area(
        "Keywords *",
        placeholder="e.g.\nTracheostomy\nCOVID-19\nOtolaryngology\nSurgical volume",
        height=130,
        help="One per line. Required.",
    )

    title = st.text_input(
        "Working title (optional)",
        placeholder="Tracheostomy Procedure Volume Trends…",
    )

    abstract = st.text_area(
        "Abstract (optional)",
        placeholder="Paste your abstract here for better journal matching…",
        height=160,
    )

    st.divider()
    st.header("⚙️ Search settings")

    years = st.slider("Publication window (years)", 1, 10, 5)
    n_journals = st.slider("Max journals to discover", 10, 25, 15)

    article_type = st.selectbox(
        "Article type",
        ["Original research", "Review", "Systematic review / meta-analysis",
         "Case report", "Methods / technical note"],
        index=0,
    )

    st.divider()
    st.caption(
        "**APIs used**\n"
        "• [OpenAlex](https://openalex.org) — journal discovery & metadata (free)\n"
        "• [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25500/) — article counts (free)\n"
        "• [Groq](https://console.groq.com) — search strategy & annotations (free tier)"
    )

    run_btn = st.button("▶ Find journals", type="primary", use_container_width=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def groq(messages: list[dict], temperature: float = 0.2, max_tokens: int = 2048) -> str:
    """Call Groq API, return assistant text. Raises on error."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(GROQ_URL, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def build_search_strategy(keywords: list[str], title: str, abstract: str,
                           years: int, article_type: str) -> dict:
    """
    Ask Groq to produce a calibrated search strategy.
    Returns dict with keys: pubmed_query, openalex_query, rationale, concepts
    """
    prompt_parts = [f"Keywords: {', '.join(keywords)}"]
    if title:
        prompt_parts.append(f"Working title: {title}")
    if abstract:
        prompt_parts.append(f"Abstract (first 600 chars): {abstract[:600]}")
    prompt_parts.append(f"Article type: {article_type}")
    prompt_parts.append(f"Publication window: last {years} years")

    system = (
        "You are a biomedical librarian expert in PubMed and OpenAlex search strategies. "
        "Your task: given manuscript details, produce a CALIBRATED search strategy — "
        "broad enough to find 15-25 relevant journals but narrow enough to exclude irrelevant ones. "
        "Avoid over-indexing on very specific terms that would miss closely related journals. "
        "Output ONLY valid JSON with these exact keys:\n"
        "  pubmed_query: string — a valid PubMed query using MeSH and free-text terms with "
        "    boolean operators. Must NOT include journal names or [TA] tags. Use [MeSH Terms] "
        "    and [Title/Abstract] field tags. Aim for ~200-2000 results over the specified window.\n"
        "  openalex_query: string — a shorter search string (3-6 words) for OpenAlex full-text "
        "    search optimised to return the most relevant works. No boolean operators.\n"
        "  rationale: string — 2-3 sentences explaining the strategy choices.\n"
        "  concepts: list of 4-6 strings — core subject concepts distilled from the input "
        "    (used for journal scope matching)."
    )

    user = "\n".join(prompt_parts)
    raw = groq([{"role": "system", "content": system},
                {"role": "user", "content": user}])

    # strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


@st.cache_data(ttl=3600, show_spinner=False)
def openalex_discover(query: str, years: int, n: int) -> list[dict]:
    """
    Search OpenAlex works, group by source → return top-N journals with metadata.
    Returns list of dicts: {source_id, name, issn, works_in_query, ...}
    """
    from datetime import date
    end_year = date.today().year
    start_year = end_year - years

    # Step 1: group_by to rank journals by article count for this query
    params = {
        "search": query,
        "filter": f"type:article,publication_year:{start_year}-{end_year}",
        "group_by": "primary_location.source.id",
        "per_page": 50,  # get top 50 groups; we'll filter to n after metadata lookup
        "mailto": OA_MAILTO,
    }
    r = requests.get(f"{OA_BASE}/works", params=params, timeout=20)
    r.raise_for_status()
    groups = r.json().get("group_by", [])

    # Step 2: filter out non-journal sources, take top n
    results = []
    for g in groups:
        source_id = g.get("key", "")
        count = g.get("count", 0)
        if not source_id or source_id == "unknown":
            continue
        results.append({"source_id": source_id, "works_in_query": count})
        if len(results) >= n * 2:   # fetch extra to allow filtering
            break

    # Step 3: enrich each with source metadata
    enriched = []
    for item in results:
        sid = item["source_id"].split("/")[-1]   # strip URL prefix if present
        meta = _oa_source_meta(sid)
        if not meta or meta.get("type") not in ("journal", "repository"):
            continue
        if meta.get("type") != "journal":
            continue
        item.update(meta)
        enriched.append(item)
        if len(enriched) >= n:
            break

    return enriched


@st.cache_data(ttl=86400, show_spinner=False)
def _oa_source_meta(source_id: str) -> dict:
    """Fetch one OpenAlex source record by its short ID (e.g. S12345678)."""
    url = f"{OA_BASE}/sources/{source_id}?mailto={OA_MAILTO}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        s = r.json()
        # indexation
        indices = s.get("indices", []) or []
        idx_names = [i.get("id", "").lower() for i in indices]
        scopus = any("scopus" in n for n in idx_names)
        wos    = any("wos" in n or "web of science" in n for n in idx_names)
        # IF proxy
        stats = s.get("summary_stats", {}) or {}
        if2y  = stats.get("2yr_mean_citedness")
        issns = s.get("issn", []) or []
        issn  = issns[0] if issns else ""
        return {
            "name":           s.get("display_name", ""),
            "issn":           issn,
            "type":           s.get("type", ""),
            "publisher":      (s.get("host_organization_name") or ""),
            "oa_status":      s.get("apc_usd") is not None or s.get("is_oa", False),
            "is_oa":          s.get("is_oa", False),
            "apc_usd":        s.get("apc_usd"),
            "if_2y":          round(if2y, 2) if if2y else None,
            "works_count":    s.get("works_count", 0),
            "cited_by_count": s.get("cited_by_count", 0),
            "scopus":         scopus,
            "wos":            wos,
            "homepage":       s.get("homepage_url", "") or "",
            "openalex_url":   s.get("id", ""),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def pubmed_count(journal_name: str, pubmed_query: str, years: int) -> int:
    """Return article count in PubMed for this journal + the search query."""
    from datetime import date
    end_year = date.today().year
    start_year = end_year - years
    # Combine the LLM-built query with a journal filter
    combined = f'({pubmed_query}) AND "{journal_name}"[Journal] AND {start_year}:{end_year}[pdat]'
    params = {
        "db": "pubmed",
        "term": combined,
        "retmode": "json",
        "rettype": "count",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    try:
        r = requests.get(f"{PUBMED_BASE}/esearch.fcgi", params=params, timeout=10)
        r.raise_for_status()
        return int(r.json()["esearchresult"]["count"])
    except Exception:
        return -1


def pubmed_search_url(journal_name: str, pubmed_query: str, years: int) -> str:
    from datetime import date
    end_year = date.today().year
    start_year = end_year - years
    q = f'({pubmed_query}) AND "{journal_name}"[Journal] AND {start_year}:{end_year}[pdat]'
    return "https://pubmed.ncbi.nlm.nih.gov/?" + urllib.parse.urlencode({"term": q})


def wos_search_url(journal_name: str, openalex_query: str) -> str:
    q = f'SO="{journal_name}" AND TS=({openalex_query})'
    return "https://www.webofscience.com/wos/woscc/full-search?" + urllib.parse.urlencode({"query": q})


def scopus_search_url(journal_name: str, openalex_query: str) -> str:
    q = f'SRCTITLE("{journal_name}") AND TITLE-ABS-KEY({openalex_query})'
    return "https://www.scopus.com/search/form.uri#basic?" + urllib.parse.urlencode({"query": q})


def annotate_journals(journals: list[dict], concepts: list[str],
                      article_type: str, title: str) -> list[dict]:
    """
    Ask Groq to annotate each journal with scope_fit, flags, apc_note, rank_rationale.
    Returns enriched list sorted by recommended rank.
    """
    # Build a compact journal summary for the prompt
    journal_summaries = []
    for j in journals:
        journal_summaries.append({
            "name":           j.get("name", ""),
            "issn":           j.get("issn", ""),
            "publisher":      j.get("publisher", ""),
            "if_2y":          j.get("if_2y"),
            "scopus":         j.get("scopus", False),
            "wos":            j.get("wos", False),
            "is_oa":          j.get("is_oa", False),
            "apc_usd":        j.get("apc_usd"),
            "works_in_query": j.get("works_in_query", 0),
            "pubmed_count":   j.get("pubmed_count", -1),
        })

    system = (
        "You are an expert academic publishing consultant. "
        "Given a list of candidate journals and manuscript concepts, "
        "annotate and rank the journals by suitability for this manuscript. "
        "Output ONLY a valid JSON array (no markdown, no preamble). "
        "Each element must have EXACTLY these keys:\n"
        "  name: string — exact journal name as given\n"
        "  rank: integer — 1 = best fit\n"
        "  scope_fit: string — one of: Excellent | Good | Moderate | Weak\n"
        "  scope_note: string — ≤20 words on why this journal fits or doesn't\n"
        "  red_flag: string — brief warning if any (predatory risk, wrong scope, very low IF, "
        "    high rejection rate); empty string if none\n"
        "  apc_note: string — ≤15 words on APC / OA situation\n"
        "  submission_tier: string — one of: Reach | Target | Safety\n"
        "Consider: topical fit first, then IF/quartile, then OA/APC, then indexation. "
        "Flag any journal with IF<0.5 as ⚠ low impact. "
        "Flag journals whose scope is clearly mismatched (e.g., pure oncology for a surgical volume paper)."
    )

    user = (
        f"Manuscript concepts: {', '.join(concepts)}\n"
        f"Article type: {article_type}\n"
        f"Working title: {title or 'not provided'}\n\n"
        f"Candidate journals:\n{json.dumps(journal_summaries, indent=2)}"
    )

    raw = groq(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=3000,
    )

    # strip markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    annotations = json.loads(raw)

    # merge annotations back into journal dicts by name
    ann_map = {a["name"]: a for a in annotations}
    enriched = []
    for j in journals:
        ann = ann_map.get(j["name"], {})
        j["rank"]            = ann.get("rank", 99)
        j["scope_fit"]       = ann.get("scope_fit", "–")
        j["scope_note"]      = ann.get("scope_note", "")
        j["red_flag"]        = ann.get("red_flag", "")
        j["apc_note"]        = ann.get("apc_note", "")
        j["submission_tier"] = ann.get("submission_tier", "–")
        enriched.append(j)

    return sorted(enriched, key=lambda x: x.get("rank", 99))


# ── main logic ────────────────────────────────────────────────────────────────

if not run_btn:
    st.info("Enter your manuscript details in the sidebar, then click **▶ Find journals**.")
    st.stop()

keywords = [k.strip() for k in keywords_raw.splitlines() if k.strip()]
if not keywords:
    st.error("Please enter at least one keyword.")
    st.stop()

# ── Step 1: Search strategy ───────────────────────────────────────────────────
with st.status("🧠 Building search strategy…", expanded=True) as status:
    st.write("Asking LLM to calibrate PubMed + OpenAlex queries…")
    try:
        strategy = build_search_strategy(keywords, title, abstract, years, article_type)
    except Exception as e:
        st.error(f"Groq error: {e}")
        st.stop()

    pubmed_q   = strategy["pubmed_query"]
    oa_q       = strategy["openalex_query"]
    rationale  = strategy["rationale"]
    concepts   = strategy["concepts"]

    st.write(f"**PubMed query:** `{pubmed_q}`")
    st.write(f"**OpenAlex query:** `{oa_q}`")
    st.write(f"**Rationale:** {rationale}")
    status.update(label="✅ Search strategy ready", state="complete")

# ── Step 2: OpenAlex discovery ────────────────────────────────────────────────
with st.status("📡 Discovering journals via OpenAlex…", expanded=True) as status:
    st.write(f"Searching for top {n_journals} journals publishing on this topic…")
    try:
        journals = openalex_discover(oa_q, years, n_journals)
    except Exception as e:
        st.error(f"OpenAlex error: {e}")
        st.stop()
    st.write(f"Found **{len(journals)}** candidate journals.")
    status.update(label=f"✅ {len(journals)} journals discovered", state="complete")

if not journals:
    st.warning("No journals found. Try broader keywords.")
    st.stop()

# ── Step 3: PubMed counts ────────────────────────────────────────────────────
with st.status("📊 Fetching PubMed article counts…", expanded=True) as status:
    for j in journals:
        name = j.get("name", "")
        st.write(f"  → {name[:55]}…")
        j["pubmed_count"] = pubmed_count(name, pubmed_q, years)
        j["pubmed_url"]   = pubmed_search_url(name, pubmed_q, years)
        j["wos_url"]      = wos_search_url(name, oa_q)
        j["scopus_url"]   = scopus_search_url(name, oa_q)
        time.sleep(0.15)   # NCBI rate limit without key: ~3 req/s
    status.update(label="✅ PubMed counts done", state="complete")

# ── Step 4: LLM annotation ────────────────────────────────────────────────────
with st.status("✍️ LLM ranking & annotating journals…", expanded=True) as status:
    st.write("Evaluating scope fit, flags, APC notes…")
    try:
        journals = annotate_journals(journals, concepts, article_type, title)
    except Exception as e:
        st.error(f"Groq annotation error: {e}")
        # continue without annotations rather than stopping
    status.update(label="✅ Annotation complete", state="complete")

# ── Step 5: Display ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Recommended journals")

# ── Search strategy expander
with st.expander("🔎 Search strategy used", expanded=False):
    col1, col2 = st.columns(2)
    col1.markdown("**PubMed query**")
    col1.code(pubmed_q, language="text")
    col2.markdown("**OpenAlex query**")
    col2.code(oa_q, language="text")
    st.markdown(f"**Rationale:** {rationale}")
    st.markdown(f"**Core concepts identified:** {', '.join(concepts)}")

st.caption(
    f"Showing {len(journals)} journals · "
    f"keywords: `{', '.join(keywords)}` · {years}-year window · {article_type}"
)

# ── Colour-coded tier legend
tier_colors = {"Reach": "🔵", "Target": "🟢", "Safety": "🟡"}
scope_colors = {"Excellent": "🟢", "Good": "🔵", "Moderate": "🟡", "Weak": "🔴"}
st.caption("🔵 Reach  🟢 Target  🟡 Safety  |  Scope: 🟢 Excellent  🔵 Good  🟡 Moderate  🔴 Weak")

# ── Build display dataframe
display_rows = []
for j in journals:
    display_rows.append({
        "#":                j.get("rank", "–"),
        "Journal":          j.get("name", ""),
        "Publisher":        j.get("publisher", "")[:30],
        "ISSN":             j.get("issn", ""),
        "IF (2yr)":         j.get("if_2y"),
        "Scopus":           "✓" if j.get("scopus") else "–",
        "WoS":              "✓" if j.get("wos") else "–",
        "OA":               "✓" if j.get("is_oa") else "–",
        "APC (USD)":        j.get("apc_usd"),
        "OA count (OpenAlex)": j.get("works_in_query", 0),
        "PubMed hits":      j.get("pubmed_count", "–"),
        "Scope fit":        f"{scope_colors.get(j.get('scope_fit',''), '')} {j.get('scope_fit','')}",
        "Tier":             f"{tier_colors.get(j.get('submission_tier',''), '')} {j.get('submission_tier','')}",
        "Red flag":         j.get("red_flag", ""),
        "APC note":         j.get("apc_note", ""),
        "Scope note":       j.get("scope_note", ""),
    })

df = pd.DataFrame(display_rows)

st.dataframe(
    df,
    use_container_width=True,
    height=550,
    column_config={
        "#":             st.column_config.NumberColumn(width=40),
        "Journal":       st.column_config.TextColumn(width="large"),
        "Publisher":     st.column_config.TextColumn(width="medium"),
        "ISSN":          st.column_config.TextColumn(width="small"),
        "IF (2yr)":      st.column_config.NumberColumn(format="%.2f", width="small"),
        "Scopus":        st.column_config.TextColumn(width="small"),
        "WoS":           st.column_config.TextColumn(width="small"),
        "OA":            st.column_config.TextColumn(width="small"),
        "APC (USD)":     st.column_config.NumberColumn(format="$%d", width="small"),
        "OA count (OpenAlex)": st.column_config.NumberColumn(width="small"),
        "PubMed hits":   st.column_config.NumberColumn(width="small"),
        "Scope fit":     st.column_config.TextColumn(width="medium"),
        "Tier":          st.column_config.TextColumn(width="small"),
        "Red flag":      st.column_config.TextColumn(width="large"),
        "APC note":      st.column_config.TextColumn(width="large"),
        "Scope note":    st.column_config.TextColumn(width="large"),
    },
    hide_index=True,
)

# ── Per-journal detail + links
st.subheader("🔗 Per-journal links & notes")
for j in journals:
    flag = f" ⚠ {j['red_flag']}" if j.get("red_flag") else ""
    tier = j.get("submission_tier", "")
    fit  = j.get("scope_fit", "")
    label = f"**#{j.get('rank')} {j.get('name', '')}** — {tier} · {fit}{flag}"

    with st.expander(label):
        c1, c2, c3, c4 = st.columns(4)
        if j.get("homepage"):
            c1.markdown(f"[🏠 Homepage]({j['homepage']})")
        c2.markdown(f"[🔍 PubMed]({j.get('pubmed_url', '#')})")
        c3.markdown(f"[📚 WoS ↗]({j.get('wos_url', '#')})")
        c4.markdown(f"[📊 Scopus ↗]({j.get('scopus_url', '#')})")

        col_a, col_b = st.columns(2)
        col_a.markdown(f"**Scope note:** {j.get('scope_note', '–')}")
        col_b.markdown(f"**APC note:** {j.get('apc_note', '–')}")
        if j.get("openalex_url"):
            st.caption(f"OpenAlex: {j['openalex_url']}")

# ── CSV export
st.divider()
csv_cols = ["#", "Journal", "Publisher", "ISSN", "IF (2yr)", "Scopus", "WoS",
            "OA", "APC (USD)", "OA count (OpenAlex)", "PubMed hits",
            "Scope fit", "Tier", "Scope note", "Red flag", "APC note"]
csv_data = df[csv_cols].to_csv(index=False)
st.download_button(
    "⬇ Download results as CSV",
    csv_data,
    file_name="journal_finder_results.csv",
    mime="text/csv",
)
