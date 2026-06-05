# Journal Finder v3

LLM-powered journal discovery using three parallel engines:
**OpenAlex** + **PubMed** + **Jane**, enriched with **Scimago SJR** data.

---

## What's new in v3 vs v2

| Problem in v2 | Fix in v3 |
|---|---|
| PubMed returned 0 hits | LLM prompt now explicitly forbids `[Journal]` and date tags in the query; PubMed query is separated from journal filtering |
| No indexation info | Scimago CSV bundled locally → SJR score, Scopus quartile, H-index, subject categories |
| OpenAlex `indices` field empty | Dropped — replaced with `is_in_doaj` (reliable) + Scimago CSV for Scopus coverage |
| Single discovery engine | Three parallel engines (OpenAlex, PubMed via Groq-built query, Jane) merged with score weighting |

---

## Setup — 4 steps

### Step 1: Groq API key (free)
1. Go to https://console.groq.com → sign up (email or Google, no credit card)
2. Create → API Keys → New key (starts with `gsk_…`)
3. Open `app.py`, find line 18, replace the placeholder:
```python
GROQ_API_KEY = "gsk_your_key_here"
```

### Step 2: Scimago CSV (one-time download, ~5 MB)
1. Go to https://www.scimagojr.com/journalrank.php
2. Scroll to bottom → click the **Download** button → save as CSV
3. Rename the file to `scimagojr_2024.csv`
4. Place it in the **same folder as `app.py`**

The CSV is semicolon-delimited with columns:
`Rank;Sourceid;Title;Type;Issn;Publisher;Open Access;SJR;SJR Best Quartile;H index;...;Categories`

The app reads it at startup and looks up each discovered journal by ISSN.
Update once per year when Scimago releases new data.

### Step 3: (Optional) NCBI API key — speeds up PubMed queries
Get a free key at https://www.ncbi.nlm.nih.gov/account/
Add to `app.py` line 19:
```python
NCBI_API_KEY = "your_ncbi_key_here"
```
Raises PubMed rate limit from 3 → 10 req/s, reducing query time from ~45 s → ~15 s.

### Step 4: Run

**Option A — Streamlit Community Cloud (free, public, no local install)**
1. Push this folder to a public GitHub repo
2. Go to https://share.streamlit.io → New app → select repo/file → Deploy
3. Done — permanent URL, no maintenance

**Option B — Local, Windows 11, no admin rights**
1. Download WinPython from https://winpython.github.io/ (unzip anywhere)
2. Double-click `run_local.bat`
3. Opens at http://localhost:8501

---

## How it works

```
User: keywords + optional title/abstract
         │
         ▼
[1] Groq LLM — calibrated search strategy
    ├── pubmed_query  (MeSH + free-text, no journal filters, no date filters)
    ├── openalex_query  (4–6 word phrase)
    └── jane_text  (1–3 natural language sentences)
         │
         ├─── [2a] OpenAlex works?search=<oa_query>&group_by=source
         │         → top-N journals by article count + OA/APC/DOAJ metadata
         │
         ├─── [2b] Jane suggestions.php?findJournals&text=<jane_text>
         │         → top-20 journals by Medline semantic similarity + confidence
         │
         ▼
[3] Merge & deduplicate OpenAlex + Jane results
    (weighted score: 60% OA count + 40% Jane confidence)
         │
         ▼
[4] Scimago CSV lookup by ISSN
    → SJR score, Scopus quartile, H-index, subject categories
         │
         ▼
[5] PubMed E-utilities — per-journal article count validation
    → Uses the Groq-built MeSH query filtered to each journal
         │
         ▼
[6] Groq LLM — annotates each journal
    → scope fit, tier (Reach/Target/Safety), red flags, APC note
         │
         ▼
[7] Streamlit UI — sortable table + per-journal links + CSV download
```

---

## Data source comparison

| Column | Source | Key for submission decisions |
|---|---|---|
| SJR / Q (Scopus) | Scimago CSV | Tells you Scopus indexation + quartile |
| IF proxy | OpenAlex 2yr citedness | Rough IF equivalent (not JCR) |
| H-index | Scimago CSV | Journal prestige |
| OA / DOAJ | OpenAlex | Whether free to publish / read |
| APC (USD) | OpenAlex | Actual APC if known |
| OA hits (OA) | OpenAlex group_by | How many open-access papers on your topic |
| PubMed hits | PubMed E-utilities | MEDLINE-validated count with MeSH query |
| Jane conf. % | Jane biosemantics | Semantic similarity to your text (0–100%) |

---

## Troubleshooting

**PubMed hits = -1** → timeout or rate limit; re-run, or add an NCBI API key.

**Jane returns 0 results** → Jane's server may be temporarily slow; the link at the bottom of results opens Jane directly in your browser.

**Scimago columns empty** → `scimagojr_2024.csv` not found next to `app.py`. Download and rename as described above.

**Groq error 401** → Check API key in `app.py` line 18.

**No journals found** → Try broader keywords or increase the publication window.
