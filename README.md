# Journal Finder v4

## What changed from v3

| v3 problem | v4 fix |
|---|---|
| OpenAlex used for *discovery* → poor results | OpenAlex now used only for *metadata enrichment* after PubMed/Jane discover journals |
| Single search string fired silently | LLM suggests MeSH string shown in editable box; user can modify and re-run |
| Results in one table | 4 tabs: PubMed / Jane / OpenAlex metadata / Merged & Annotated |
| PubMed counted per journal separately (slow) | PubMed fetches up to 3,000 PMIDs then counts by journal name in one pass |

## Setup

### 1. Groq API key (free)
1. Sign up at https://console.groq.com (no credit card)
2. Create an API key (starts `gsk_…`)
3. Edit `app.py` line 14: `GROQ_API_KEY = "gsk_your_key_here"`

### 2. Scimago CSV (one-time, ~5 MB)
1. Go to https://www.scimagojr.com/journalrank.php
2. Scroll to bottom → click **Download** → save as CSV
3. Rename to `scimagojr_2024.csv`, place next to `app.py`

Columns used: `Issn`, `SJR`, `SJR Best Quartile`, `H index`, `Categories`, `Open Access`

Update once a year when Scimago releases new data.

### 3. Optional NCBI API key (speeds up PubMed)
Get free at https://www.ncbi.nlm.nih.gov/account/
Edit `app.py` line 15: `NCBI_API_KEY = "your_key"`
Raises rate limit 3 → 10 req/s.

### 4. Run

**Streamlit Community Cloud (recommended — free, public URL)**
1. Push folder to a public GitHub repo
2. https://share.streamlit.io → New app → select repo/`app.py` → Deploy

**Local, Windows 11, no admin rights**
1. Download WinPython from https://winpython.github.io/ (unzip anywhere)
2. Double-click `run_local.bat`
3. Opens at http://localhost:8501

---

## How it works

```
Sidebar: keywords + optional title/abstract
    │
    ▼
[1] 🧠 Groq LLM
    → identifies 2-3 core MeSH terms
    → builds calibrated PubMed query string
    → generates Jane natural-language text
    → shown in EDITABLE text_area

[2] PubMed + Jane fire in PARALLEL immediately
    │                        │
    ▼                        ▼
PubMed esearch             Jane suggestions.php
→ fetch up to 3,000 PMIDs → parse HTML
→ esummary in batches      → journal name + confidence %
→ count by journal name    → matching article count
→ rank by frequency

User can EDIT the search string → Re-run button fires PubMed + Jane again

[3] Tabs show raw output:
    PubMed tab  → journal | PubMed count (bar)
    Jane tab    → journal | confidence % (bar) | articles
    OpenAlex tab→ lazy-loaded metadata for all discovered journals
                  (ISSN, publisher, OA, DOAJ, APC, IF proxy)

[4] "Build merged table" button:
    → Merge PubMed + Jane lists (deduplicate by name)
    → Score: 55% PubMed count + 45% Jane confidence
    → OpenAlex metadata lookup (name search)
    → Scimago CSV lookup by ISSN
    → Groq annotation (scope fit, tier, red flags)
    → Final ranked table + CSV download
```

---

## Column guide (Merged tab)

| Column | Source | Use |
|---|---|---|
| SJR / Q | Scimago CSV | Scopus indexation + quartile |
| H | Scimago CSV | Journal prestige |
| IF proxy | OpenAlex | 2-year mean citedness (not JCR IF) |
| OA | OpenAlex | Whether journal is open access |
| APC (USD) | OpenAlex | Article processing charge |
| PubMed n | PubMed E-utils | Articles matching your MeSH query |
| Jane % | Jane biosemantics | Semantic similarity score (0–100%) |
| Tier | Groq LLM | Reach / Target / Safety submission tier |
| Scope | Groq LLM | Excellent / Good / Moderate / Weak |
| ⚠ | Groq LLM | Red flags: wrong scope, low SJR, etc. |

---

## Troubleshooting

**PubMed returns empty list** → Check the search string for syntax errors.
Open the PubMed link to validate the query directly.

**Jane returns 0 results** → Jane's server may be slow. Click "Open Jane ↗" link
to run it manually and compare.

**SJR columns all empty** → `scimagojr_2024.csv` not found next to `app.py`.

**Groq error 401** → Check `GROQ_API_KEY` in `app.py` line 14.
