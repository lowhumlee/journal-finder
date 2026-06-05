# Journal Finder v2

LLM-powered journal discovery tool for academic manuscripts.  
Finds the 15–20 most relevant journals for your paper using live OpenAlex + PubMed data,
annotated and ranked by Groq (Llama 3.3 70B).

---

## Quick start

### 1. Get a free Groq API key
1. Go to https://console.groq.com
2. Sign up (email or Google, no credit card)
3. Create → API Keys → New key
4. Copy the key (starts with `gsk_…`)

### 2. Add your key to the app
Open `app.py` and find line 18:
```python
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"
```
Replace the placeholder with your actual key and save.

### 3. Run the app

**Option A — Streamlit Community Cloud (free, recommended, no local install)**
1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io → sign in → New app.
3. Select repo / branch `main` / file `app.py` → Deploy.
4. Done — permanent public URL, runs in the cloud.

**Option B — Local on Windows 11 without admin rights**
1. If Python isn't already on your PATH, download **WinPython** from
   https://winpython.github.io/ — unzip anywhere, no install needed.
2. Double-click `run_local.bat`.
   First run installs streamlit, requests, pandas into your user profile.
3. App opens at http://localhost:8501.

---

## How it works

```
User input (keywords + optional title/abstract)
        │
        ▼
[1] Groq LLM
    → builds calibrated PubMed query (MeSH + free-text, not too narrow/wide)
    → builds short OpenAlex query
    → extracts 4-6 core concepts
        │
        ▼
[2] OpenAlex API  (free, no key)
    /works?search=<oa_query>&group_by=primary_location.source.id
    → top-N journals ranked by article count in the query window
    → enriches each with: IF proxy, OA status, APC, Scopus/WoS flags, publisher
        │
        ▼
[3] PubMed E-utilities  (free, optional NCBI key for speed)
    → validates article count per journal using the LLM-built PubMed query
    → generates direct PubMed, WoS, Scopus search links per journal
        │
        ▼
[4] Groq LLM (second call)
    → reads journal list + metadata
    → annotates each: scope fit, red flags, APC note, submission tier
    → outputs ranked JSON
        │
        ▼
[5] Streamlit UI
    → sortable table + per-journal expanders with live links
    → CSV export
```

---

## APIs used

| API | Key needed | Free limits | Used for |
|---|---|---|---|
| Groq | Yes (free tier) | 30 RPM / 500K TPD | Search strategy + journal annotation |
| OpenAlex | No | 100K req/day | Journal discovery + metadata |
| PubMed E-utilities | No (optional) | 3 req/s (10 with key) | Article count validation |

---

## Optional: NCBI API key (faster PubMed queries)
Get a free key at https://www.ncbi.nlm.nih.gov/account/
Add it to `app.py` line 19:
```python
NCBI_API_KEY = "your_ncbi_key_here"
```
This raises the PubMed rate limit from 3 to 10 requests/second, reducing search time from ~45s to ~15s for 15 journals.

---

## Interpreting results

**Submission tiers**
- 🔵 **Reach** — high IF or selective journal; worth trying if the paper is strong
- 🟢 **Target** — best fit overall; submit here first
- 🟡 **Safety** — lower IF or less selective; reliable fallback

**Scope fit**
- 🟢 **Excellent** — journal explicitly covers this topic
- 🔵 **Good** — strong thematic overlap
- 🟡 **Moderate** — adjacent field, framing adjustment needed
- 🔴 **Weak** — included for context only; not recommended

**OA count (OpenAlex)** — number of open-access articles matching the query in that journal
over the selected time window. A rough proxy for how actively the journal publishes in your niche.

**PubMed hits** — article count using the LLM-built PubMed query filtered to the journal.
More precise than the OpenAlex count because it uses MeSH terms.

---

## Files

```
journal_finder_v2/
├── app.py              Main Streamlit application (~300 lines)
├── requirements.txt    Python dependencies (3 packages)
├── run_local.bat       Windows launcher, no admin rights needed
└── README.md           This file
```

---

## Troubleshooting

**"Groq error: 401"** → Check your API key in `app.py` line 18.

**"No journals found"** → Try broader keywords, or increase the publication window.

**PubMed hits show -1** → Network timeout or rate limit hit; re-run.

**Streamlit not found after running bat** → The pip user install may not be on PATH.
Open a WinPython Command Prompt and run:
```
pip install streamlit requests pandas
streamlit run app.py
```
