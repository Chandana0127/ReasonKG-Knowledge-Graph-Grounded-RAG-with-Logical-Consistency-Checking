# ReasonKG — Knowledge-Graph-Grounded RAG with Logical Consistency Checking

> A hybrid Retrieval-Augmented Generation system for financial document QA that combines **PageIndex** (reasoning-based, vectorless retrieval), a **Neo4j knowledge graph** of SEC financial metrics, and a **rule-based consistency checker** that detects and auto-corrects hallucinated numerical claims in LLM outputs.

**Course Project · CSE 579 — Knowledge Representation & Reasoning · ASU**
**Result:** ~90% reduction in hallucination rate vs. a vanilla RAG baseline on a 20-query benchmark across 4 companies.

---

## Table of Contents

1. [Motivation](#motivation)
2. [System Architecture](#system-architecture)
3. [Key Results](#key-results)
4. [Repository Structure](#repository-structure)
5. [Tech Stack](#tech-stack)
6. [Setup](#setup)
7. [Running the Pipeline](#running-the-pipeline)
8. [Pipeline Stages](#pipeline-stages)
9. [Evaluation Methodology](#evaluation-methodology)
10. [Skills & Concepts Demonstrated](#skills--concepts-demonstrated)
11. [Limitations & Future Work](#limitations--future-work)
12. [References](#references)

---

## Motivation

LLMs hallucinate numerical facts. In financial QA, a model can fluently report that "Apple's FY2024 revenue was $383 billion" when the verified figure from the 10-K is $391 billion — and a downstream user has no easy way to catch it. Vector-based RAG narrows but does not eliminate this gap, because retrieval similarity does not imply factual grounding.

**ReasonKG** addresses this by routing every numerical claim through a structured knowledge graph populated from SEC EDGAR XBRL and Yahoo Finance, then automatically rewriting any sentence that disagrees with the KG ground truth. The result is a pipeline where:

- Retrieval is **reasoning-driven**, not embedding-driven (via PageIndex)
- Generation is **constrained** by a verifiable KG (Neo4j)
- Output is **self-correcting** — hallucinated values are rewritten in-place using the KG figure

---

## System Architecture

```
                            ┌─────────────────────────────┐
                            │   SEC EDGAR (10-K filings)  │
                            │   • AAPL, MSFT, AMZN, NVDA  │
                            └──────────────┬──────────────┘
                                           │
                  ┌────────────────────────┼────────────────────────┐
                  │                        │                        │
                  ▼                        ▼                        ▼
        ┌──────────────────┐    ┌──────────────────┐     ┌──────────────────┐
        │  XBRL Facts API  │    │  PageIndex Tree  │     │  Chunked HTML    │
        │  + Yahoo Finance │    │  (LLM-navigable) │     │  (FAISS index)   │
        └────────┬─────────┘    └────────┬─────────┘     └────────┬─────────┘
                 │                       │                        │
                 ▼                       │                        │
        ┌──────────────────┐             │                        │
        │  Neo4j KG        │             │                        │
        │  Company         │             │                        │
        │  ├─FILED→Filing  │             │                        │
        │  └─HAS_METRIC→   │             │                        │
        │     FinancialMetric            │                        │
        └────────┬─────────┘             │                        │
                 │                       │                        │
                 │       ┌───────────────┴────────┬───────────────┘
                 │       │                        │
                 │       ▼                        ▼
                 │  ┌──────────────────────────────────────────┐
                 │  │       Hybrid Retrieval Router            │
                 │  │  PageIndex if doc uploaded               │
                 │  │  KG-guided FAISS otherwise               │
                 │  └─────────────────┬────────────────────────┘
                 │                    │
                 │                    ▼
                 │       ┌──────────────────────────┐
                 │       │   LLM (GPT-4o)           │
                 │       └────────────┬─────────────┘
                 │                    │ raw_answer
                 │                    ▼
                 │       ┌──────────────────────────┐
                 │       │   Claim Extractor        │
                 │       │   spaCy + regex          │
                 │       │   metric-first parsing   │
                 │       └────────────┬─────────────┘
                 │                    │ claims[]
                 │                    ▼
                 │       ┌──────────────────────────┐
                 └──────►│   KG Verifier            │
                         │   5% tolerance check     │
                         │   → VERIFIED /           │
                         │     HALLUCINATED /       │
                         │     UNVERIFIABLE         │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Answer Rewriter        │
                         │   Hallucinated sentences │
                         │   regenerated with KG    │
                         │   value substituted in   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                              ┌──────────────┐
                              │ final_answer │
                              └──────────────┘
```

---

## Key Results

Evaluated on **20 queries × 3 systems = 60 runs** across AAPL, MSFT, AMZN, NVDA for fiscal years 2022–2025. Each query asks for a verifiable numerical metric (revenue, net income, gross profit, operating income).

| System | Hallucination Rate ↓ | Verification Coverage ↑ | Avg. Claims / Query |
|---|---|---|---|
| Vanilla RAG (FAISS only) | High | Moderate | ~1.0 |
| KG + PageIndex RAG | Moderate | High | ~1.0 |
| **ReasonKG (full system)** | **~90% lower than Vanilla** | **~100%** | **~1.0** |

Headline finding: **The consistency-checking + auto-correction layer drives nearly the entire hallucination reduction.** Retrieval quality matters for getting the right candidate value into context, but the KG-grounded verification step is what guarantees the final answer agrees with ground truth.

Detailed breakdowns by query difficulty and by company are produced in `figures/` after running the evaluation pipeline (see [Running the Pipeline](#running-the-pipeline)).

---

## Repository Structure

```
reasonkg/
├── README.md                 ← this file
├── ReasonKG_Final.ipynb      ← end-to-end pipeline (Colab-ready)
├── pageindex_doc_map.json    ← {ticker_accession: pageindex_doc_id}
├── fig1_hallucination_rate.png
├── fig2_by_difficulty.png
├── fig3_by_company.png
├── fig4_heatmap.png
├── fig5_dashboard.png
├── eval_results_final.csv    ← per-query, per-system raw results
├── eval_summary_final.csv    ← aggregated metrics
```

> The notebook is the single source of truth for the pipeline. If you want to split it into a Python package (`src/ingest/`, `src/retrieval/`, `src/consistency/`, `src/eval/`), each block in the notebook maps cleanly to one module — see [Pipeline Stages](#pipeline-stages) below.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| **LLM** | GPT-4o (via institutional API gateway) | Precise on numerical reasoning; low cost per call |
| **Retrieval (primary)** | [PageIndex](https://github.com/VectifyAI/PageIndex) (VectifyAI) | Reasoning-based, vectorless — LLM navigates a hierarchical document tree instead of relying on embeddings |
| **Retrieval (fallback)** | FAISS + `BAAI/bge-large-en-v1.5` | KG-guided filtering on top of standard dense retrieval, used when a doc isn't in PageIndex |
| **Knowledge Graph** | Neo4j Aura (managed) | `Company → HAS_METRIC → FinancialMetric` schema, populated from XBRL + Yahoo Finance |
| **KG sources** | SEC EDGAR XBRL API, `yfinance` | XBRL gives audited historical figures (2015+); yfinance fills recent-year gaps |
| **NER / parsing** | spaCy `en_core_web_sm` + regex | Metric-first claim extraction; EPS handled by a dedicated small-number pattern |
| **Document store** | Google Drive (Colab-mounted) | Persists 10-K downloads and the FAISS index across Colab sessions |
| **Eval** | Custom verdict logic (VERIFIED / HALLUCINATED / UNVERIFIABLE) | Tolerance: 5% to absorb rounding between XBRL and yfinance |

---

## Setup

### Prerequisites

- Python 3.10+
- Google Colab (recommended) or a local Jupyter environment with ~8 GB RAM
- A Neo4j instance (Aura free tier works)
- A PageIndex API key — sign up at [VectifyAI](https://vectify.ai)
- An LLM endpoint (OpenAI / Anthropic / a gateway). The notebook is wired to a CreateAI-style endpoint; swap `llm_chat()` for your provider if needed.

### Configure secrets

Set the following as environment variables (or in **Colab → Secrets** if using Colab):

```bash
# LLM gateway
CREATEAI_API_URL=...
CREATEAI_API_TOKEN=...
CREATEAI_PROJECT_ID=...

# Neo4j
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j

# SEC (any contact email works as the User-Agent)
SEC_USER_AGENT=youremail@example.com

# PageIndex
PAGEINDEX_API_KEY=...
```

> **Do not commit credentials.** Use Colab Secrets or a `.env` file in `.gitignore`. Rotate any key that's ever been in a notebook before pushing the repo public.

### Install dependencies

```bash
pip install requests beautifulsoup4 lxml spacy neo4j tqdm \
            pageindex faiss-cpu sentence-transformers yfinance \
            pandas matplotlib openai fpdf2
python -m spacy download en_core_web_sm
```

---

## Running the Pipeline

The notebook is organized as six sections, each runnable independently once setup is done:

| Section | Blocks | What it does | Time |
|---|---|---|---|
| 1. Setup | 1–3 | Install deps, mount Drive, verify LLM/Neo4j connections | ~3 min |
| 2. Data & KG | 4–5 | Populate Neo4j from XBRL + yfinance; download 10-Ks (2020+) | ~5–10 min |
| 3. Retrieval | 6–7 | Build FAISS index; configure PageIndex client | ~2–3 min |
| 4. Consistency Layer | 8–10 | Claim extractor, KG verifier, full orchestrator + 10 edge-case tests | ~1 min |
| 5. Evaluation | 11–12 | 3-way eval (Vanilla / KG+PageIndex / ReasonKG) + 5 figures | ~10 min |
| 6. Final Expansion | Steps 1–5 | 20-query benchmark across 4 companies, hybrid retrieval router | ~15 min |

**Quick start — run the smoke test:**

```python
result = consistency_check(
    query          = "What was Apple's total revenue in fiscal year 2024?",
    context_ticker = "AAPL",
    context_year   = 2024,
)

print(result["final_answer"])           # auto-corrected answer
print(result["verdict_summary"])        # {VERIFIED: n, HALLUCINATED: n, UNVERIFIABLE: n}
print(result["hallucination_rate"])     # 0.0 if all claims agree with KG
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion & KG Construction (Blocks 4–5)

- **`Neo4jStore`** (`init_schema`, `upsert_financial_metric`, `query_metric`, `query_all_metrics`, `query_filings_for_ticker`) — thin Cypher wrapper.
- **`enrich_from_xbrl(ticker, start_year=2015)`** — pulls audited annual figures from SEC's `companyfacts` API, filtered to `form == "10-K"`. Maps XBRL concepts (`Revenues`, `NetIncomeLoss`, `GrossProfit`, `OperatingIncomeLoss`, `EarningsPerShareBasic`) to internal metric keys.
- **`enrich_from_yfinance(ticker)`** — fills the most recent 4 fiscal years from Yahoo Finance to plug XBRL recency gaps.
- **`fix_duplicate_metrics(ticker)`** — post-pass that collapses quarterly sub-totals leaked through the XBRL filter to the single max value per `(metric, year)`.
- **`collect_10k_filings(cik)`** — paginates SEC `submissions` API to find every 10-K since 2020.

**Schema:**
```cypher
(:Company {ticker})
   -[:FILED]-> (:Filing {accession_no})
   -[:HAS_METRIC]-> (:FinancialMetric {metric_id, ticker, metric, year, value, unit})
```

### Stage 2 — KG-Grounded Retrieval (Blocks 6–7)

Three retrieval modes share a common interface — each returns a list of `{ticker, text, ...}` chunks:

1. **`vanilla_rag_retrieve(query)`** — pure FAISS over `BAAI/bge-large-en-v1.5` embeddings of 400-word chunks from selected 10-Ks. The plain baseline.
2. **`pageindex_retrieve(query, kg_candidates, top_docs=1)`** — uses VectifyAI's tree-based reasoning retrieval:
   - `llm_tree_search` asks the LLM to pick 2–3 relevant section IDs from the document tree as JSON.
   - Falls back to a `scored_fallback` heuristic (favoring Item 7 / Item 8, penalizing Risk / Legal sections).
   - Final fallback: first 3 nodes.
3. **`kg_guided_retrieve(query, ticker, top_k=3)`** — confirms the ticker exists in Neo4j, then filters the global FAISS results to that ticker's chunks only. Used when no PageIndex doc is available for the ticker.

**`hybrid_retrieve(query, ticker)`** routes between PageIndex and KG-FAISS based on `doc_map` availability and returns `(chunks, retrieval_method)`.

### Stage 3 — Consistency Checking (Blocks 8–10) — *the novel contribution*

A claim is a tuple `(metric, ticker, year, value)`. The checker extracts every such claim from an LLM answer, looks it up in Neo4j, and produces a verdict.

- **`extract_claims(answer_text, context_ticker, context_year)`** — metric-first parser. For each sentence:
  1. Detect ticker (from sentence or context) and year (from sentence or context).
  2. Find all metric keyword positions (`revenue`, `net income`, `gross profit`, `operating income`, `EPS`, …).
  3. Find all dollar amounts (handles `$X trillion / billion / million`).
  4. For each metric keyword, attach the nearest dollar amount that follows it.
  5. Special path for EPS — values are small decimals like `$6.11`, so they're matched separately from the billion/million patterns.
  6. Deduplicate on `(ticker, metric, year)`.
- **`verify_claim(claim)`** — looks up the KG ground truth and returns one of three verdicts using a 5% tolerance:
  - `VERIFIED` — claim within tolerance of KG value
  - `HALLUCINATED` — claim disagrees with KG by more than tolerance
  - `UNVERIFIABLE` — KG has no value for this `(ticker, metric, year)`
- **`build_corrected_answer(original_answer, verified_claims)`** — for each HALLUCINATED claim, asks the LLM to rewrite the offending sentence with the KG-verified value substituted in. Other sentences pass through unchanged.
- **`consistency_check(query, ...)`** — full orchestrator. Returns `{raw_answer, final_answer, hallucination_rate, verification_rate, verdict_summary, retrieval_method, claims, verified_claims}`.

**Edge-case test suite (10 cases):** billions with decimal, millions with context-supplied year, multiple metrics in one sentence, ticker from sentence text, EPS small numbers, multiple companies in one answer, "totaled" phrasing, sentences with no dollar amount, NVIDIA fiscal-year phrasing, gross profit + operating income together. All pass after the EPS-specific path was added in the v3 iteration of the claim extractor.

### Stage 4 — Evaluation (Blocks 11–12, Step 4)

- **`EVAL_QUERIES_FINAL`** — 20 queries spanning 4 tickers × revenue / net_income / gross_profit / operating_income × FY2022–FY2025, tagged `simple` / `medium` for difficulty stratification.
- **`run_vanilla_eval`, `run_hybrid_eval`, `run_reasonkg_eval`** — wrap each system in a common interface so the same query is evaluated identically across all three.
- **Aggregation** — per-system means + breakdowns by difficulty and by ticker, exported to CSV.

### Stage 5 — Integration & Visualization (Step 5)

Five publication-ready figures (matplotlib, 150 DPI):
1. Overall hallucination rate by system
2. Hallucination rate by query difficulty
3. Hallucination rate by company
4. Per-query heatmap (green = correct, red = hallucinated)
5. Combined summary dashboard (3 panels)

---

## Evaluation Methodology

**Metrics:**

- **Hallucination Rate** = `# HALLUCINATED / # checkable claims` (lower is better; `checkable = VERIFIED + HALLUCINATED`)
- **Verification Coverage** = `# checkable / # total claims` (higher is better — measures how much of the answer the KG can actually verify)
- **Avg. Claims per Query** = sanity check that all three systems are producing comparable answer surface area

**Tolerance:** 5% relative error. Wide enough to absorb rounding differences between XBRL (raw USD) and yfinance (sometimes rounded to thousands), tight enough that genuine errors trip it.

**What this does *not* measure:**
- Semantic correctness of non-numerical claims (qualitative statements, narrative).
- Logical consistency across multiple claims (e.g., revenue – COGS ≠ gross profit).
- Performance under adversarial queries (e.g., questions about metrics the KG doesn't store).

These are deliberate scoping choices documented in the project notes.

---

## Skills & Concepts Demonstrated

- **Knowledge graph design and Cypher querying** — schema choice (Company / Filing / FinancialMetric), constraint setup, idempotent `MERGE`-based upserts, multi-source enrichment with conflict resolution.
- **Retrieval-Augmented Generation architecture** — dense retrieval (FAISS + `bge-large-en-v1.5`), reasoning-based retrieval (PageIndex tree navigation), and hybrid routing between them.
- **Hallucination detection and grounded generation** — a claim extractor that handles compositional cases (multiple metrics per sentence, EPS as a small-number edge case), a tolerance-based KG verifier, and using the LLM as a *corrector* rather than just a *generator*.
- **Working with regulated financial data** — SEC EDGAR submissions API, XBRL company facts, Yahoo Finance, fiscal-year vs calendar-year handling, audit-quality cross-checking between sources.
- **Evaluation rigor** — 3-way controlled comparison on a stratified query set, with breakdowns by company and difficulty; reported both a primary metric (hallucination rate) and a coverage metric (verification rate) to prevent gaming.
- **LLM tooling & orchestration** — prompt engineering for structured JSON outputs (tree-search node IDs), defensive parsing with multi-level fallbacks, and a Colab-friendly persistence layer (Drive-mounted FAISS index + JSON doc map).

---

## Limitations & Future Work

- **Numerical claims only.** The consistency checker validates `(metric, ticker, year, value)` tuples. Qualitative claims and multi-step reasoning chains (e.g., "revenue grew because of iPhone sales") are out of scope.
- **KG coverage = ceiling on verification.** If the KG doesn't have a value for a claim, the verdict is `UNVERIFIABLE` and the claim flows through unchecked. Expanding the schema to cover segment data, balance-sheet items, and YoY growth rates would lift this ceiling significantly.
- **PageIndex coverage is uneven.** Only 2 of 26 filings were uploaded to PageIndex during the project window; the remaining 24 fall back to KG-guided FAISS. A more aggressive upload pipeline would let PageIndex be tested at its full strength.
- **No FinanceBench head-to-head yet.** Running on the published [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) benchmark would make the result directly comparable to published baselines. The current evaluation uses an in-house 20-query set tailored to the KG schema.
- **Single LLM tested.** All three systems use the same GPT-4o backbone. Cross-model evaluation (Llama 3, Claude, etc.) would tease apart how much of the hallucination reduction is LLM-specific vs. architecture-driven.

---

## References

- **PageIndex** — VectifyAI, [github.com/VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)
- **SEC EDGAR XBRL Facts API** — [data.sec.gov/api/xbrl/companyfacts](https://data.sec.gov/api/xbrl/companyfacts)
- **FinanceBench** — Patronus AI, [huggingface.co/datasets/PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench)
- **BAAI/bge-large-en-v1.5** — [huggingface.co/BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **Neo4j Cypher** — [neo4j.com/docs/cypher-manual](https://neo4j.com/docs/cypher-manual)

---

*CSE 579 · Spring 2026 · ASU*
