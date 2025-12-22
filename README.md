# Medical RAG Pipeline

This repository contains a retrieval-augmented generation (RAG) system for answering
drug-related medical questions using authoritative public sources such as **DailyMed**,
**RxNorm**, and **SNOMED CT**.

The project focuses on system design, data integration, and source-grounded retrieval
rather than end-to-end model training or benchmark optimization.

---

## What this project does

- Parses natural-language drug queries and detects intent
- Resolves drug entities across DailyMed and RxNorm identifiers
- Resolves clinical conditions using SNOMED CT concepts
- Retrieves relevant regulatory text (e.g., indications, contraindications)
- Produces structured, citation-backed summaries

---

## Repository organization

This repository is organized by **functional role rather than strict package conventions**.

Folders include:
- **Runtime pipeline code** for query parsing, entity resolution, retrieval, and summarization
- **Ingestion and build scripts** for normalizing DailyMed, RxNorm, and SNOMED CT
- **Linker scripts** that construct cross-source identifier mappings (e.g., DailyMed ↔ RxNorm)

Folder names reflect development stages and roles; see individual directories for details.

---

## Data usage

This repository intentionally **does not include** raw datasets, processed artifacts,
embeddings, or indices.

External data sources used during development:
- DailyMed (FDA drug labels)
- RxNorm (normalized drug identifiers)
- SNOMED CT (clinical terminology)

Ingestion and linker scripts are provided to prepare these datasets locally.
All generated data is expected to live outside of version control.

See `data/README.md` for details.

---

## Evaluation

The system was evaluated on a small, internal set of queries during development.
Evaluation was used to validate pipeline correctness and identify coverage limitations,
rather than to establish a comprehensive or statistically meaningful benchmark.

A large-scale evaluation was out of scope for this project.

---

## Notes

This repository represents a completed prototype emphasizing:
- explicit identifier resolution
- source-grounded retrieval
- deterministic summarization with citations

It is intended as a systems and integration project rather than a production-ready service.

## Quickstart (Sanity Check)
After configuring environment variables and preparing local data, the pipeline can be
exercised with a single query via the module entrypoint:

```bash
python -m rag_med.pipeline_router "Is metformin contraindicated at eGFR 25?"

