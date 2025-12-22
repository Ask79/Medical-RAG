# Data (Local Only)

Large datasets and generated artifacts are intentionally NOT included in this repository.

## Sources used during development
- DailyMed (FDA drug labels)
- RxNorm (RRF files)
- SNOMED CT (release files)

## Expected local layout (example)
data/
├── raw/            # downloaded source files (DailyMed/RxNorm/SNOMED)
├── interim/         # normalized/chunked intermediate outputs
└── processed/       # final outputs used by runtime (links, etc.)

## Key generated outputs
- daily_rx_links_resolved.jsonl
  Built by the link-resolver script (e.g., `resolve_links`) and used at runtime
  via an environment variable (e.g., `SPL_RXCUI_PATH`).

## Note
Because these files are large and update over time, they should remain outside version control.

## Licensing note
Datasets used during development are available from official public sources. 
Some resources may require users to obtain access under applicable terms or licenses.

Official sources:
- DailyMed: https://dailymed.nlm.nih.gov/
- RxNorm: https://www.nlm.nih.gov/research/umls/rxnorm/
- SNOMED CT (US Edition): https://www.nlm.nih.gov/healthit/snomedct
