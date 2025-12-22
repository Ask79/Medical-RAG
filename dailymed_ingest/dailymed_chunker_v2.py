#!/usr/bin/env python3
"""
DailyMed Chunker v2 — Boss-schema compliant (revised)

Decisions baked in:
• NDCs: merge product_ndc11 + package_ndc11 into ndc11s (unique)
• Boxed warning: use top-level `boxed_warning_present` if present, else infer from sections
• Optional passthroughs: only the ones in boss schema (no title/manufacturer)
• effective_time: derive from `effective_date` as ISO YYYY-MM-DD when possible
• route: coerce to array[str] if master uses string

Robustness:
• Handles JSONL and a single JSON array file
• Extracts sections from nested shape: sections.unclassified.{...}
• Accepts dict/list/paragraphs for section text
• Windowed chunking with overlap and gentle sentence boundary nudge
• Per-(SPL, section) dedupe
• Deterministic chunk_id (UUIDv5) when --stable-ids true
"""

import argparse
import json
import os
import re
import sys
import uuid
import hashlib
from datetime import datetime

# ------------------------------
# Globals / constants
# ------------------------------

NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-med/chunk:v2")
RUN_ID = str(uuid.uuid4())

SENTENCE_END_RE = re.compile(r"[\.\!\?][\)\]\"']?\s|\n\n")
MULTISPACE_RE = re.compile(r"\s+")

# ------------------------------
# Logging / small utils
# ------------------------------

def log(msg):
    print(msg, file=sys.stderr, flush=True)

def norm_space(text):
    return MULTISPACE_RE.sub(" ", text).strip()

def to_snake_case(name):
    if not name:
        return name
    s = name.strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = MULTISPACE_RE.sub(" ", s)
    s = s.replace(" ", "_")
    return s

def normalize_section_key(key):
    if not key:
        return key
    return to_snake_case(key)

def sha1_hex(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def det_uuid5_chunk_id(spl_set_id, section_path, char_start, char_end, text):
    name = f"{spl_set_id}|{'>' .join(section_path)}|{char_start}|{char_end}|{sha1_hex(norm_space(text))[:8]}"
    return str(uuid.uuid5(NAMESPACE, name))

def parse_effective_date(val):
    """Return ISO YYYY-MM-DD if val looks like YYYYMMDD; else return val as-is."""
    if isinstance(val, str) and len(val) == 8 and val.isdigit():
        try:
            dt = datetime.strptime(val, "%Y%m%d")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return val
    return val

# ------------------------------
# Master-field candidates (tolerant)
# ------------------------------

CANDIDATES = {
    "spl_set_id": ["spl_set_id", "SPL_SET_ID", "splSetId", "set_id", "SET_ID"],
    # include "id" because your masters use it as version identifier
    "spl_version_id": ["spl_version_id", "SPL_VERSION_ID", "splVersionId", "version_id", "VERSION_ID", "id"],

    "sections": ["sections", "label_sections", "LABEL_SECTIONS"],

    "brand_name": ["brand_name", "brand", "brandName", "brand_names", "brandNames"],
    "generic_name": ["generic_name", "generic", "genericName"],

    "route": ["route", "routes"],
    "dosage_form": ["dosage_form", "dosageForm", "form"],
    "strengths": ["strengths", "strength", "dose_strengths"],

    # merge these into ndc11s
    "ndc11s": ["ndc11s", "ndc11_list", "NDC11", "ndc11", "ndcs", "product_ndc11", "package_ndc11"],

    "upcs": ["upcs", "upc"],
    "unii": ["unii", "UNII", "uniis", "UNII_list"],
    "source_url": ["source_url", "dailymed_url", "url"],

    # normalize `effective_date` to effective_time
    "effective_time": ["effective_time", "effectiveTime", "effective_date"],
    "last_updated": ["last_updated", "lastUpdate", "last_updated_time"],
}

def get_first(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

# ------------------------------
# Section extraction
# ------------------------------

def extract_sections(record):
    """
    Return { canonical_section_name: text }.

    Accepts:
      - sections: { "unclassified": { "INDICATIONS": "...", ... }, ... }
      - sections: { "INDICATIONS AND USAGE": <str|dict|list>, ... }
      - sections: [ { name/title, text/body/content/paragraphs }, ... ]
      - Fallback: top-level known section-like keys
    """
    def coerce_to_text(val):
        if val is None:
            return None
        if isinstance(val, str):
            return val.strip() or None
        if isinstance(val, list):
            acc = []
            for x in val:
                if isinstance(x, str):
                    if x.strip():
                        acc.append(x.strip())
                elif isinstance(x, dict):
                    s = x.get("text") or x.get("content") or x.get("body")
                    if isinstance(s, list):
                        s = "\n\n".join([p for p in s if isinstance(p, str) and p.strip()])
                    if isinstance(s, str) and s.strip():
                        acc.append(s.strip())
            return "\n\n".join(acc) if acc else None
        if isinstance(val, dict):
            s = val.get("text") or val.get("content") or val.get("body")
            if s is None and isinstance(val.get("paragraphs"), list):
                s = "\n\n".join([p for p in val["paragraphs"] if isinstance(p, str) and p.strip()])
            if s is None and isinstance(val.get("value"), str):
                s = val["value"]
            if isinstance(s, list):
                s = "\n\n".join([p for p in s if isinstance(p, str) and p.strip()])
            if isinstance(s, str) and s.strip():
                return s.strip()
            return None
        return None

    sections_raw = get_first(record, CANDIDATES.get("sections", [])) or {}

    out = {}

    # Case 0: your common shape -> sections: { "unclassified": { ... } }
    if isinstance(sections_raw, dict) and "unclassified" in sections_raw and isinstance(sections_raw["unclassified"], dict):
        inner = sections_raw["unclassified"]
        for k, v in inner.items():
            text = coerce_to_text(v)
            if text:
                out[normalize_section_key(str(k))] = text

    # Case 1: sections is a dict of sections directly
    elif isinstance(sections_raw, dict):
        for k, v in sections_raw.items():
            text = coerce_to_text(v)
            if text:
                out[normalize_section_key(str(k))] = text

    # Case 2: sections is a list of dicts
    elif isinstance(sections_raw, list):
        for item in sections_raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("section") or item.get("title")
            if not name:
                continue
            text = (
                item.get("text")
                or item.get("content")
                or item.get("body")
                or coerce_to_text(item.get("value"))
                or coerce_to_text(item.get("paragraphs"))
            )
            if text is None:
                text = coerce_to_text(item)
            if text:
                out[normalize_section_key(str(name))] = text

    # Fallback to top-level keys, just in case
    if not out:
        for k in list(record.keys()):
            low = to_snake_case(k)
            if low in {
                "indications_and_usage",
                "dosage_and_administration",
                "warnings",
                "warnings_and_precautions",
                "contraindications",
                "adverse_reactions",
                "precautions",
                "boxed_warning",
            }:
                text = coerce_to_text(record[k])
                if text:
                    out[low] = text

    return out

def has_boxed_warning_in_sections(sections):
    for k in sections.keys():
        if "boxed_warning" in k:
            return True
    return False

def choose_primary_brand(brand_field):
    if brand_field is None:
        return None
    if isinstance(brand_field, list):
        return brand_field[0] if brand_field else None
    return str(brand_field)

# ------------------------------
# Chunking helpers
# ------------------------------

def find_sentence_boundary(section_text, start, end):
    """Adjust end index to the previous sentence boundary if possible."""
    if end >= len(section_text):
        return len(section_text)
    window_start = max(start, end - 240)
    window = section_text[window_start:end]
    matches = list(SENTENCE_END_RE.finditer(window))
    if matches:
        pos = window_start + matches[-1].end()
        return min(pos, len(section_text))
    return end

def make_chunks_for_section(
    spl_set_id,
    spl_version_id,
    section_key,
    section_text,
    target_chars,
    max_chars,
    overlap_chars,
    stable_ids,
    dedupe,
    dedupe_set=None,
    estimate_tokens=False,
):
    text = section_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # collapse triple+ blank lines
    text = text.strip()
    if not text:
        return []

    section_path = [section_key]
    idx = 0
    chunk_index = 0
    seen_hashes = dedupe_set if dedupe_set is not None else set()

    while idx < len(text):
        end = min(idx + target_chars, len(text))
        end = min(end, idx + max_chars)
        end = find_sentence_boundary(text, idx, end)
        if end <= idx:
            end = min(idx + max_chars, len(text))
        chunk_text = text[idx:end].strip()
        if not chunk_text:
            break

        # Dedupe within this SPL+section
        chunk_norm = norm_space(chunk_text).lower()
        h = sha1_hex(chunk_norm)
        if dedupe and h in seen_hashes:
            next_idx = end - overlap_chars if overlap_chars > 0 else end
            if next_idx <= idx:
                next_idx = end
            idx = min(max(next_idx, idx + 1), len(text))
            continue
        seen_hashes.add(h)

        char_start = idx
        char_end = end

        token_count = None
        if estimate_tokens:
            token_count = max(1, round(len(chunk_text) / 4))

        if stable_ids:
            chunk_id = det_uuid5_chunk_id(spl_set_id, section_path, char_start, char_end, chunk_text)
        else:
            chunk_id = str(uuid.uuid4())

        yield {
            "spl_set_id": spl_set_id,
            "spl_version_id": spl_version_id,
            "chunk_id": chunk_id,
            "section": section_key,
            "section_path": section_path,
            "text": chunk_text,
            "char_start": char_start,
            "char_end": char_end,
            "chunk_index": chunk_index,
            "overlap_chars": 0 if chunk_index == 0 else overlap_chars,
            "token_count": token_count,
        }

        # Advance with overlap
        next_idx = end - overlap_chars if overlap_chars > 0 else end
        if next_idx <= idx:
            next_idx = end
        idx = min(max(next_idx, idx + 1), len(text))
        chunk_index += 1

# ------------------------------
# Main pipeline
# ------------------------------

def process(
    in_path,
    out_path,
    sections_filter,
    target_chars,
    max_chars,
    overlap_chars,
    dedupe,
    stable_ids,
    estimate_tokens,
    limit=None,
):
    total_records = 0
    total_chunks = 0
    parse_errors = 0
    found_any_sections = 0

    sections_filter_norm = set(normalize_section_key(s) for s in sections_filter) if sections_filter else None

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Detect JSON array vs JSONL
    is_array_file = False
    with open(in_path, "r", encoding="utf-8") as fin_peek:
        head = fin_peek.read(2048).lstrip()
        if head.startswith("["):
            is_array_file = True

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        if is_array_file:
            log("INFO: Detected JSON array; loading whole file (may be memory-heavy).")
            try:
                data = json.load(fin)
            except json.JSONDecodeError:
                log("ERROR: Could not parse input as JSON array.")
                return 0, 0
            records_iter = (r for r in data if isinstance(r, dict))
        else:
            def iter_jsonl(f):
                nonlocal parse_errors
                for raw in f:
                    if not raw.strip():
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue
                    if isinstance(rec, dict):
                        yield rec
            records_iter = iter_jsonl(fin)

        for rec in records_iter:
            total_records += 1

            spl_set_id = get_first(rec, CANDIDATES["spl_set_id"]) or ""
            spl_version_id = get_first(rec, CANDIDATES["spl_version_id"]) or ""
            if not spl_set_id or not spl_version_id:
                if limit and total_records >= limit:
                    break
                continue

            sections = extract_sections(rec)
            if sections:
                found_any_sections += 1
            if not sections:
                if limit and total_records >= limit:
                    break
                continue

            # has_boxed_warning: prefer top-level if present
            has_boxed = has_boxed_warning_in_sections(sections)
            if "boxed_warning_present" in rec:
                try:
                    has_boxed = bool(rec["boxed_warning_present"])
                except Exception:
                    pass

            # Optional passthroughs per agreed schema
            brand_raw = get_first(rec, CANDIDATES["brand_name"])  # may be list or str
            brand_name = choose_primary_brand(brand_raw)
            generic_name = get_first(rec, CANDIDATES["generic_name"]) or None

            route_val = get_first(rec, CANDIDATES["route"])
            if isinstance(route_val, str) and route_val.strip():
                route = [route_val.strip().lower()]
            elif isinstance(route_val, list):
                route = [str(x).strip().lower() for x in route_val if str(x).strip()]
                route = route if route else None
            else:
                route = None

            dosage_form = get_first(rec, CANDIDATES["dosage_form"]) or None
            strengths = get_first(rec, CANDIDATES["strengths"]) or None

            # merge ndc arrays to ndc11s
            ndc_candidates = get_first(rec, CANDIDATES["ndc11s"])
            ndc11s = None
            if ndc_candidates is not None:
                # could be list or a single list hidden in dict
                vals = []
                if isinstance(ndc_candidates, list):
                    vals.extend(ndc_candidates)
                elif isinstance(ndc_candidates, str):
                    vals.append(ndc_candidates)
                elif isinstance(ndc_candidates, dict):
                    for v in ndc_candidates.values():
                        if isinstance(v, list):
                            vals.extend(v)
                        elif isinstance(v, str):
                            vals.append(v)
                # Also specifically look for both product/package arrays if present
                for k in ("product_ndc11", "package_ndc11"):
                    if k in rec and isinstance(rec[k], list):
                        vals.extend(rec[k])
                # normalize
                vals = [v.strip() for v in vals if isinstance(v, str) and v.strip()]
                if vals:
                    # unique but keep order
                    seen = set()
                    uniq = []
                    for v in vals:
                        if v not in seen:
                            seen.add(v)
                            uniq.append(v)
                    ndc11s = uniq

            upcs = get_first(rec, CANDIDATES["upcs"]) or None
            unii = get_first(rec, CANDIDATES["unii"]) or None
            source_url = get_first(rec, CANDIDATES["source_url"]) or None
            effective_time = parse_effective_date(get_first(rec, CANDIDATES["effective_time"]))
            last_updated = get_first(rec, CANDIDATES["last_updated"]) or None

            # Per-section dedupe set
            per_section_seen = {}

            for sec_key_raw, sec_text in sections.items():
                sec_key = normalize_section_key(sec_key_raw)
                if sections_filter_norm and sec_key not in sections_filter_norm:
                    continue
                if not isinstance(sec_text, str) or not sec_text.strip():
                    continue

                seen = per_section_seen.setdefault(sec_key, set()) if dedupe else None

                for chunk in make_chunks_for_section(
                    spl_set_id=spl_set_id,
                    spl_version_id=spl_version_id,
                    section_key=sec_key,
                    section_text=sec_text,
                    target_chars=target_chars,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                    stable_ids=stable_ids,
                    dedupe=dedupe,
                    dedupe_set=seen,
                    estimate_tokens=estimate_tokens,
                ):
                    out_rec = {
                        **chunk,
                        "source": "dailymed",
                        "has_boxed_warning": bool(has_boxed),
                    }
                    if brand_name is not None:
                        out_rec["brand_name"] = brand_name
                    if generic_name is not None:
                        out_rec["generic_name"] = generic_name
                    if route is not None:
                        out_rec["route"] = route
                    if dosage_form is not None:
                        out_rec["dosage_form"] = dosage_form
                    if strengths is not None:
                        out_rec["strengths"] = strengths
                    if ndc11s is not None:
                        out_rec["ndc11s"] = ndc11s
                    if upcs is not None:
                        out_rec["upcs"] = upcs
                    if unii is not None:
                        out_rec["unii"] = unii
                    if source_url is not None:
                        out_rec["source_url"] = source_url
                    if effective_time is not None:
                        out_rec["effective_time"] = effective_time
                    if last_updated is not None:
                        out_rec["last_updated"] = last_updated

                    out_rec["extras"] = {"run_id": RUN_ID}

                    print(json.dumps(out_rec, ensure_ascii=False), file=fout)
                    total_chunks += 1

            if limit and total_records >= limit:
                break

    if parse_errors:
        log(f"WARN: JSON parse errors encountered: {parse_errors}")
    log(f"INFO: records_with_sections={found_any_sections} of {total_records}")
    return total_records, total_chunks

# ------------------------------
# CLI
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DailyMed Chunker v2")
    p.add_argument("--in", dest="in_path", required=True, help="Input master JSON(L) path")
    p.add_argument("--out", dest="out_path", required=True, help="Output chunks JSONL path")
    p.add_argument("--sections", nargs="*", default=[], help="Sections to include (names are normalized to snake_case)")
    p.add_argument("--target-chars", type=int, default=1100)
    p.add_argument("--max-chars", type=int, default=1800)
    p.add_argument("--overlap-chars", type=int, default=160)
    p.add_argument("--dedupe", type=str, default="true", help="true/false")
    p.add_argument("--stable-ids", type=str, default="true", help="true (UUIDv5) or false (UUIDv4)")
    p.add_argument("--estimate-tokens", type=str, default="false", help="true/false to compute token_count via heuristic")
    p.add_argument("--limit", type=int, default=None, help="Optional: limit number of master records to process")
    return p.parse_args()

def str2bool(s):
    return str(s).strip().lower() in {"1", "true", "t", "yes", "y"}

def main():
    args = parse_args()
    in_path = args.in_path
    out_path = args.out_path
    sections = args.sections
    target_chars = args.target_chars
    max_chars = args.max_chars
    overlap_chars = args.overlap_chars
    dedupe = str2bool(args.dedupe)
    stable_ids = str2bool(args.stable_ids)
    estimate_tokens = str2bool(args.estimate_tokens)
    limit = args.limit

    if not os.path.exists(in_path):
        log(f"ERROR: input not found: {in_path}")
        sys.exit(2)

    if target_chars <= 0 or max_chars <= 0 or overlap_chars < 0:
        log("ERROR: target/max/overlap must be positive (overlap >= 0)")
        sys.exit(2)
    if max_chars < target_chars:
        log("WARN: max_chars < target_chars; using max_chars for window size")

    log("DailyMed Chunker v2 starting…")
    log(f"run_id: {RUN_ID}")
    log(f"in: {in_path}")
    log(f"out: {out_path}")
    log(f"sections (filter, normalized): { [normalize_section_key(s) for s in sections] if sections else '[ALL AVAILABLE]'}")
    log(f"target={target_chars}, max={max_chars}, overlap={overlap_chars}, dedupe={dedupe}, stable_ids={stable_ids}, estimate_tokens={estimate_tokens}")

    recs, chks = process(
        in_path=in_path,
        out_path=out_path,
        sections_filter=sections,
        target_chars=target_chars,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        dedupe=dedupe,
        stable_ids=stable_ids,
        estimate_tokens=estimate_tokens,
        limit=limit,
    )

    log(f"DONE. master_records_processed={recs}, chunks_written={chks}")

if __name__ == "__main__":
    main()
