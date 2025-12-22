#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json, os, sqlite3, sys
from collections import Counter, defaultdict

RELA_KEEP = {
    "HAS_INGREDIENT", "INGREDIENT_OF",
    "HAS_DOSE_FORM",  "DOSE_FORM_OF",
    "HAS_TRADENAME",  "TRADE_NAME_OF", "TRADENAME_OF", "BRAND_OF",
}
RXNORM_SAB_ONLY = True

# --- SAT ATN -> canonical keys mapping (raw SAT -> concept-attrs) ---
ATN_MAP = {
    # identifiers
    "NDC11":      ("ndc11s", str),
    "NDC":        ("ndc11s", str),         # will be normalized upstream; here we just collect
    "SPL_SET_ID": ("spl_set_ids", str),
    "SPL_ID":     ("spl_ids", str),
    "UNII":       ("unii", str),

    # route & dose form
    "ROUTE":         ("routes", str),          # human-friendly (MTHSPL)
    "RXN_ROUTE":     ("rxn_routes", str),      # RxNorm controlled route
    "RXN_DOSE_FORM": ("rxn_dose_forms", str),  # RxNorm dose form label
    "RXTERM_FORM":   ("rxn_dose_forms", str),  # RxTerm form label -> same bucket

    # strengths (strings; e.g., "300 MG", "0.5 MG/ML")
    "STRENGTH":                   ("strengths", str),
    "RXN_STRENGTH":               ("strengths", str),
    "RXN_AVAILABLE_STRENGTH":     ("strengths", str),
    "RXN_BOSS_STRENGTH_NUM_VALUE":   ("strengths", str),
    "RXN_BOSS_STRENGTH_NUM_UNIT":    ("strengths", str),
    "RXN_BOSS_STRENGTH_DENOM_VALUE": ("strengths", str),
    "RXN_BOSS_STRENGTH_DENOM_UNIT":  ("strengths", str),
}

# --- Heuristics to fill gaps when SAT is missing ---
ROUTE_PATTERNS = {
    "ORAL":           [" oral ", " by mouth", " po ", "(oral)"],
    "INTRAVENOUS":    [" intravenous ", " iv ", "(iv)"],
    "INTRAMUSCULAR":  [" intramuscular ", " im ", "(im)"],
    "SUBCUTANEOUS":   [" subcutaneous ", " sc ", " subcut ", "(sc)"],
    "TOPICAL":        [" topical ", "(topical)"],
    "OPHTHALMIC":     [" ophthalmic ", " eye ", "(ophthalmic)"],
    "OTIC":           [" otic ", " ear ", "(otic)"],
    "NASAL":          [" nasal ", "(nasal)"],
    "INHALATION":     [" inhalation ", " inhaled ", "(inhalation)"],
    "RECTAL":         [" rectal ", "(rectal)"],
    "VAGINAL":        [" vaginal ", "(vaginal)"],
    "BUCCAL":         [" buccal ", "(buccal)"],
    "SUBLINGUAL":     [" sublingual ", "(sublingual)"],
    "TRANSDERMAL":    [" transdermal ", " patch ", "(transdermal)"],
}

DOSE_FORM_PATTERNS = {
    "Tablet":     [" tablet", " tab ", "(tablet)"],
    "Capsule":    [" capsule", " cap ", "(capsule)"],
    "Solution":   [" solution", " soln ", "(solution)"],
    "Suspension": [" suspension", "(suspension)"],
    "Syrup":      [" syrup", "(syrup)"],
    "Elixir":     [" elixir", "(elixir)"],
    "Injection":  [" injection", " inj ", "(injection)"],
    "Powder":     [" powder", " pwdr ", "(powder)"],
    "Ointment":   [" ointment", " oint ", "(ointment)"],
    "Cream":      [" cream", "(cream)"],
    "Gel":        [" gel", "(gel)"],
    "Patch":      [" patch", "(patch)"],
    "Spray":      [" spray", "(spray)"],
    "Drops":      [" drops", " drop ", "(drops)"],
    "Lotion":     [" lotion", "(lotion)"],
    "Shampoo":    [" shampoo", "(shampoo)"],
    "Foam":       [" foam", "(foam)"],
    "Lozenge":    [" lozenge", "(lozenge)"],
    "Granules":   [" granules", " granule", "(granules)"],
    "Suppository":[" suppository", "(suppository)"],
}

def _infer_routes_from_name(name: str):
    if not name: return []
    s = f" {name.lower()} "
    hits = []
    for canon, needles in ROUTE_PATTERNS.items():
        if any(n in s for n in needles):
            hits.append(canon)
    out, seen = [], set()
    for x in hits:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def _infer_doseform_from_name(name: str):
    if not name: return None
    s = name.lower()
    for canon, needles in DOSE_FORM_PATTERNS.items():
        if any(n in s for n in needles):
            return canon
    # Common RxNorm “… Oral Tablet / … Intravenous Solution” tail patterns
    parts = name.split()
    for i in range(max(0, len(parts) - 2)):
        chunk = " ".join(parts[i:i+2]).lower()
        if any(df in chunk for df in ["tablet","capsule","solution","suspension","injection","patch","cream","gel","ointment"]):
            last = parts[i+1]
            return last[0].upper() + last[1:].lower()
    return None

def _open(path):
    return open(path, "r", encoding="utf-8", errors="replace", newline="")

def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def jdump(o):
    return json.dumps(o, ensure_ascii=False)

def uniq_list(seq):
    seen, out = set(), []
    for x in seq:
        if x is None:
            continue
        k = x if isinstance(x, str) else jdump(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def build_sqlite(db_path, rel_path, sat_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    ensure_dir(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # speed-ups for bulk writes
    cur.execute("PRAGMA journal_mode=OFF")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("PRAGMA temp_store=MEMORY")

    # REL table
    cur.execute("CREATE TABLE rel(src TEXT, rela TEXT, dst TEXT)")
    cur.execute("CREATE INDEX idx_rel_src ON rel(src, rela)")
    cur.execute("CREATE INDEX idx_rel_dst ON rel(dst, rela)")

    # SAT concept-level table
    cur.execute("CREATE TABLE sat(rxcui TEXT PRIMARY KEY, attrs TEXT)")

    # --------- Load REL ---------
    sys.stderr.write("Indexing REL → SQLite...\n")
    read_rel = kept_rel = 0
    with _open(rel_path) as f:
        for line in f:
            read_rel += 1
            if not line.strip():
                continue
            row = json.loads(line)

            sab = (row.get("sab") or "").upper()
            if RXNORM_SAB_ONLY and sab != "RXNORM":
                continue

            rela_raw = (row.get("rela") or row.get("rel") or "").upper()
            if rela_raw not in RELA_KEEP:
                continue

            r1 = (row.get("src_rxcui") or row.get("rxcui1") or "").strip()
            r2 = (row.get("dst_rxcui") or row.get("rxcui2") or "").strip()
            if not (r1.isdigit() and r2.isdigit()):
                continue

            if rela_raw in ("HAS_INGREDIENT", "HAS_DOSE_FORM", "HAS_TRADENAME"):
                src, dst, reln = r1, r2, rela_raw
            else:
                inv = {
                    "INGREDIENT_OF": "HAS_INGREDIENT",
                    "DOSE_FORM_OF":  "HAS_DOSE_FORM",
                    "TRADE_NAME_OF": "HAS_TRADENAME",
                    "TRADENAME_OF":  "HAS_TRADENAME",
                    "BRAND_OF":      "HAS_TRADENAME",
                }
                src, dst, reln = r2, r1, inv[rela_raw]

            cur.execute("INSERT INTO rel(src, rela, dst) VALUES (?,?,?)", (src, reln, dst))
            kept_rel += 1
            if read_rel % 500_000 == 0:
                conn.commit()
                sys.stderr.write(f".. REL read {read_rel:,}, kept {kept_rel:,}\n")

    conn.commit()
    sys.stderr.write(f"REL indexed: kept {kept_rel:,}\n")

    # --------- Load SAT ---------
    sys.stderr.write("Indexing SAT → SQLite...\n")

    peek = None
    with _open(sat_path) as f:
        for l in f:
            if l.strip():
                try:
                    peek = json.loads(l)
                except Exception:
                    pass
                break

    if peek is None:
        sys.stderr.write("SAT empty, skipping.\n")
        return conn

    # Robust detection: RAW if it has raw keys; otherwise treat as aggregated
    has_raw_keys = any(k in peek for k in ("atn", "atv", "attr", "value"))
    is_agg = not has_raw_keys
    sys.stderr.write(f".. detected SAT format: {'AGGREGATED (concept-level)' if is_agg else 'RAW attribute rows → aggregating in SQLite'}\n")

    if is_agg:
        read_sat = kept_sat = 0
        with _open(sat_path) as f:
            for line in f:
                read_sat += 1
                if not line.strip():
                    continue
                obj = json.loads(line)
                rx = obj.get("rxcui")
                if not rx:
                    continue
                cur.execute("INSERT OR REPLACE INTO sat(rxcui, attrs) VALUES (?,?)", (rx, jdump(obj)))
                kept_sat += 1
                if read_sat % 500_000 == 0:
                    conn.commit()
                    sys.stderr.write(f".. SAT read {read_sat:,}, kept {kept_sat:,}\n")
        conn.commit()
        sys.stderr.write(f"SAT indexed: kept {kept_sat:,}\n")

    else:
        # RAW → aggregate in-memory then write compact rows
        buckets = defaultdict(lambda: defaultdict(set))
        read_sat = 0

        def add_attr(rx, atn, atv):
            m = ATN_MAP.get(atn.upper())
            if not m:
                return
            key, cast = m
            try:
                val = cast(atv)
            except Exception:
                val = str(atv)
            if val:
                buckets[rx][key].add(val)

        with _open(sat_path) as f:
            for line in f:
                read_sat += 1
                if not line.strip():
                    continue
                row = json.loads(line)
                rx = (row.get("rxcui") or "").strip()
                atn = (row.get("atn") or row.get("attr") or "").strip()
                atv = (row.get("atv") or row.get("value") or "").strip()
                if not rx or not atn:
                    continue
                add_attr(rx, atn, atv)
                if read_sat % 500_000 == 0:
                    sys.stderr.write(f".. SAT raw read {read_sat:,}, unique CUIs {len(buckets):,}\n")

        kept_sat = 0
        for rx, kv in buckets.items():
            obj = {"rxcui": rx}
            for k, s in kv.items():
                obj[k] = sorted(s)
            cur.execute("INSERT OR REPLACE INTO sat(rxcui, attrs) VALUES (?,?)", (rx, jdump(obj)))
            kept_sat += 1
            if kept_sat % 20_000 == 0:
                conn.commit()
        conn.commit()
        sys.stderr.write(f"SAT aggregated concepts: {kept_sat:,}\n")

    return conn

def rel_targets(cur, rxcui, rela):
    cur.execute("SELECT dst FROM rel WHERE src=? AND rela=?", (rxcui, rela))
    return [r for (r,) in cur.fetchall()]

def sat_get(cur, rxcui):
    cur.execute("SELECT attrs FROM sat WHERE rxcui=?", (rxcui,))
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Enrich RxNorm master with REL + SAT (raw or aggregated).")
    ap.add_argument("--master-in", required=True)
    ap.add_argument("--rel-in", required=True)
    ap.add_argument("--sat-in", required=True)   # accepts raw SAT jsonl OR aggregated SAT-attrs jsonl
    ap.add_argument("--out", required=True)
    ap.add_argument("--db", default="data/tmp/rx_enrich.idx.sqlite")
    ap.add_argument("--prefer-sat-strength", dest="prefer_sat_strength", action="store_true")
    ap.add_argument("--progress-every", type=int, default=20000)
    args = ap.parse_args()

    ensure_dir(args.out)
    ensure_dir(args.db)

    conn = build_sqlite(args.db, args.rel_in, args.sat_in)
    cur = conn.cursor()

    # name/tty maps from master for pretty related entries
    sys.stderr.write("Building name maps from master...\n")
    name_map, tty_map = {}, {}
    with _open(args.master_in) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rx = obj.get("rxcui")
            if rx:
                name_map[rx] = obj.get("preferred_term")
                tty_map[rx]  = obj.get("tty")
    sys.stderr.write(f"Name map entries: {len(name_map):,}\n")

    cov = Counter()
    written = 0
    with _open(args.master_in) as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            base = json.loads(line)
            rxcui = base.get("rxcui")
            if not rxcui:
                continue

            def add_names(rx_list):
                return [{"rxcui": rx, "name": name_map.get(rx), "tty": tty_map.get(rx)} for rx in rx_list]

            ing = rel_targets(cur, rxcui, "HAS_INGREDIENT")
            dfs = rel_targets(cur, rxcui, "HAS_DOSE_FORM")
            bns = rel_targets(cur, rxcui, "HAS_TRADENAME")

            enriched = dict(base)
            if ing:
                enriched["ingredients"] = uniq_list(add_names(ing)); cov["ingredients"] += 1
            if dfs:
                enriched["dose_forms"]  = uniq_list(add_names(dfs)); cov["dose_forms"]  += 1
            if bns:
                enriched["brand_names"] = uniq_list(add_names(bns)); cov["brand_names"] += 1

            sat = sat_get(cur, rxcui) or {}
            for key in ("ndc11s", "spl_set_ids", "spl_ids", "unii", "routes", "rxn_routes", "rxn_dose_forms", "strengths"):
                vals = sat.get(key)
                if vals:
                    enriched[key] = uniq_list(vals); cov[key] += 1

            # prefer SAT strengths if requested
            if args.prefer_sat_strength and sat.get("strengths"):
                enriched["strengths"] = uniq_list(sat["strengths"])

            # ---- Fallback inference from preferred_term when SAT is missing (non-destructive) ----
            pt = (enriched.get("preferred_term") or "")

            # routes: only if not present from SAT
            if not enriched.get("routes"):
                inferred_routes = _infer_routes_from_name(pt)
                if inferred_routes:
                    enriched["routes"] = inferred_routes
                    cov["routes"] += 1  # count fills

            # dose form: prefer rxn_dose_forms; else infer a single canonical label
            has_rxn_df = bool(enriched.get("rxn_dose_forms"))
            if (not has_rxn_df) and (not enriched.get("dose_form")):
                inferred_df = _infer_doseform_from_name(pt)
                if inferred_df:
                    enriched["dose_form"] = inferred_df
                    cov["dose_forms"] += 1

            f_out.write(jdump(enriched) + "\n")
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                sys.stderr.write(f".. wrote {written:,}\n")

    conn.close()

    sys.stderr.write("\n=== Enrichment coverage ===\n")
    for k in ("ingredients","dose_forms","brand_names","ndc11s","spl_set_ids","spl_ids","unii","routes","rxn_routes","rxn_dose_forms","strengths"):
        sys.stderr.write(f"{k:14s}: {cov.get(k,0):,}\n")
    sys.stderr.write(f"\nDone. Wrote {written:,} rows → {args.out}\n")

if __name__ == "__main__":
    main()
