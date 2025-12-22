import os, json, csv, argparse, collections, re

GENERIC_PAT = re.compile(r"\b(disease|disorder|finding|symptom|syndrome|abnormality|reaction|adverse|related|condition|NOS|unspecified)\b", re.I)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollup", required=True, help="data/links/rxcui_condition_rollup.jsonl")
    ap.add_argument("--snomed", default=None, help="Optional: SNOMED concepts file with PT/FSN (jsonl)")
    ap.add_argument("--out", required=True, help="CSV output (stats)")
    ap.add_argument("--suggest_blocklist", default=None, help="Optional: write a TXT of top generic IDs")
    ap.add_argument("--top_n", type=int, default=50)
    args = ap.parse_args()

    pt = {}
    if args.snomed and os.path.exists(args.snomed):
        with open(args.snomed,"r",encoding="utf-8") as f:
            for line in f:
                try: o=json.loads(line)
                except: continue
                cid = str(o.get("conceptId") or o.get("sctid") or "")
                if not cid: continue
                nm  = o.get("pt") or o.get("preferredTerm") or o.get("name") or ""
                fsn = o.get("fsn") or o.get("fullySpecifiedName") or ""
                pt[cid] = (nm, fsn)

    stats = {}
    # collect degree and basic stats from rollup
    by_sctid = collections.defaultdict(lambda: {
        "degree_rx": set(),
        "edge_count": 0,
        "sum_conf": 0.0,
        "sum_spl": 0,
        "link_types": collections.Counter(),
        "has_negated_only": True,   # will be revised if any example negated==False
        "examples_seen": 0
    })

    with open(args.rollup,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: o=json.loads(line)
            except: continue
            rxcui = str(o.get("rxcui","")).strip()
            sctid = str(o.get("snomed_id", o.get("sctid",""))).strip()
            if not (rxcui and sctid): continue
            d = by_sctid[sctid]
            d["degree_rx"].add(rxcui)
            d["edge_count"] += 1
            d["sum_conf"]   += float(o.get("max_confidence",0.0) or 0.0)
            d["sum_spl"]    += int(o.get("spl_count",0) or 0)
            d["link_types"].update([str(o.get("link_type","")).lower()])
            exs = o.get("examples") or []
            for ex in exs:
                if isinstance(ex, dict) and "negated" in ex:
                    d["examples_seen"] += 1
                    if ex.get("negated") is False:
                        d["has_negated_only"] = False

    rows = []
    for sctid, d in by_sctid.items():
        nm, fsn = pt.get(sctid, ("",""))
        text = (nm or fsn or "")
        name_is_generic = True if (text and GENERIC_PAT.search(text)) else False
        rows.append({
            "snomed_id": sctid,
            "pt": nm,
            "fsn": fsn,
            "degree_rx": len(d["degree_rx"]),
            "edge_count": d["edge_count"],
            "mean_max_conf": (d["sum_conf"] / max(d["edge_count"],1)),
            "mean_spl_count": (d["sum_spl"] / max(d["edge_count"],1)),
            "top_link_types": ",".join([f"{k}:{v}" for k,v in d["link_types"].most_common(3)]),
            "examples_with_negation_info": d["examples_seen"],
            "all_examples_negated": d["has_negated_only"],
            "name_is_generic": name_is_generic,
        })

    rows.sort(key=lambda r: (r["degree_rx"], r["edge_count"], r["mean_spl_count"]), reverse=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    if args.suggest_blocklist:
        with open(args.suggest_blocklist,"w",encoding="utf-8") as f:
            for r in rows[:args.top_n]:
                f.write(str(r["snomed_id"])+"\n")

if __name__ == "__main__":
    main()
