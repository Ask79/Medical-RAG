#!/usr/bin/env python3
import argparse, json, uuid
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

def now_utc_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                yield json.loads(line)

def load_spl_to_rx(dm_rx_path: Path):
    spl2rx = defaultdict(list)
    for r in read_jsonl(dm_rx_path):
        ssid = r.get("spl_set_id")
        if not ssid: continue
        rows = []
        pri = r.get("primary") or {}
        if pri.get("rxcui"): rows.append(pri)
        for sec in (r.get("secondary") or []):
            if sec.get("rxcui"): rows.append(sec)
        for x in rows:
            spl2rx[ssid].append({
                "rxcui": str(x["rxcui"]),
                "tty": str(x.get("tty") or ""),
                "has_ndc": bool(x.get("has_ndc", False)),
                "via": str(x.get("via") or "spl_set_id"),
                "score": x.get("score", 0),
            })
    return spl2rx

def main():
    ap = argparse.ArgumentParser(description="Join DailyMed→SNOMED links with DailyMed→RxNorm SPL mapping")
    ap.add_argument("--dm-snomed-links", required=True, help="chunk_condition_links.jsonl")
    ap.add_argument("--daily-rx-links",  required=True, help="daily_rx_links_resolved.jsonl")
    ap.add_argument("--out-chunk", required=True, help="rxcui_condition_links.jsonl")
    ap.add_argument("--out-rollup", required=True, help="rxcui_condition_rollup.jsonl")
    ap.add_argument("--min-conf", type=float, default=0.0, help="min confidence to emit")
    args = ap.parse_args()

    dm_sn_path = Path(args.dm_snomed_links)
    dm_rx_path = Path(args.daily_rx_links)
    out_chunk  = Path(args.out_chunk)
    out_rollup = Path(args.out_rollup)
    out_chunk.parent.mkdir(parents=True, exist_ok=True)
    out_rollup.parent.mkdir(parents=True, exist_ok=True)

    print("Loading SPL→RxNorm map...")
    spl2rx = load_spl_to_rx(dm_rx_path)
    print(f"  SPLs with RxNorm: {len(spl2rx):,}")

    print("Joining to produce RxNorm↔SNOMED chunk links...")
    now = now_utc_iso()
    emitted = set()  # (chunk_id, rxcui, snomed_id, link_type)
    roll = {}        # (rxcui, snomed_id, link_type) -> agg dict
    n_out = 0

    with out_chunk.open("w", encoding="utf-8") as fout:
        for r in read_jsonl(dm_sn_path):
            if float(r.get("confidence", 0)) < args.min_conf: 
                continue
            ssid = r.get("spl_set_id")
            if not ssid or ssid not in spl2rx:
                continue
            for rx in spl2rx[ssid]:
                sig = (r["chunk_id"], rx["rxcui"], r["snomed_id"], r["link_type"])
                if sig in emitted: 
                    continue
                emitted.add(sig)

                out = {
                    "link_id": str(uuid.uuid4()),
                    "rxcui": rx["rxcui"],
                    "tty": rx["tty"],
                    "has_ndc": rx["has_ndc"],
                    "via": rx["via"],
                    "spl_set_id": ssid,
                    "chunk_id": r["chunk_id"],
                    "section": r["section"],
                    "snomed_id": r["snomed_id"],
                    "match_text": r.get("match_text",""),
                    "char_start": r.get("char_start"),
                    "char_end": r.get("char_end"),
                    "link_type": r["link_type"],
                    "confidence": r.get("confidence", 0.0),
                    "method": "join_spl_rxcui.v1",
                    "qualifiers": r.get("qualifiers", {}),
                    "last_updated": now,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                n_out += 1

                k = (rx["rxcui"], r["snomed_id"], r["link_type"])
                a = roll.get(k)
                if not a:
                    a = roll[k] = {
                        "rxcui": rx["rxcui"],
                        "snomed_id": r["snomed_id"],
                        "link_type": r["link_type"],
                        "sections": set(),
                        "spl_set_ids": set(),
                        "chunk_count": 0,
                        "max_confidence": 0.0,
                        "examples": [],
                        "last_updated": now,
                    }
                a["sections"].add(r["section"])
                a["spl_set_ids"].add(ssid)
                a["chunk_count"] += 1
                a["max_confidence"] = max(a["max_confidence"], float(r.get("confidence",0)))
                if len(a["examples"]) < 5:
                    a["examples"].append({"spl_set_id": ssid, "section": r["section"], "text": r.get("match_text","")})

    print(f"  wrote {n_out:,} chunk-level links -> {out_chunk}")

    print("Writing rollup...")
    with out_rollup.open("w", encoding="utf-8") as f:
        for a in roll.values():
            f.write(json.dumps({
                "rxcui": a["rxcui"],
                "snomed_id": a["snomed_id"],
                "link_type": a["link_type"],
                "sections": sorted(a["sections"]),
                "spl_set_ids": sorted(a["spl_set_ids"])[:50],  # cap to keep rows light
                "spl_count": len(a["spl_set_ids"]),
                "chunk_count": a["chunk_count"],
                "max_confidence": round(a["max_confidence"],3),
                "examples": a["examples"],
                "last_updated": a["last_updated"],
            }, ensure_ascii=False) + "\n")
    print(f"  rollup rows: {len(roll):,} -> {out_rollup}")

if __name__ == "__main__":
    main()
