"""Builds cache/bpi2017_mined.pkl — run ONCE offline: python build_cache.py"""
import pickle, time
from pathlib import Path
import pm4py
import pipeline as pl

PATH = "data/BPI Challenge 2017_1_all/BPI Challenge 2017.xes.gz"
COL_MAP = {"case:concept:name": "case_id", "concept:name": "activity",
           "time:timestamp": "timestamp", "org:resource": "resource"}
TIMEBOX_FILTER = False

t0 = time.time()
print("1/4 ingesting (5-15 min for 1.2M events)...")
df_clean = pl.ingest(PATH, COL_MAP, keep_attrs=["case:RequestedAmount"])
print(f"    {len(df_clean):,} events, {df_clean['case_id'].nunique():,} cases [{time.time()-t0:.0f}s]")

df_pm = pl.to_pm(df_clean)
if TIMEBOX_FILTER:
    df_pm = pm4py.filter_variants_top_k(df_pm, 20)
    print("    (filtered to top-20 variants for conformance)")

print("2/4 mining petri net...")
net, im, fm = pm4py.discover_petri_net_inductive(df_pm)

print("3/4 conformance (this is the slow one — let it run)...")
fit = pm4py.fitness_token_based_replay(df_pm, net, im, fm)["log_fitness"]
prec = pm4py.precision_token_based_replay(df_pm, net, im, fm)
print(f"    fitness={fit:.4f} precision={prec:.4f}")

print("3.5/4 rendering process maps (top-10 + full)...")
maps = {}
for key, k in [("top10", 10), ("full", None)]:
    png = pl.discover_map(df_clean, top_k_variants=k)
    maps[key] = Path(png).read_bytes()

print("4/4 writing cache...")
Path("cache").mkdir(exist_ok=True)
with open("cache/bpi2017_mined.pkl", "wb") as f:
    pickle.dump({"df_clean": df_clean, "net": net, "im": im, "fm": fm,
                 "fitness": fit, "precision": prec, "maps": maps}, f)
print(f"DONE in {(time.time()-t0)/60:.1f} min — cache/bpi2017_mined.pkl overwritten with real metrics.")