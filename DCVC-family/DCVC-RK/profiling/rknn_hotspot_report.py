import json, re
H = json.load(open("/tmp/hotspot.json"))
def gm(x): return f"{x/1e9:7.2f}G"

subs = sorted(H.items(), key=lambda kv:-kv[1]["total_macs"])
grand = sum(s["total_macs"] for _,s in subs)
intra = sum(s["total_macs"] for n,s in subs if n.startswith("intra") or n.startswith("y_spatial"))
inter = grand - intra

print("="*80)
print("SUBNET HOTSPOT RANKING  (predicted compute = MACs; TC_COST compute_us ~ MACs)")
print("="*80)
print(f"{'#':<3}{'subnet':<28}{'MACs':>10}{'%tot':>7}{'convs':>6}{'fb':>3}   biggest single op")
print("-"*80)
for i,(n,s) in enumerate(subs,1):
    top=max(s["ops"], key=lambda o:o["macs"])
    short=top["op"].replace("Conv:/","").replace("/Conv","")
    print(f"{i:<3}{n:<28}{gm(s['total_macs']):>10}{100*s['total_macs']/grand:>6.1f}%{s['nconv']:>6}{s['nfb']:>3}"
          f"   {short} {gm(top['macs'])} [{top['kind']}]")
print("-"*80)
print(f"    TOTAL{'':<22}{gm(grand):>10}        (1 I-frame + 1 P-frame; per-frame budget splits below)")
print(f"    intra (I-frame) {gm(intra):>10}  inter (P-frame) {gm(inter):>10}")

allops=[(n,o) for n,s in H.items() for o in s["ops"]]
allops.sort(key=lambda x:-x[1]["macs"])
print("\n"+"="*80)
print("TOP-12 SINGLE-OP HOTSPOTS")
print("="*80)
print(f"{'MACs':>9}  subnet : op")
print("-"*80)
for n,o in allops[:12]:
    short=o["op"].replace("Conv:/","").replace("/Conv","")
    print(f"{gm(o['macs']):>9}  {n} : {short}  [{o['kind']}]{'  TILED' if o['tiled'] else ''}")

print("\n"+"="*80)
print("COMPUTE BY OP-KIND (global)")
print("="*80)
bk={}
for n,s in H.items():
    for o in s["ops"]:
        bk.setdefault(o["kind"],[0,0]); bk[o["kind"]][0]+=o["macs"]; bk[o["kind"]][1]+=1
for k,(m,c) in sorted(bk.items(), key=lambda x:-x[1][0]):
    print(f"  {k:<11}{gm(m):>10}  {100*m/grand:5.1f}%   x{c}")
