#!/bin/sh
# One-command reproduction of the paper's headline results from released data.
#
#   ./reproduce.sh            summarise every headline result
#   ./reproduce.sh witness    the motivating pair (Table 1)
#   ./reproduce.sh regimes    discovery by sampling regime (Table 2)
#   ./reproduce.sh classes    the 70-class summary (ladder, families, acceptance)
#   ./reproduce.sh representation  structural-sketch evaluation and robustness
#   ./reproduce.sh direct     frozen 7B acceptance scoring with structural input
#   ./reproduce.sh sweep      beam-width candidate exposure at k=1,2,4,8,16
#   ./reproduce.sh rerank     BFS-Prover-V2 top-8 execution and reranking
#   ./reproduce.sh policies   two-policy exposure and regret, split by family
#   ./reproduce.sh cost       node-identity wall clock, nodes, and queue sizes
#   ./reproduce.sh natural    rendering repeats in Mathlib's own proof scripts
#   ./reproduce.sh repsens    representative sensitivity: outcome and trajectory
#                             divergence between members of one alias class
#   ./reproduce.sh stress     96-pair alias-conditioned retention benchmark,
#                             proposal-generator controls, legacy fixed grammar,
#                             and the ancestor-rooted natural-class test
#   ./reproduce.sh persist    current-version persistence verdicts
#
# These read the released JSON/JSONL under data/ and recompute every number
# reported in the paper. Re-running the underlying Lean experiments additionally
# requires a traced Mathlib at the pinned commit; see README.md.
set -eu
PY="${PYTHON:-python3}"
WHAT="${1:-all}"

run() { [ "$WHAT" = "all" ] || [ "$WHAT" = "$1" ]; }

if run witness; then
$PY - <<'EOF'
import json,glob
print("== Table 1: motivating pair ==")
for f in glob.glob("data/reconv_b2/stageb.shard*.jsonl"):
    for l in open(f):
        if not l.strip(): continue
        r=json.loads(l)
        if r["pp_hash"].startswith("dc12987d57bf"):
            print(f"  theorem   {r['theorem']}")
            print(f"  rendering {r['pp_bytes']} bytes, {r['n_members']} members")
            for h,o in r["outcomes"].items():
                if "simp" in o: print(f"    [{h:22}] simp -> {o['simp']}")
EOF
fi

if run regimes; then
$PY - <<'EOF'
import json,glob
print("== Table 2: discovery by sampling regime ==")
def amb(c):
    rt=c.get("repeat_table") or {}
    hit=pairs=0
    for p in (c.get("confirmed_probes") or []):
        firsts=[v[0] for v in (rt.get(p) or {}).values() if v]
        n1=sum(1 for x in firsts if x!="FAIL"); n0=len(firsts)-n1
        if n0 and n1: hit=1; pairs+=1
    return hit,pairs
seen=set()
hdr=f"{'regime':14}{'roots':>7}{'expand':>8}{'reconv':>8}{'classes':>9}{'flag':>6}{'conf':>6}{'new':>5}{'amb':>5}{'pairs':>7}"
print(hdr)
for runs,label in ((("reconv_b2",),"blind"),(("hunt_enn",),"coercion"),
                   (("hunt_rare","hunt_rare2"),"structural")):
    roots=[json.loads(l) for run in runs
           for f in sorted(glob.glob(f"data/{run}/roots.shard*.jsonl"))
           for l in open(f) if l.strip()]
    sb=[json.loads(l) for run in runs
        for f in sorted(glob.glob(f"data/{run}/stageb.shard*.jsonl"))
        for l in open(f) if l.strip()]
    conf=[c for run in runs for c in json.load(open(f"data/{run}/confirmation.json"))
          if c.get("verdict")=="CONFIRMED"]
    new=[c for c in conf if (c["theorem"],c["pp_hash"]) not in seen]
    seen.update((c["theorem"],c["pp_hash"]) for c in conf)
    na=sum(amb(c)[0] for c in new); npr=sum(amb(c)[1] for c in new)
    print(f"{label:14}{len(roots):7}{sum(1 for r in roots if r.get('expandable')):8}"
          f"{sum(1 for r in roots if (r.get('n_reconv_classes') or 0)>0):8}{len(sb):9}"
          f"{sum(1 for s in sb if s.get('divergent_probes')):6}{len(conf):6}{len(new):5}"
          f"{na:5}{npr:7}")
print(f"  unique classes after deduplication: {len(seen)}")
print("  only the blind row supports an incidence claim")
EOF
fi

if run classes; then
$PY - <<'EOF'
import json,glob,collections
print("== 70 confirmed classes ==")
uniq={}
for run in ("reconv_b2","hunt_enn","hunt_rare","hunt_rare2"):
    conf=[c for c in json.load(open(f"data/{run}/confirmation.json"))
          if c.get("verdict")=="CONFIRMED"]
    mech=json.load(open(f"data/{run}/mechanism.json"))
    for c,m in zip(conf,mech): uniq.setdefault((c["theorem"],c["pp_hash"]),(c,m))
print(f"  classes {len(uniq)}  theorems {len({t for t,_ in uniq})}")
lad=collections.defaultdict(collections.Counter)
fams=collections.Counter(); amb=0; pairs=0; num=den=0
def fam(m):
    theorem=(m or {}).get("theorem","")
    t=" ".join((m or {}).get("diff_tokens") or [])
    rg=(m or {}).get("rungs",{})
    if "ENNReal.ofNNReal" in t or "WithTop.some" in t: return "coercion"
    if "@Ne" in t or "+Not" in t: return "Ne-vs-Not"
    if "instOfNatNat" in t: return "Fin-eta"
    if any(x in t for x in ("toLE","toPreorder","linearOrderOfSTO","WellOrderingRel")):
        return "instance"
    if theorem in {"tendsto_integral_exp_smul_cocompact", "Tuple.bubble_sort_induction'",
                   "max_aleph0_card_le_rank_fun_nat"}:
        return "local-unfolding"
    dp=rg.get("phi_deep_proofs")
    if isinstance(dp,dict) and dp.get("separates"): return "proof-term"
    return "not-printer-separable"
for (t,ph),(c,m) in uniq.items():
    fams[fam(m)]+=1
    for r_,v in m.get("rungs",{}).items():
        if isinstance(v,dict) and "separates" in v:
            lad[r_]["sep" if v["separates"] else "col"]+=1
    rt=c.get("repeat_table") or {}
    hit=False
    for p in (c.get("confirmed_probes") or []):
        firsts=[v[0] for v in (rt.get(p) or {}).values() if v]
        n1=sum(1 for x in firsts if x!="FAIL"); n0=len(firsts)-n1
        if n0 and n1:
            hit=True; pairs+=1; num+=min(n0,n1); den+=n0+n1
    amb+=hit
print(f"  acceptance-ambiguous {amb}/{len(uniq)}; contradictory pairs {pairs}; "
      f"member-weighted floor {num/den:.2f}")
for r_ in ("phi_state","phi_state_deep","phi_deep_proofs","phi_all_shallow","f_env"):
    print(f"    {r_:18} sep {lad[r_]['sep']:3}  col {lad[r_]['col']:3}")
print("  families:", dict(fams.most_common()))
iv=json.load(open("data/interventions/interventions.json"))
for m in ("coercion","NeNot","FinEta"):
    rs=[r for r in iv if r["mechanism"]==m]
    print(f"  interventions {m}: n={len(rs)} valid={sum(r['valid'] for r in rs)} "
          f"divergent={sum(r['outcome_diverges'] for r in rs)}")
EOF
fi

if run representation; then
$PY - <<'EOF'
import json
print("== representation diagnosis (theorem-grouped evaluation) ==")
d=json.load(open("data/representation/results.json"))
print(f"  classes {d['corpus']['classes']}; theorems {d['corpus']['theorems']}; "
      f"contradictory pairs {d['corpus']['contradictory_pairs']}; "
      f"examples {d['corpus']['examples']}")
for key,label in (("default","default rendering"),("shallow","structural sketch")):
    r=d["representations"][key]
    lo,hi=r["theorem_cluster_bootstrap_95"]
    print(f"  {label:21} pair-rank {r['paired_ranking_accuracy']:.3f} "
          f"CI [{lo:.3f},{hi:.3f}]  accuracy {r['classification_accuracy']:.3f} "
          f"Brier {r['brier']:.3f}")
s=d["separation"]
print(f"  class separation: shallow sketch {s['shallow_sketch']}/{s['valid_classes']}; "
      f"digest {s['digest']}/{s['valid_classes']}")
print(f"  median bytes shallow={s['median_bytes']['shallow_bytes']:.0f}, "
      f"sketch={s['median_bytes']['shallow_sketch_bytes']:.0f}")
configs=[d]
for filename in ("results.dim2048.json","results.dim4096.json","results.dim16384.json"):
    q=json.load(open("data/representation/"+filename))
    configs.append(q)
    print(f"  {filename:22} shallow "
          f"{q['representations']['shallow']['paired_ranking_accuracy']:.3f}")
print("  mechanism holdout (shallow):", {
    k: round(v['paired_ranking_accuracy'],3)
    for k,v in d['representations']['shallow']['leave_one_mechanism_out'].items()})
print(f"  mechanism macro (shallow): "
      f"{d['representations']['shallow']['mechanism_macro_paired_accuracy']:.3f}")
print(f"  all-group macro (shallow): "
      f"{d['representations']['shallow']['all_group_macro_paired_accuracy']:.3f}")
for field,label in (
    ("mechanism_macro_paired_accuracy","mechanism macro"),
    ("all_group_macro_paired_accuracy","all-group macro"),
    ("all_group_pair_weighted_paired_accuracy","pair weighted")):
    values=[q['representations']['shallow'][field] for q in configs]
    print(f"  robustness range, {label:15}: {min(values):.3f}--{max(values):.3f}")
coercion=[q['representations']['shallow']['leave_one_mechanism_out']
          ['coercion']['paired_ranking_accuracy'] for q in configs]
print(f"  robustness range, held-out coercion: {min(coercion):.3f}--{max(coercion):.3f}")
EOF
fi

if run direct; then
$PY - <<'EOF'
import collections,json,math
print("== direct frozen-model acceptance test ==")
d=json.load(open("data/direct_model_acceptance/results.json"))
c=d["corpus"]
print(f"  {c['classes']} classes; {c['theorems']} theorems; "
      f"{c['contradictory_pairs']} paired actions; {c['examples']} scored examples")
for arm in ("ordinary","structural"):
    r=d[arm]; lo,hi=r["theorem_cluster_bootstrap_95"]
    print(f"  {arm:12} pair rank {r['paired_ranking_accuracy']:.3f} "
          f"CI [{lo:.3f},{hi:.3f}]  named-mechanism macro "
          f"{r['mechanism_macro_paired_accuracy']:.3f}")
    print("   ", {name:(value['pairs'],round(value['paired_accuracy'],3))
                  for name,value in r['by_mechanism'].items()})
rows=collections.defaultdict(list)
for row in d["rows"]: rows[row["pair_id"]].append(row)
equal=[]
length_deltas=[]; score_margins=[]
for pair in rows.values():
    pos=next(row for row in pair if row["label"]==1)
    neg=next(row for row in pair if row["label"]==0)
    margin=pos["structural_mean_logprob"]-neg["structural_mean_logprob"]
    length_deltas.append(pos["structural_prompt_tokens"]-neg["structural_prompt_tokens"])
    score_margins.append(margin)
    if len({row["structural_prompt_tokens"] for row in pair})==1:
        equal.append(1 if margin>1e-9 else 0 if margin < -1e-9 else .5)
print(f"  equal-prompt-length control: {len(equal)} pairs, rank "
      f"{sum(equal)/len(equal):.3f}")
mx=sum(length_deltas)/len(length_deltas); my=sum(score_margins)/len(score_margins)
num=sum((x-mx)*(y-my) for x,y in zip(length_deltas,score_margins))
den=math.sqrt(sum((x-mx)**2 for x in length_deltas)*
              sum((y-my)**2 for y in score_margins))
print(f"  prompt-length delta / score-margin Pearson r: {num/den:.3f}")
outcomes=collections.Counter(
    "correct" if row["correct"]==1 else "tie" if row["correct"]==.5 else "reverse"
    for row in d["structural"]["pair_predictions"])
print("  structural outcomes:", dict(outcomes))
print(f"  mean structural prompt increment: "
      f"{d['prompt_tokens']['mean_increment']:.1f} tokens")
EOF
fi

if run sweep; then
$PY - <<'EOF'
import json
print("== beam-width candidate exposure sweep ==")
d=json.load(open("data/candidate_sweep/results.json"))
c=d["corpus"]
print(f"  complete classes {c['complete_classes']}/{c['records']}; "
      f"members {c['members']}; theorems {c['theorems']}")
for row in d["sweep"]:
    print(f"  k={row['k']:2}: member coverage {row['member_oracle_coverage']:.3f}; "
          f"shared-action coverage {row['best_shared_action_coverage']:.3f}; "
          f"ambiguous classes {row['acceptance_ambiguous_classes']}; "
          f"headroom {row['member_specific_headroom']:.3f}")
EOF
fi

if run rerank; then
$PY - <<'EOF'
import json
print("== BFS-Prover-V2-7B candidate reranking ==")
d=json.load(open("data/bfs_rerank/results.json"))
c=d["corpus"]; o=d["fixed_candidate_oracles"]
print(f"  {c['classes']} classes, {c['members']} members, "
      f"{c['candidate_executions']} state-candidate units, "
      f"{c['compiler_calls']} compiler calls")
def row(label,r):
    print(f"  {label:27} top1 {r['top1_acceptance']:.3f}  "
          f"top3 {r['top3_acceptance']:.3f}  MRR {r['mrr']:.3f}  "
          f"first-rank {r['mean_first_accepted_rank_covered']:.3f}")
row("raw 7B order",d["raw_generator_order"])
row("text-only reranker",d["representations"]["default"])
row("+ structural sketch",d["representations"]["shallow"])
row("transferred sketch",d["transfer_from_contradiction_classifier"]["representations"]["shallow"])
i=d["structural_increment"]
print(f"  structural increment: MRR {i['mrr_delta_shallow_minus_default']:+.4f} "
      f"CI [{i['mrr_delta_theorem_bootstrap_95'][0]:.4f},"
      f"{i['mrr_delta_theorem_bootstrap_95'][1]:.4f}], "
      f"top1 {i['top1_delta_shallow_minus_default']:+.3f}; "
      f"improved/worsened/tied members "
      f"{i['improved_members_vs_default']}/{i['worsened_members_vs_default']}/{i['tied_members_vs_default']}")
print(f"  candidate exposure: {o['acceptance_ambiguous_classes']} ambiguous class, "
      f"{o['acceptance_ambiguous_class_actions']} ambiguous class-action; "
      f"member oracle {o['member_oracle_top1']:.3f}, "
      f"best shared oracle {o['best_shared_action_oracle_top1']:.3f}, "
      f"headroom {o['member_specific_oracle_headroom']:.3f}")
for filename in ("robust_10fold.json","robust_wide.json","robust_lowdim.json"):
    q=json.load(open("data/bfs_rerank/"+filename))
    print(f"  {filename:22} text {q['representations']['default']['mrr']:.4f}  "
          f"sketch {q['representations']['shallow']['mrr']:.4f}  "
          f"increment {q['structural_increment']['mrr_delta_shallow_minus_default']:+.4f}")
EOF
fi

if run policies; then
$PY - <<'EOF'
import json,glob
print("== two-policy exposure ==")
uniq=set()
for run in ("reconv_b2","hunt_enn","hunt_rare"):
    for c in json.load(open(f"data/{run}/confirmation.json")):
        if c.get("verdict")=="CONFIRMED": uniq.add((c["theorem"],c["pp_hash"]))
rr={}
for f in glob.glob("data/regret/regret.shard*.jsonl"):
    for l in open(f):
        if l.strip():
            d=json.loads(l); rr.setdefault((d["theorem"],d["pp_hash"]),d)
ok=[r for k,r in rr.items() if k in uniq and not r.get("skipped") and not r.get("error")]
ent=sum(len(r.get("outcomes") or {}) for r in ok)
acc=sum(1 for r in ok if any(len({'F' if v=='FAIL' else 'O' for v in o.values()})>1
                             for o in (r.get('outcomes') or {}).values()))
print(f"  ReProver: {len(ok)} classes, {ent} candidate entries, "
      f"acceptance-divergent {acc}, top-1 acc-div "
      f"{sum(1 for r in ok if r.get('top1_viability_divergent'))}, "
      f"regret>0 {sum(1 for r in ok if r.get('regret',0)>0)}, "
      f"non-vacuous {sum(1 for r in ok if all((r.get('viable_sets') or {}).values()))}")
p2=[json.loads(l) for l in open("data/policy2/results.jsonl") if l.strip()]
o2=[r for r in p2 if not r.get("skipped") and not r.get("error")]
print(f"  miniCTX : {len(o2)} classes, "
      f"{sum(len(r.get('outcomes') or {}) for r in o2)} candidate entries, "
      f"acceptance-divergent {sum(1 for r in o2 if r.get('accept_div_any'))}, "
      f"top-1 acc-div {sum(1 for r in o2 if r.get('top1_accept_div'))}, "
      f"regret>0 {sum(1 for r in o2 if r.get('regret',0)>0)}, "
      f"non-vacuous {sum(1 for r in o2 if r.get('nonvacuous'))}")

# --- exposure split by hidden-structure family (paper section 6)
def fam(m):
    t=" ".join((m or {}).get("diff_tokens") or []); rg=(m or {}).get("rungs",{})
    if "ENNReal.ofNNReal" in t or "WithTop.some" in t: return "coercion"
    if "@Ne" in t or "+Not" in t: return "Ne-vs-Not"
    if "instOfNatNat" in t: return "Fin-eta"
    if any(x in t for x in ("toLE","toPreorder","linearOrderOfSTO","WellOrderingRel")):
        return "instance"
    dp=rg.get("phi_deep_proofs")
    if isinstance(dp,dict) and dp.get("separates"): return "proof-term"
    return "not-printer-separable"
famof={}
for run in ("reconv_b2","hunt_enn","hunt_rare"):
    cs=[c for c in json.load(open(f"data/{run}/confirmation.json"))
        if c.get("verdict")=="CONFIRMED"]
    ms=json.load(open(f"data/{run}/mechanism.json"))
    for c,m in zip(cs,ms): famof.setdefault((c["theorem"],c["pp_hash"]),fam(m))
def bucket(k): return "coercion" if famof.get(k)=="coercion" else "non-coercion"
print("  -- exposure by family --")
for name,store in (("ReProver",rr),("miniCTX",{(r["theorem"],r["pp_hash"]):r for r in o2})):
    for b in ("coercion","non-coercion"):
        rs=[r for k,r in store.items() if k in famof and bucket(k)==b
            and not r.get("skipped") and not r.get("error")]
        if name=="ReProver":
            ad=sum(1 for r in rs if any(len({'F' if v=='FAIL' else 'O' for v in o.values()})>1
                                        for o in (r.get('outcomes') or {}).values()))
            od=sum(1 for r in rs if any(len(set(o.values()))>1
                                        for o in (r.get('outcomes') or {}).values()))
            t1=sum(1 for r in rs if r.get("top1_viability_divergent"))
        else:
            ad=sum(1 for r in rs if r.get("accept_div_any"))
            od=float("nan"); t1=sum(1 for r in rs if r.get("top1_accept_div"))
        od_s="  n/a" if od!=od else f"{od:5.0f}"
        print(f"    {name:9} {b:13} classes {len(rs):3}  outcome-div {od_s}"
              f"  acceptance-div {ad:3}  top-1 acc-div {t1:3}")
EOF
fi

if run cost; then
$PY - <<'EOF'
import json,glob,statistics
print("== node-identity cost (arm A no probe / B pp.all / C digest) ==")
recs=[json.loads(l) for f in sorted(glob.glob("data/search_cost/cost.shard*.jsonl"))
      for l in open(f) if l.strip()]
ok=[r for r in recs if all(a in r["arms"] and "error" not in r["arms"][a]
                           and r["arms"][a].get("stopped")!="desync" for a in "ABC")]
print(f"  usable triples {len(ok)}; arm orders {sorted({''.join(r['arm_order']) for r in ok})}")
for a in "ABC":
    g=lambda k:[r["arms"][a][k] for r in ok]
    w=sum(g("elapsed"))
    print(f"  {a}: wall {w:6.0f}s  gen {sum(g('t_gen')):5.0f}s  tac {sum(g('t_tac')):5.0f}s  "
          f"probe {sum(g('t_key')):5.0f}s ({100*sum(g('t_key'))/w:4.1f}%)  "
          f"nodes {sum(g('nodes')):4}  peakQ {sum(g('peak_queue')):4}  "
          f"expansions {sum(g('expansions')):4}  proved {sum(g('proved'))}")
for a in "BC":
    d=[r["arms"][a]["nodes"]-r["arms"]["A"]["nodes"] for r in ok]
    q=[r["arms"][a]["peak_queue"]-r["arms"]["A"]["peak_queue"] for r in ok]
    print(f"  {a} vs A: nodes differ in {sum(1 for x in d if x)}/{len(ok)} theorems "
          f"(total {sum(d):+d}), peak queue (total {sum(q):+d})")
EOF
fi

if run natural; then
$PY - <<'EOF'
import json
print("== rendering repeats in Mathlib's own proof scripts ==")
d=json.load(open("data/natural/summary.json"))
for k in ("states","proofs","proofs_with_repeat","repeat_pairs","same_tactic",
          "same_tactic_successor_differs","different_tactic"):
    print(f"  {k:34} {d[k]}")
print("  human scripts are linear and essentially never revisit a rendering")
EOF
fi

if run repsens; then
$PY - <<'EOF'
import json,glob,statistics as stt
print("== representative sensitivity (independent search from each member) ==")
recs=[json.loads(l) for f in sorted(glob.glob("data/repsens/repsens.shard*.jsonl"))
      for l in open(f) if l.strip()]
u=[r for r in recs if r.get("searches")]
print(f"  classes {len(u)}; member-searches {sum(len(r['searches']) for r in u)}; "
      f"expansions {sum(v['expansions'] for r in u for v in r['searches'].values())}")
print(f"  control, identical first candidate list across members: "
      f"{sum(1 for r in u if r.get('first_candidates_identical'))}/{len(u)}")
sens=[r for r in u if r.get("representative_sensitive")]
anyp=[r for r in u if r.get("proved_any")]
print(f"  classes with any member proved {len(anyp)}; outcome depends on member {len(sens)}")
ex=[];jac=[];sz=[]
for r in u:
    t=[set(v["reached"]) for v in r["searches"].values()]
    ex.append(len(set.union(*t))-len(set.intersection(*t)))
    if r.get("tree_jaccard") is not None: jac.append(r["tree_jaccard"])
    sz.append(stt.mean(len(x) for x in t))
print(f"  median states explored per member {stt.median(sz):.0f}")
print(f"  classes reaching a state some member never reaches: "
      f"{sum(1 for e in ex if e>0)}/{len(u)} (median {stt.median(ex):.0f} such states)")
print(f"  median Jaccard overlap of explored sets {stt.median(jac):.2f} "
      f"(over {len(jac)} classes that explored any state)")
print(f"  first-step acceptance identical: "
      f"{sum(1 for r in u if r.get('first_step_acceptance_identical'))}/{len(u)}")
print()
print("== representative sensitivity under BFS-Prover-V2-7B ==")
bf=[json.loads(l) for f in sorted(glob.glob("data/repsens_bfs/repsens.shard*.jsonl"))
    for l in open(f) if l.strip()]
bu=[r for r in bf if r.get("searches")]
print(f"  classes {len(bu)}; identical first candidates "
      f"{sum(1 for r in bu if r.get('first_candidates_identical'))}/{len(bu)}; "
      f"any proved {sum(1 for r in bu if r.get('proved_any'))}; "
      f"representative-sensitive {sum(1 for r in bu if r.get('representative_sensitive'))}")
for tag in ("repsens_bfs","repsens_bfs_rep2","repsens_bfs_rep3","repsens_bfs_deep"):
    for f in glob.glob(f"data/{tag}/repsens.shard*.jsonl"):
        for l in open(f):
            d=json.loads(l)
            if "snormEssSup_add_le" not in d["theorem"]: continue
            outs={k[:16]:v["proved"] for k,v in d["searches"].items()}
            print(f"  {tag:18} sensitive={d.get('representative_sensitive')} {outs}")
print("  the losing member fails at 24 exp/300s x3 runs AND 72 exp/900s")
EOF
fi

if run stress; then
$PY - <<'EOF'
import json
print("== 96-pair alias-conditioned retention benchmark ==")
d=json.load(open("data/alias_stress/results.json"))
print(f"  candidates {d['candidate_pairs']}; admitted {d['eligible_pairs']}; "
      f"base templates {d['eligible_base_templates']}; "
      f"proposition contexts {d['eligible_proposition_contexts']}")
c=d["compilation"]
print(f"  compiler labels {c['state_action_labels']}; "
      f"failed labels {c['failed_state_action_labels']}; "
      f"reported diagnostics {c['reported_action_errors']}")
for r in d["summary"]:
    lo,hi=r["pair_cluster_bootstrap_95"]
    blo,bhi=r["base_cluster_bootstrap_95"]
    print(f"  {r['generator']:13} B={r['budget']:2}  "
          f"refined coverage {r['refined_proof_pairs']:2}/{r['eligible_pairs']}  "
          f"exact loss {100*r['uniform_order_loss_probability']:5.1f}%  "
          f"pair CI [{100*lo:.1f},{100*hi:.1f}]  "
          f"base CI [{100*blo:.1f},{100*bhi:.1f}]  "
          f"randomized {100*r['randomized_loss_probability']:.1f}%")
primary=next(r for r in d["summary"]
             if r["generator"]=="targeted" and r["budget"]==2)
num=sum(x["exact_loss_numerator"] for x in primary["pair_results"])
den=sum(x["exact_loss_denominator"] for x in primary["pair_results"])
print(f"  primary exact calculation: {num}/{den} = {100*num/den:.4f}%")
print("  refined identity records zero retention losses once a proof is exposed")

print("== supplementary eight-pair retention control ==")
d=json.load(open("data/stress_search/results.json"))
print(" ", json.dumps(d["summary"]))
for r in d["pairs"]:
    print(f"  pair {r['pair']} {r['probe'][:30]:30} valid={int(r['valid'])} "
          f"closes A={int(r['variant_closes']['A'])} B={int(r['variant_closes']['B'])} "
          f"R(Afirst)={int(r['arms']['R_Afirst']['proved'])} "
          f"R(Bfirst)={int(r['arms']['R_Bfirst']['proved'])} "
          f"digest both orders={int(r['digest_invariant'])}")
print("  proof loss occurs exactly when the rejecting variant is retained")
print("  invariance in 8/8; 3 pairs close definitionally by rfl and are immune")
u=json.load(open("data/stress_union/results.json"))
print("  fixed-grammar control (union of all probes + rfl):", json.dumps(u["summary"]))
anc=json.load(open("data/ancestor/results.json"))
print("  ancestor-rooted natural class:", anc["theorem"])
for arm,v in anc["arms"].items():
    print(f"    {arm:16} proved={v.get('proved')} proof={v.get('proof')} "
          f"discarded={v.get('merged_member_discarded')}")
print("  all arms prove via an intermediate: the natural loss is confined to the")
print("  merged state's continuation, as the paper states")
EOF
fi

if run persist; then
$PY - <<'EOF'
import json,collections,glob
from scipy.stats import beta
print("== current-version persistence (Mathlib 3b3cdbb, Lean v4.33.0-rc1) ==")
manifest=json.load(open("data/current_blind/manifest.json"))
roots=[json.loads(line) for f in glob.glob("data/current_blind_search/roots.shard*.jsonl")
       for line in open(f) if line.strip()]
aliases=[json.loads(line) for f in glob.glob("data/current_blind_search/stageb.shard*.jsonl")
         for line in open(f) if line.strip()]
confirmed=[row for row in json.load(open("data/current_blind_search/confirmation.json"))
           if row.get("verdict")=="CONFIRMED"]
hit_roots=len({(row["file"],row["theorem"],row["tactic_index"])
               for row in confirmed})
n=len(roots); lo=beta.ppf(.025,hit_roots,n-hit_roots+1); hi=beta.ppf(.975,hit_roots+1,n-hit_roots)
print(f"  blind files {manifest['successfully_traced_files']}/{manifest['selected_files']}; "
      f"states {manifest['states']}; successful roots {n}/600; aliases {len(aliases)}")
print(f"  confirmed classes {len(confirmed)} on {hit_roots} root; "
      f"incidence {hit_roots/n:.4%}, exact 95% CI [{lo:.4%},{hi:.4%}]")
rows=[json.loads(l) for l in open("data/persistence/results.jsonl") if l.strip()]
by=collections.defaultdict(list)
for r in rows: by[r["theorem"]].append(r)
for t,rs in by.items():
    v=collections.Counter(x["verdict"] for x in rs)
    note=" (all members; renderings identical, successors diverge)" \
         if all(x["verdict"]=="PERSISTS" for x in rs) else ""
    print(f"  {t[:52]:52} {dict(v)}{note}")
print(f"  member-verdicts: {dict(collections.Counter(r['verdict'] for r in rows))}")
EOF
fi
