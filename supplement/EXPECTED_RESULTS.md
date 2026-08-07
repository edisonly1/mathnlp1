# Expected results and record map

This file maps the principal paper claims to the records and reproduction
commands in the supplement. Values are recomputed by `reproduce.sh`; they are
included here so reviewers can identify accidental environment or archive
problems quickly.

## Main paper

| Paper location | Command | Expected result | Primary records |
|---|---|---|---|
| Table 1 | `./reproduce.sh witness` | Identical 206-byte rendering; `simp` fails after `intro h; simp` and succeeds after `simp; intro h` | `data/reconv_b2/stageb.shard*.jsonl` |
| Table 2 | `./reproduce.sh regimes` | 70 unique confirmed classes after deduplication; 68 acceptance-ambiguous classes; 103 contradictory pairs | `data/reconv_b2`, `data/hunt_enn`, `data/hunt_rare`, `data/hunt_rare2` |
| Tables 3 and 4 | `./reproduce.sh representation` | Theorem-grouped pair rank .976, 95% CI [.943, 1.000]; mechanism macro .836; all-group macro .769 | `data/representation/results*.json` |
| Table 5 | `./reproduce.sh direct` | Frozen 7B structural pair rank .539, 95% CI [.440, .644]; audited-mechanism macro .504 | `data/direct_model_acceptance/results.json` |
| Table 6 | `./reproduce.sh sweep` | Member coverage .376, .798, .882, .938, .938 at k=1,2,4,8,16; zero member-specific headroom throughout | `data/candidate_sweep/results.json`, `executions.shard*.jsonl` |
| Table 7 | `./reproduce.sh stress` | 96 admitted pairs; targeted refined coverage 96/96; rendering loss 93/192, or 48.4%, under exact uniform arrival order | `data/alias_stress/results.json` |

## Additional central checks

| Claim | Command | Expected result | Primary records |
|---|---|---|---|
| Current-stack blind incidence | `./reproduce.sh persist` | 60/60 files traced; 2,507 states; 570/600 successful roots; 98 aliases; two confirmed classes on one root; incidence .1754%, exact 95% CI [.0044%, .9736%] | `data/current_blind/manifest.json`, `data/current_blind_search` |
| Structural separation | `./reproduce.sh representation` | Shallow sketch separates 67/70 classes; digest separates 70/70 | `data/representation` |
| Frozen-model decomposition | `./reproduce.sh direct` | Ordinary rendering ties every pair at .500; structural input gives 53 correct rankings, five ties, and 45 reversals | `data/direct_model_acceptance/results.json` |
| Candidate repeat control | `./reproduce.sh sweep` | 70 complete classes, 178 members, zero unstable state-action labels | `data/candidate_sweep` |
| Natural member sensitivity | `./reproduce.sh repsens` | BFS-Prover-V2 has one representative-sensitive class; the result repeats three times and at triple budget | `data/repsens_bfs*` |
| Broad paired-search null | `./reproduce.sh all` | Same theorem result in all 158 paired searches; four searches contain a fingerprint-distinct merge | `data/search_ab`, `data/search_abc`, `data/search_probe2` |
| Identity cost | `./reproduce.sh cost` | Relative total time 1.00 for rendering, 2.79 for shallow `pp.all`, and 1.08 for the structural digest | `data/search_cost` |
| Current package identity | inspect saved audit | LeanDojo 2.0.2 equality and hashing use the rendering; active LeanDojo-v2/Pantograph preserves distinct state handles while model prompts remain textual | `data/current_identity/results.json` |

## Corpus composition

The 70-class deduplicated corpus contains 56 coercion classes, four `Fin` eta
classes, four local-definition unfolding classes, two instance-path classes,
two residual classes, one `Ne` versus `Not (Eq ...)` class, and one proof-term
class. The acceptance evaluation contains 68 classes, 51 theorems, 103 paired
actions, and 206 balanced examples.

## Expected command status

`./reproduce.sh all` should exit with status 0. It reads only files under
`data/` and does not modify the supplement. The command does not require Lean,
network access, a model checkpoint, or a GPU.
