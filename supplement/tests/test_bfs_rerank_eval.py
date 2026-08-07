from tools.bfs_rerank_eval import candidate_oracles, ranking_metrics, stable_label


def test_stable_label_excludes_inconsistent_repeats():
    assert stable_label(["STATE", "STATE", "STATE"]) == 1
    assert stable_label(["COMPLETE", "COMPLETE", "COMPLETE"]) == 1
    assert stable_label(["FAIL", "FAIL", "FAIL"]) == 0
    assert stable_label(["FAIL", "STATE", "FAIL"]) is None
    assert stable_label([]) is None


def test_ranking_metrics_measures_functional_reranking():
    examples = [
        {"action": "bad", "raw_rank": 1, "label": 0},
        {"action": "good", "raw_rank": 2, "label": 1},
        {"action": "also_bad", "raw_rank": 3, "label": 0},
    ]
    member = {"member_id": "m", "theorem": "T", "pp_hash": "p",
              "family": "f", "indices": [0, 1, 2]}
    raw, _ = ranking_metrics(examples, [member])
    reranked, rows = ranking_metrics(examples, [member], scores=[0.1, 0.9, 0.2])
    assert raw["top1_acceptance"] == 0.0
    assert raw["mrr"] == 0.5
    assert reranked["top1_acceptance"] == 1.0
    assert reranked["mrr"] == 1.0
    assert rows[0]["order"][0] == "good"


def test_candidate_oracle_detects_member_specific_headroom():
    examples = [
        {"action": "a", "label": 1}, {"action": "b", "label": 0},
        {"action": "a", "label": 0}, {"action": "b", "label": 1},
    ]
    members = [
        {"theorem": "T", "pp_hash": "p", "indices": [0, 1]},
        {"theorem": "T", "pp_hash": "p", "indices": [2, 3]},
    ]
    result = candidate_oracles(examples, members)
    assert result["member_oracle_top1"] == 1.0
    assert result["best_shared_action_oracle_top1"] == 0.5
    assert result["member_specific_oracle_extra_successes"] == 1
    assert result["acceptance_ambiguous_classes"] == 1
    assert result["acceptance_ambiguous_class_actions"] == 2
