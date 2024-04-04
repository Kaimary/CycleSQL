"""Spider Exact Match metric."""
from typing import Dict, Any
from .spider import evaluation as spider_evaluation
from .spider.evaluation import print_scores


def compute_exact_match_metric(predictions, references, verbose=False) -> Dict[str, Any]:
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = spider_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    etype = "all"
    evaluator = spider_evaluation.Evaluator(references[0]["db_path"], foreign_key_maps, etype)
    exact_scores = []
    exec_scores = []
    hardness = []

    i = 0
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        score = evaluator.evaluate_one(reference["db_id"], reference["query"], prediction)
        exact_scores.append(int(score["exact"]))
        exec_scores.append(int(score["exec"]))
        hardness.append(score["hardness"])
        i += 1
    evaluator.finalize()
    if verbose:
        print_scores(evaluator.scores, etype)
    return {
        "exact_match": evaluator.scores["all"]["exact"],
        "exact_match_scores": exact_scores,
        "exec_match": evaluator.scores["all"]["exec"],
        "exec_match_scores": exec_scores,
        "hardness": hardness,
        "raw_spider_match": evaluator.scores
    }
