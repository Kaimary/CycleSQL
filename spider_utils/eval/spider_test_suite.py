"""Spider Test Suite Execution Accuracy metric."""
import logging
from typing import Optional, Dict, Any

from tqdm import tqdm

from .test_suite import evaluation as test_suite_evaluation
from .test_suite.evaluation import print_scores

logger = logging.getLogger(__name__)


def compute_test_suite_metric(
        predictions, references, db_dir: Optional[str] = None, verbose=False,
        debug=False, progress_bar_for_each_datapoint=False
) -> Dict[str, Any]:
    if db_dir is None:
        db_dir = references[0]["db_path"]
    foreign_key_maps = dict()
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = test_suite_evaluation.build_foreign_key_map(
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
    evaluator = test_suite_evaluation.Evaluator(
        db_dir=db_dir,
        kmaps=foreign_key_maps,
        etype=etype,
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=progress_bar_for_each_datapoint,
    )
    # Only used for Sparc/CoSQL
    turn_scores = {"exec": [], "exact": []}
    scores = []
    for prediction, reference in tqdm(zip(predictions, references)):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        try:
            score = evaluator.evaluate_one(
                reference["db_id"],
                reference["query"],
                prediction,
                turn_scores,
                idx=turn_idx,
            )
            scores.append(int(score['exec']))
        except AssertionError as e:
            logger.warning(f"unexpected evaluation error: {e.args[0]}")
            scores.append(0)
    evaluator.finalize()
    if verbose:
        print_scores(evaluator.scores, etype, include_turn_acc=False)
    return {
        "exec": evaluator.scores["all"]["exec"],
        "exec_scores": scores,
        "exec_raw": evaluator.scores
    }
