import os
import random
import time
import argparse

from tqdm import tqdm

from eval.distilled_test_suite_eval.fuzz.fuzz import generate_random_db_with_queries_wrapper


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_gold_file", type=str, required=True)
    parser.add_argument("--original_db_dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=1000)
    args_ = parser.parse_args()
    return args_


def create_dbs(dev_gold_file: str, db_dir: str, num: int = 1000):
    gold_data = open(dev_gold_file, "r").readlines()

    db_gold = {}
    for gold in gold_data:
        gold_query, db_id = gold.split("\t")[0].strip(), gold.split("\t")[-1].strip()
        db_gold.setdefault(db_id, [])
        db_gold[db_id].append(gold_query)

    for db_id, golds in db_gold.items():
        original_database_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
        random.shuffle(golds)
        for t in tqdm(range(num), desc=f"{db_id}"):
            sampled_database_w_path = os.path.join(db_dir, db_id, f"{db_id}_{t}.sqlite")
            generate_random_db_with_queries_wrapper((original_database_path, sampled_database_w_path, golds, {}))


if __name__ == '__main__':
    start = time.time()
    args = get_args()
    create_dbs(args.dev_gold_file, args.original_db_dir, args.num)
    print(f"Test-suite DB gen time cost: {time.time() - start}")
