# -*- coding: utf-8 -*-
# @Time    : 2023/7/31 09:46
# @Author  : Ray
# @Email   : httdty2@163.com
# @File    : evaluation.py
# @Software: PyCharm
import argparse
import json

from .spider_evaluator import EvaluateTool


def main(gold, pred, db_dir, ts_db):
    with open(gold, 'r') as f:
        gold = json.load(f)
    with open(pred, 'r') as f:
        preds = [p.strip() for p in f.readlines()]

    if len(preds[-1]) == 0:
        preds.pop(-1)
    assert len(preds) == len(gold)

    eval_tool = EvaluateTool(verbose=True)
    eval_tool.register_golds(gold, db_dir, ts_db)
    eval_tool.evaluate(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str)
    parser.add_argument("--pred", type=str)
    parser.add_argument("--db", type=str)
    parser.add_argument("--ts_db", default="", type=str)
    args = parser.parse_args()

    main(args.gold, args.pred, args.db, args.ts_db)
