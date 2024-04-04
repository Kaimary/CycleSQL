# encoding=utf8
import json
import os
import time
from typing import List
import argparse

from .spider.preprocess.get_tables import dump_db_json_schema
from .spider_exact_match import compute_exact_match_metric
from .spider_test_suite import compute_test_suite_metric
from .spider import evaluation as spider_evaluation
from .test_suite import evaluation as test_suite_evaluation
# from eval.spider.evaluation import print_scores as spider_score
from .test_suite.evaluation import print_scores as test_suite_score


class EvaluateTool(object):
    def __init__(self, verbose=False, **kwargs):
        # self.args = args
        self.schema_cache = dict()
        self.golds: List[dict] = []
        self.verbose = verbose
        # self.spider_evaluator = None
        self.exec_evaluator = None
        self.test_suite_evaluator = None
        self.sample_db_path = kwargs.get("sample_db_path", "")
        self.sample_ts_res = []

    @staticmethod
    def _human_fix(sample):
        if 'query' not in sample:
            return
        if (
                sample['query']
                == 'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 '
                   'ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON '
                   'T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1'
        ):
            sample[
                'query'
            ] = 'SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON ' \
                'T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1'
            sample['query_toks'] = [
                'SELECT',
                'T1.company_type',
                'FROM',
                'Third_Party_Companies',
                'AS',
                'T1',
                'JOIN',
                'Maintenance_Contracts',
                'AS',
                'T2',
                'ON',
                'T1.company_id',
                '=',
                'T2.maintenance_contract_company_id',
                'ORDER',
                'BY',
                'T2.contract_end_date',
                'DESC',
                'LIMIT',
                '1',
            ]
            sample['query_toks_no_value'] = [
                'select',
                't1',
                '.',
                'company_type',
                'from',
                'third_party_companies',
                'as',
                't1',
                'join',
                'maintenance_contracts',
                'as',
                't2',
                'on',
                't1',
                '.',
                'company_id',
                '=',
                't2',
                '.',
                'maintenance_contract_company_id',
                'order',
                'by',
                't2',
                '.',
                'contract_end_date',
                'desc',
                'limit',
                'value',
            ]
            sample['question'] = 'What is the type of the company who concluded its contracts most recently?'
            sample['question_toks'] = [
                'What',
                'is',
                'the',
                'type',
                'of',
                'the',
                'company',
                'who',
                'concluded',
                'its',
                'contracts',
                'most',
                'recently',
                '?',
            ]
        if sample['query'].startswith(
                'SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN'
        ):
            sample['query'] = sample['query'].replace('IN (SELECT T2.dormid)', 'IN (SELECT T3.dormid)')
            index = sample['query_toks'].index('(') + 2
            assert sample['query_toks'][index] == 'T2.dormid'
            sample['query_toks'][index] = 'T3.dormid'
            index = sample['query_toks_no_value'].index('(') + 2
            assert sample['query_toks_no_value'][index] == 't2'
            sample['query_toks_no_value'][index] = 't3'

    def register_golds(self, dataset, db_path, sample_db_path: str = ""):
        for idx, sample in enumerate(dataset):
            self._human_fix(sample)

            db_id = sample["db_id"]
            if sample_db_path:
                self.sample_db_path = sample_db_path
            if db_id not in self.schema_cache:
                self.schema_cache[db_id] = dump_db_json_schema(
                    db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                )
            schema = self.schema_cache[db_id]

            self.golds.append(
                {
                    "query": sample["query"],
                    "question": sample["question"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "sample_db_path": self.sample_db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": {
                        "table_id": [table_id for table_id, _ in schema["column_names_original"]],
                        "column_name": [column_name for _, column_name in schema["column_names_original"]],
                    },
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": {
                        "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
                        "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]],
                    },
                }
            )

    def evaluate(self, preds):
        if self.verbose:
            print("################################################ Spider Evaluation "
                  "################################################")
        # exact_match = compute_exact_match_metric(preds, self.golds, verbose=self.verbose)
        # if self.verbose:
        #     print("\n\n")
        #     print("################################################  Exec Evaluation  "
        #           "################################################")
        exec_match = compute_test_suite_metric(preds, self.golds, db_dir=None, verbose=self.verbose)

        test_suite = {}
        if self.sample_db_path:
            if self.verbose:
                print("\n\n")
                print("############################################## Test Suite Evaluation "
                      "##############################################")
            start = time.time()
            test_suite = compute_test_suite_metric(
                preds, self.golds, db_dir=self.sample_db_path,
                verbose=self.verbose, progress_bar_for_each_datapoint=False
            )
            print(f"Test suite on sample dbs time cost: {time.time() - start}")


        return {**exec_match, **test_suite}

    def evaluate_one(self, idx, prediction):
        if not self.exec_evaluator:
            self._init_evaluator()
        reference = self.golds[idx]

        # exact_score = self.spider_evaluator.evaluate_one(
        #     reference["db_id"],
        #     reference["query"],
        #     prediction
        # )

        turn_scores = {"exec": [], "exact": []}
        turn_idx = reference.get("turn_idx", 0)
        exec_score = self.exec_evaluator.evaluate_one(
            reference["db_id"],
            reference["query"],
            prediction,
            turn_scores,
            idx=turn_idx,
        )

        turn_scores = {"exec": [], "exact": []}
        turn_idx = reference.get("turn_idx", 0)
        ts_score = {
            "exec": 0
        }
        if self.test_suite_evaluator:
            ts_score = self.test_suite_evaluator.evaluate_one(
                reference["db_id"],
                reference["query"],
                prediction,
                turn_scores,
                idx=turn_idx,
            )

        return {
            "exact_match": int(exec_score["exact"]),
            # "exec_match": int(s_score["exec"]),
            # "test_suite_match": int(ts_score["exec"]),
            "exec_match": int(exec_score["exec"]),
            "test_suite_match": int(ts_score["exec"]),
        }

    def _init_evaluator(self):
        foreign_key_maps = dict()
        for reference in self.golds:
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
        # self.spider_evaluator = spider_evaluation.Evaluator(self.golds[0]["db_path"], foreign_key_maps, "all")
        
        foreign_key_maps = dict()
        for reference in self.golds:
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

        self.exec_evaluator = test_suite_evaluation.Evaluator(
            db_dir=self.golds[0]["db_path"],
            kmaps=foreign_key_maps,
            etype="all",
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )


        if self.sample_db_path:
            self.test_suite_evaluator = test_suite_evaluation.Evaluator(
                db_dir=self.sample_db_path,
                kmaps=foreign_key_maps,
                etype="all",
                plug_value=False,
                keep_distinct=False,
                progress_bar_for_each_datapoint=False,
            )

    def print_score(self, include_turn_acc=False):
        print("################################################ Spider Evaluation "
              "################################################")
        # self.spider_evaluator.finalize()
        # spider_score(self.spider_evaluator.scores, self.spider_evaluator.etype)
        # print("\n\n")
        #
        # print("################################################  Exec Evaluation  "
        #       "################################################")
        self.exec_evaluator.finalize()
        test_suite_score(
            self.exec_evaluator.scores,
            self.exec_evaluator.etype,
            include_turn_acc=include_turn_acc
        )
        print("\n\n")

        if self.test_suite_evaluator:
            print("############################################## Test Suite Evaluation "
                  "##############################################")
            self.test_suite_evaluator.finalize()
            test_suite_score(
                self.test_suite_evaluator.scores,
                self.test_suite_evaluator.etype,
                include_turn_acc=include_turn_acc
            )
            print("\n\n")


    #     if self.sample_ts_res:
    #         self.print_sample_ts_score()
    #         print("\n\n")
    #
    # def print_sample_ts_score(self):
    #     def print_formated_s(row_name, l_, element_format):
    #         template = "{:20} " + " ".join([element_format] * len(l_))
    #         print(template.format(row_name, *l_))
    #
    #     levels = ["easy", "medium", "hard", "extra", "all"]
    #     scores = {
    #         "count": {},
    #         'pass': {}
    #     }
    #     for l in levels:
    #         scores['count'][l] = 0
    #         scores['pass'][l] = 0
    #     for hardness, res in self.sample_ts_res:
    #         scores['count'][hardness] += 1
    #         scores['pass'][hardness] += int(res)
    #
    #     print("############################### Sample DB Test Suite Evaluation ###############################")
    #     print("=====================   Sample DB Test Suite ACCURACY     =====================")
    #     print_formated_s("", levels, "{:20}")
    #     real_ts_scores = [scores['pass'][l] / scores['count'][l] for l in levels]
    #     real_ts_scores.append(sum([scores['pass'][l] for l in levels]) / len(self.golds))
    #     print_formated_s("Sample DB Test Suite", real_ts_scores, "{:<20.3f}")
