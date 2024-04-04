import os, json, argparse, sqlite3
from os import listdir
from tqdm import tqdm
from copy import deepcopy
from os.path import isfile, join
from typing import List, Union
from prettytable import PrettyTable

from src.explainer import Explainer
from spider_utils.utils import read_single_dataset_schema_from_database
from spider_utils.process_sql import Schema, get_schema, get_sql
from src.utils import EXCLUDE_KEYWORDS, clause2dialect, execute_sqlite_query, find_relationship_tables, \
    get_db_statistics, get_one_query_result, parse_query_semantics, remove_join_conditions, rewrite_asterisk_query, sql_format
from spider_utils.evaluation import build_foreign_key_map_from_json, eval_exec_match1

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_file_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--table_file_path', type=str, required=True, help='processed table path')
    arg_parser.add_argument('--db_dir', type=str, default='data/database')
    arg_parser.add_argument('--beam_output_dir', type=str, required=True, help='raw beam outputs')
    arg_parser.add_argument('--output_path', type=str, required=True, help='output preprocessed dataset')
    arg_parser.add_argument('--overwrite', action='store_true', help='whether overwrite the exisiting outputs')
    args = arg_parser.parse_args()

    # Initialize some variables
    # evaluator = Evaluator()
    kmaps = build_foreign_key_map_from_json(args.table_file_path)
    explainer = Explainer(args.db_dir)

    id2schema = {ex['db_id']: ex for ex in json.load(open(args.table_file_path))}
    train = json.load(open(args.train_file_path))
    output_file = os.path.join(args.output_path, 'train.beam.all.exec.json')
    if os.path.exists(output_file) and args.overwrite: os.remove(output_file)
    writer = open(output_file, 'a')

    raw_beam_output_files = [f for f in listdir(args.beam_output_dir) if isfile(join(args.beam_output_dir, f))]
    print(f'There are {len(raw_beam_output_files)} beam output files detected:')
    for b in raw_beam_output_files: print(f'{b}')

    beam_outputs_list: List[list] = []
    beam_size_list: List[int] = []
    for beam_output_file in raw_beam_output_files:
        file_name = os.path.splitext(os.path.basename(beam_output_file))[0]
        model_name = file_name.split('.')[0]
        beam_size = int(file_name[-1])
        # Extract beam predictions from file
        beam_preds = [l.strip(';\n ') for l in open(os.path.join(args.beam_output_dir, beam_output_file)).readlines() if l.strip()]
        if 'resdsql' in model_name:
            if beam_preds[0][0] != 's': # exclude natural language questions for dev beam outputs
                beam_preds = [l for idx, l in enumerate(beam_preds) if idx % 9 != 0]
            beam_preds = [l.split('|')[1].strip() if '|' in l else l for l in beam_preds]
        elif model_name == 'smbop':
            beam_preds = [l.split('\t')[0].strip() for l in beam_preds]
        beam_outputs_list.append(beam_preds)
        beam_size_list.append(beam_size)

    prev_db_id = None
    for idx, ex in tqdm(enumerate(train), total=len(train)):
        # if idx < 103: continue
        sqls = []
        db_id = ex['db_id']
        if db_id != prev_db_id:
            db_file = os.path.join(args.db_dir, db_id, f'{db_id}.sqlite')
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors = 'ignore')
            cursor = conn.cursor()
            schema = read_single_dataset_schema_from_database(id2schema[db_id], cursor)
            schema2ids = Schema(get_schema(db_file))
            relation2tables = find_relationship_tables(schema)
            prev_db_id = db_id
        nl = ex['question']
        gold = ex['query']
        data = {
            'db_id': db_id,
            'question': nl
        }
        try:
            g_sql, _ = get_sql(schema2ids, gold)
            data['query'] = sql_format(gold)
            rewrite_sql = rewrite_asterisk_query(gold, schema, db_file)
            g_rewrite_sql, table2alias = get_sql(schema2ids, rewrite_sql)
            result_set: Union[list, list, list] = execute_sqlite_query(rewrite_sql, g_rewrite_sql, db_file)  # rows, cols, types
            if not result_set[1]: continue # skip if sql is invalid
            # To have `accurate` join semantics, we calculate the statistics of the newly (joined) table
            # and determine the semantics of `asterisk` symbol in SQL from statistics
            table2statistics = get_db_statistics(gold, g_sql, table2alias, schema, cursor)
            join_semantics, group_semantics = parse_query_semantics(g_sql, schema, relation2tables, table2statistics)
                
            # Overall explanation of the query execution with following aspects:
            # - row and column counts the query returned
            # - any filtering criteria (WHERE clause)
            # - any sorting criteria (ORDER LIMIT clause)
            filtering_modifier: str = clause2dialect(g_sql, schema, group_semantics, cls_type='where', prefix='filtered by ', bracket=True)
            filtering_modifier += clause2dialect(g_sql, schema, group_semantics, cls_type='having', prefix=' and ', bracket=True) if filtering_modifier else \
                clause2dialect(g_sql, schema, group_semantics, cls_type='having', prefix='filtered by ', bracket=True)
            sorting_modifier: str = clause2dialect(g_sql, schema, group_semantics, cls_type='orderBy', prefix='sorted ', bracket=True)
            exp = f'The query returns a result set with {len(result_set[1])} column{"s" if len(result_set[1]) > 1 else ""} ' + \
                f'[{", ".join([f"{c}" if any(kw in "".join(c.split()).lower() for kw in EXCLUDE_KEYWORDS) or t== "NoneType" else f"{c} ({t})" for c, t in zip(result_set[1], result_set[2])])}] and ' + \
                f'{len([r for r in result_set[0] if any(c for c in r)])} row{"s" if len(result_set[0]) > 1 else ""}{", " if filtering_modifier else ""}' + \
                f'{filtering_modifier if filtering_modifier else ""}{", " if sorting_modifier else ""}{sorting_modifier if sorting_modifier else ""}.'

            # If no provenance, only use the above `overall explanation` and skip explanation process
            if not result_set[0] or all(r is None for r in result_set[0][0]):
                data['explanation'] = exp
                data['result'] = ''
            else:
                exp += f'{" Among them, t" if len(result_set[0]) > 1 else " T"}he result, {"for example, " if len(result_set[0]) > 1 else ""}'
                # Randomly select one to-explained query result
                to_explain_result: PrettyTable = \
                    get_one_query_result(result_set, g_rewrite_sql)
                explainer.set_database_context(db_id, schema, schema2ids, relation2tables, table2statistics, table2alias, join_semantics, group_semantics)
                explainer.set_context(question=nl, sql=rewrite_sql, parsed_sql=g_rewrite_sql, result=to_explain_result)
                # Specific explanation of the query execution on one specific query result
                exp += explainer.explain()
                data['explanation'] = exp
                data['result'] = ", ".join([f"{field}: {value}" for field, value in \
                    zip(to_explain_result.field_names, to_explain_result.rows[0])]).strip()
            data['query'] = remove_join_conditions(data['query'])
            sqls.append(data['query'])
            data['label'] = 0
            json.dump(data, writer, indent=4)
            writer.write(',\n')
        except: pass

        beams = set()
        for beam_preds, beam_size in zip(beam_outputs_list, beam_size_list):
            beams.update(beam_preds[idx*beam_size: (idx+1)*beam_size])
        for sql in beams:
            data = {
                'db_id': db_id,
                'question': nl,
                'gold': gold
            }
            try:
                # print(f"index:{idx}\nNL: {nl}\nSQL: {sql}")
                p_sql, _ = get_sql(schema2ids, sql)
                data['query'] = sql_format(sql)
                rewrite_sql = rewrite_asterisk_query(sql, schema, db_file)
                p_rewrite_sql, table2alias = get_sql(schema2ids, rewrite_sql)
                result_set: Union[list, list, list] = execute_sqlite_query(rewrite_sql, p_rewrite_sql, db_file)  # rows, cols, types
                if not result_set[1]: continue # skip if sql is invalid
                
                # To have `accurate` join semantics, we calculate the statistics of the newly (joined) table
                # and determine the semantics of `asterisk` symbol in SQL from statistics
                table2statistics = get_db_statistics(sql, p_sql, table2alias, schema, cursor)
                join_semantics, group_semantics = parse_query_semantics(p_sql, schema, relation2tables, table2statistics)
                
                # Overall explanation of the query execution with following aspects:
                # - row and column counts the query returned
                # - any filtering criteria (WHERE clause)
                # - any sorting criteria (ORDER LIMIT clause)
                filtering_modifier: str = clause2dialect(p_sql, schema, group_semantics, cls_type='where', prefix='filtered by ', bracket=True)
                filtering_modifier += clause2dialect(p_sql, schema, group_semantics, cls_type='having', prefix=' and ', bracket=True) if filtering_modifier else \
                    clause2dialect(p_sql, schema, group_semantics, cls_type='having', prefix='filtered by ', bracket=True)
                sorting_modifier: str = clause2dialect(p_sql, schema, group_semantics, cls_type='orderBy', prefix='sorted ', bracket=True)
                exp = f'The query returns a result set with {len(result_set[1])} column{"s" if len(result_set[1]) > 1 else ""} ' + \
                    f'[{", ".join([f"{c}" if any(kw in "".join(c.split()).lower() for kw in EXCLUDE_KEYWORDS) or t== "NoneType" else f"{c} ({t})" for c, t in zip(result_set[1], result_set[2])])}] and ' + \
                    f'{len([r for r in result_set[0] if any(c for c in r)])} row{"s" if len(result_set[0]) > 1 else ""}{", " if filtering_modifier else ""}' + \
                    f'{filtering_modifier if filtering_modifier else ""}{", " if sorting_modifier else ""}{sorting_modifier if sorting_modifier else ""}.'

                # If no provenance, only use the above `overall explanation` and skip explanation process
                if not result_set[0] or all(r is None for r in result_set[0][0]):
                    data['explanation'] = exp
                    data['result'] = ''
                else:
                    exp += f'{" Among them, t" if len(result_set[0]) > 1 else " T"}he result, {"for example, " if len(result_set[0]) > 1 else ""}'
                    # Randomly select one to-explained query result
                    to_explain_result: PrettyTable = \
                        get_one_query_result(result_set, p_rewrite_sql)
                    explainer.set_database_context(db_id, schema, schema2ids, relation2tables, table2statistics, table2alias, join_semantics, group_semantics)
                    explainer.set_context(question=nl, sql=rewrite_sql, parsed_sql=p_rewrite_sql, result=to_explain_result)
                    # Specific explanation of the query execution on one specific query result
                    exp += explainer.explain()
                    data['explanation'] = exp
                    data['result'] = ", ".join([f"{field}: {value}" for field, value in \
                        zip(to_explain_result.field_names, to_explain_result.rows[0])]).strip()
                
                data['query'] = remove_join_conditions(data['query'])
                if data['query'] in sqls: continue
                # Remove duplicate instances
                sqls.append(data['query'])
                if eval_exec_match1(db_file, sql, gold, deepcopy(p_sql), deepcopy(g_sql)) == 1: continue  
                data['label'] = 1
                json.dump(data, writer, indent=4)
                writer.write(',\n')
            except: continue