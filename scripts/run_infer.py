from copy import deepcopy
import os, json, torch, argparse, sqlite3
from tqdm import tqdm
from typing import Union
from prettytable import PrettyTable

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from spider_utils.evaluation import build_foreign_key_map_from_json, build_valid_col_units, eval_exec_match1, rebuild_sql_col, rebuild_sql_val

from spinner import Spinner
from src.explainer import Explainer
from src.utils import EXCLUDE_KEYWORDS, clause2dialect, execute_sqlite_query, find_relationship_tables, \
    get_db_statistics, parse_query_semantics, remove_join_conditions, rewrite_asterisk_query, get_one_query_result, sql_format
from spider_utils.process_sql import Schema, get_schema, get_sql
from spider_utils.utils import read_single_dataset_schema_from_database

if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_name', type=str, required=True, help='model_name')
    arg_parser.add_argument('--beam_size', type=int, required=True, help='beam size')
    arg_parser.add_argument('--test_file_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--beam_output_file_path', type=str, required=True, help='raw beam output')
    arg_parser.add_argument('--nli_model_dir', type=str, required=True, help='nli model directory')
    arg_parser.add_argument('--table_file_path', type=str, required=True, help='table JSON path')
    arg_parser.add_argument('--db_dir', type=str, default='data/database')
    arg_parser.add_argument('--output_file_path', type=str, required=True, help='CycleSQL prediction output txt file')
    args = arg_parser.parse_args()

    
    kmaps = build_foreign_key_map_from_json(args.table_file_path)
    id2schema = {ex['db_id']: ex for ex in json.load(open(args.table_file_path))}
    data = json.load(open(args.test_file_path))

    file_name = os.path.splitext(os.path.basename(args.beam_output_file_path))[0]
    beam_serialization_file = os.path.join(os.path.dirname(args.output_file_path), f'{file_name}.json')
    nl2tuple = {}
    if os.path.exists(beam_serialization_file): 
        serialization = json.load(open(beam_serialization_file))
        for ex in serialization:
            nl2tuple[(ex['question'], ex['query'])] = (ex['explanation'], ex['result'])
            
    beam_preds = [l.strip(';\n ') for l in open(args.beam_output_file_path).readlines() if l.strip()]
    if 'resdsql' in args.model_name:
        if beam_preds[0][0] != 's': # exclude natural language questions for dev beam outputs
            beam_preds = [l for idx, l in enumerate(beam_preds) if idx % 9 != 0]
        beam_preds = [l.split('|')[1].strip() if '|' in l else l for l in beam_preds]
    elif args.model_name == 'smbop':
        beam_preds = [l.split('\t')[0].strip() for l in beam_preds]
    assert len(beam_preds) / args.beam_size == len(data)
        
    explainer = Explainer(args.db_dir)
    spinner = Spinner("Loading...")
    # Pre-load the pre-trained NLI model into memory
    config = AutoConfig.from_pretrained(args.nli_model_dir)
    with spinner:
        tokenizer = AutoTokenizer.from_pretrained(args.nli_model_dir)
        nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model_dir)
        nli_model = nli_model.to('cuda:0') if torch.cuda.is_available() else nli_model
                
    prev_db_id = None
    outputs, outputs2 = [], []
    for idx, ex in tqdm(enumerate(data), total=len(data)):
        gold_idx=-1
        
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
        try: g_sql, _ = get_sql(schema2ids, gold)
        except: pass

        sqls = []
        pred = None
        beams = beam_preds[idx*args.beam_size: (idx+1)*args.beam_size]
        for idx, sql in enumerate(beams):
            try: 
                p_sql, _ = get_sql(schema2ids, sql)
                if eval_exec_match1(db_file, sql, gold, deepcopy(p_sql), deepcopy(g_sql)) == 1: 
                    gold_idx=idx
                    break
            except:
                continue
            
        for idy, sql in enumerate(beams):
            try:
                if nl2tuple:
                    if not (nl, sql) in nl2tuple.keys(): continue
                    else:
                        exp, result = nl2tuple[(nl, sql)]
                        query = remove_join_conditions(sql_format(sql))
                else:
                    p_sql, _ = get_sql(schema2ids, sql)
                    # Generate natural language explanation (and result)
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
                        result = ''
                    else:
                        exp += f'{" Among them, t" if len(result_set[0]) > 1 else " T"}he result, {"for example, " if len(result_set[0]) > 1 else ""}'
                        # Randomly select one to-explained query result
                        to_explain_result: PrettyTable = \
                            get_one_query_result(result_set, p_rewrite_sql, None)
                        explainer.set_database_context(db_id, schema, schema2ids, relation2tables, table2statistics, table2alias, join_semantics, group_semantics)
                        explainer.set_context(question=nl, sql=rewrite_sql, parsed_sql=p_rewrite_sql, result=to_explain_result)
                        # Specific explanation of the query execution on one specific query result
                        exp += explainer.explain()
                        result = ", ".join([f"{field}: {value}" for field, value in \
                            zip(to_explain_result.field_names, to_explain_result.rows[0])]).strip()
                    
                    # data['beam'] = idy+1
                    query = remove_join_conditions(sql_format(sql))
                    if query in sqls: continue
                    # Remove duplicate instances
                    sqls.append(query)
                    
                    outputs2.append({
                        'db_id': db_id,
                        'question': nl,
                        'query': sql,
                        'explanation': exp,
                        'result': result
                    })
                    
                text = f"premise: (sql) {query} | (query result) {result} | (explanation) {exp} hypothesis: {nl}"
                input = tokenizer(text, truncation=True, return_tensors="pt")
                input = input.to('cuda:0') if torch.cuda.is_available() else input
                # Check the textual entailment relationship between NL query (as well as query result) and NL explanation
                with torch.no_grad(): #, spinner:
                    logits = nli_model(**input).logits
                    predicted_class_id = logits.argmax().item()
                    if predicted_class_id == 0: # entailment
                        pred = sql
                        if eval_exec_match1(db_file, sql, gold, deepcopy(p_sql), deepcopy(g_sql)) != 1 and gold_idx > -1:
                            p_sql, _ = get_sql(schema2ids, beams[gold_idx])
                            # Generate natural language explanation (and result)
                            rewrite_sql = rewrite_asterisk_query(sql, schema, db_file)
                            p_rewrite_sql, table2alias = get_sql(schema2ids, rewrite_sql)
                            result_set: Union[list, list, list] = execute_sqlite_query(rewrite_sql, p_rewrite_sql, db_file)  # rows, cols, types
                            if not result_set[1]: continue # skip if sql is invalid
                            print(f"\nNL query: {nl}\nPredicted SQL index: {idx}\nPredicted SQL: {sql}\nPredicted SQL Explanation:{exp}\nResult: {result}")
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
                                result = ''
                            else:
                                exp += f'{" Among them, t" if len(result_set[0]) > 1 else " T"}he result, {"for example, " if len(result_set[0]) > 1 else ""}'
                                # Randomly select one to-explained query result
                                to_explain_result: PrettyTable = \
                                    get_one_query_result(result_set, p_rewrite_sql, None)
                                explainer.set_database_context(db_id, schema, schema2ids, relation2tables, table2statistics, table2alias, join_semantics, group_semantics)
                                explainer.set_context(question=nl, sql=rewrite_sql, parsed_sql=p_rewrite_sql, result=to_explain_result)
                                # Specific explanation of the query execution on one specific query result
                                exp += explainer.explain()
                                result = ", ".join([f"{field}: {value}" for field, value in \
                                    zip(to_explain_result.field_names, to_explain_result.rows[0])]).strip()
                                
                            print(f"Gold SQL Index: {gold_idx}\nGold SQL: {gold}\nGold SQL Explanation: {exp}\nResult: {result}")
                        break
                # data['label'] = 0 if eval_exec_match1(db_file, sql, gold, deepcopy(p_sql), deepcopy(g_sql)) == 1 else 1
            except: continue

        if not pred:
            add = False
            for sql in beams:
                try: 
                    p_sql, _ = get_sql(schema2ids, sql)
                    p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], schema2ids)
                    p_sql = rebuild_sql_val(p_sql)
                    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmaps[db_id])
                    cursor.execute(sql)
                    outputs.append(sql)
                    add = True
                    break
                except: continue
            if not add: outputs.append(beams[0])
        else: outputs.append(pred)
    
    if not os.path.exists(beam_serialization_file): 
        with open(beam_serialization_file, "w") as f:
            json.dump(outputs2, f, indent=4)
            
    with open(args.output_file_path, "w") as f:
        for o in outputs:
            f.write(f"{o}\n")