import os, json, sqlite3, argparse, logging
from typing import Union
from prettytable import PrettyTable
from tqdm import tqdm

from src.explainer import Explainer
from spider_utils.process_sql import Schema, get_schema, get_sql
from spider_utils.utils import read_single_dataset_schema_from_database
from src.utils import EXCLUDE_KEYWORDS, SQL_OPS, clause2dialect, get_db_statistics, find_relationship_tables, parse_query_semantics, get_one_query_result, execute_sqlite_query, rewrite_asterisk_query

is_from_subquery = lambda sql: 'from (' in sql.lower()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="json input file for nl2sql data")
    parser.add_argument("--tables_file", help="json file for database schema")
    parser.add_argument("--database_dir", help="directory for sqlite databases")
    parser.add_argument("--output_file", help="json output file for nl2sql data")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input_file_name = os.path.basename(args.input_file)
    if os.path.exists(f"output/logfile.{input_file_name}"): os.remove(f"output/logfile.{input_file_name}")
    logging.basicConfig(level=logging.DEBUG, filename=f"output/logfile.{input_file_name}", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")

    id2schema = {}
    for ex in json.load(open(args.tables_file)):
        id2schema[ex['db_id']] = ex
    data = json.load(open(args.input_file))
    
    explainer = Explainer(args.database_dir)
    
    outputs = []
    prev_db_id = None
    for i, ex in tqdm(enumerate(data), total = len(data), desc ="Processing on explaining"):
        # if i < 1536: continue
        db_id = ex['db_id']
        nl = ex['question']
        sql = ex['query'].strip(';')
        # if i < 43 or '(select' not in sql.lower(): continue
        # if all(kw not in sql.lower() for kw in SQL_OPS): continue
        print(f"index: {i}\nNL: {nl}\nSQL: {sql}")
        if is_from_subquery(sql): continue
        # **********************************************************************************************************
        if db_id != prev_db_id:
            db_file = os.path.join(args.database_dir, db_id, f'{db_id}.sqlite')
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors = 'ignore')
            cursor = conn.cursor()
            schema = read_single_dataset_schema_from_database(id2schema[db_id], cursor)
            schema2ids = Schema(get_schema(db_file))
            relation2tables = find_relationship_tables(schema)
            prev_db_id = db_id
        
        try:
            exp = ""
            p_sql, _ = get_sql(schema2ids, sql)
            rewrite_sql = rewrite_asterisk_query(sql, schema, db_file)
            p_rewrite_sql, table2alias = get_sql(schema2ids, rewrite_sql)
            result_set: Union[list, list, list] = execute_sqlite_query(rewrite_sql, p_rewrite_sql, db_file)  # rows, cols, types
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
                ex['explanation'] = exp
                ex['result'] = ''
                # print(f'Explanation: {exp}')
            else:
                exp += f'{" Among them, t" if len(result_set[0]) > 1 else " T"}he result, {"for example, " if len(result_set[0]) > 1 else ""}'
                # Randomly select one to-explained query result
                exclude_key = ex['tag'] if 'tag' in ex.keys() else None
                to_explain_result: PrettyTable = \
                    get_one_query_result(result_set, p_rewrite_sql, exclude_key)
                # print(to_explain_result)
                explainer.set_database_context(db_id, schema, schema2ids, relation2tables, table2statistics, table2alias, join_semantics, group_semantics)
                explainer.set_context(question=nl, sql=rewrite_sql, parsed_sql=p_rewrite_sql, result=to_explain_result)
                # Specific explanation of the query execution on one specific query result
                exp += explainer.explain()
                ex['explanation'] = exp
                # print(f'Explanation: {exp}')
                ex['result'] = ", ".join([f"{field}: {value}" for field, value in \
                    zip(to_explain_result.field_names, to_explain_result.rows[0])]).strip()
                
            if 'query_toks' in ex.keys():
                del ex['query_toks']
                del ex['query_toks_no_value']
                del ex['question_toks']
                del ex['sql']
            outputs.append(ex)
        except:
            print(f"index: {i}\nNL: {nl}\nSQL: {sql}")
            continue

    with open(args.output_file, "w") as out:
        json.dump(outputs, out, indent=4)




if __name__ == "__main__":
    main()