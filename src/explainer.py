import os, re, sqlite3, random, logging, editdistance
from typing import Union
from colorama import Fore, Style
from prettytable import PrettyTable

from spider_utils.process_sql import get_sql
# from src.refiner.refiner import Refiner
from src.translator.xql2nl import Translator
from src.annotator.annotate import Annotator
from src.utils import EXCLUDE_KEYWORDS, Query_type, break_down_query, clause2dialect, execute_sqlite_query, extract_parsed_from_structure, \
    get_group_cols, get_having_cols, get_order_cols, get_where_cols, parse_query_semantics, rewrite_asterisk_query, use_alias, append_blank_space


has_agg = lambda s: s.find('(') > 0
has_asterisk = lambda s: s.find('*') > 0

class Explainer:
    
    def __init__(
        self, 
        database_dir: str
    ):
        self._db_dir = database_dir

        self._annotator = Annotator()
        self._translator = Translator()
        # self._refiner = Refiner()
    
        self._db_id = None

    def set_database_context(
        self, 
        db_id: str,
        schema: dict, 
        schema2ids: dict, 
        relation2tables: dict, 
        table2statistics: dict, 
        table2alias: dict, 
        join_semantics: str,
        group_semantics: dict
    ):
        if self._db_id != db_id:
            self._db_id = db_id
            self._db_file = os.path.join(self._db_dir, db_id, f'{db_id}.sqlite')
            self._schema = schema
            self._schema2ids = schema2ids
            self._relation2tables = relation2tables
            
        self._table2alias = table2alias
        self._table2statistics = table2statistics
        self._annotator.join_semantics = join_semantics
        self._annotator.group_semantics = group_semantics
    
    def set_context(
        self, 
        question: str, 
        sql: str, 
        parsed_sql: dict,
        result: PrettyTable
    ):
        self._question = question
        self._sql = sql
        self._parsed_sql = parsed_sql # rewrite parsed query
        self._to_explain_result = result
    
    def rewrite_query_for_provenance(
        self
    ) -> Union[dict, str, str]:
        def __is_not_count_having__(h):
            _, _, val_unit, _, _ = h[0] #TODO assume only can be one `having`
            _, col_unit, _ = val_unit
            agg_id, _, _ = col_unit
            return agg_id != 3
        def __fix_group_select_mismatch__(p_rewrite_sql, p_sql, field2):
            new_key = None
            if field2: new_key = {'groupBy': [(0, f'__{field2}__', False)]}
            else:
                _, sels = p_sql['select']
                non_agg_sel_cols  = [sel[1][1] for sel in sels if sel[0] == 0]
                group_col = [g for g in p_sql['groupBy']][0]
                if non_agg_sel_cols and all(s != group_col for s in non_agg_sel_cols):
                    sc = random.choice(non_agg_sel_cols)
                    new_key = {'groupBy': [sc]}
            if new_key: p_rewrite_sql.update(new_key)
            return p_rewrite_sql
        
        keep = {}
        rewrite_sql, rewrite_sql2 = self._sql, self._sql
        field = self._to_explain_result.field_names[0]
        field2 = self._to_explain_result.field_names[1] if len(self._to_explain_result.field_names) > 1 else None
        is_one_table_q = True if len(self._parsed_sql['from']['table_units']) == 1 else False
        # Rule #1 (Result Proj2Selection)
        if all(kw not in field.lower() for kw in EXCLUDE_KEYWORDS):
            # For single-table query, use column name directly
            # For multi-tables query, disambiguate table name with alias if original SQL defines alias
            pred_c = field.split('.')[1] if is_one_table_q else use_alias(field, self._table2alias)
            pred_v = self._to_explain_result.rows[0][0]
            pred = f'{pred_c} = {pred_v}' \
                if self._schema['col2type'][field] == 'number' else f'{pred_c} = \'{pred_v}\''
            # Find the position to insert the result-related predicate
            insert_p = len(self._sql)
            if self._parsed_sql['groupBy']: insert_p = self._sql.lower().rfind(' group by ')
            elif self._parsed_sql['orderBy']: insert_p = self._sql.lower().rfind(' order by ')
            elif self._parsed_sql['where']:
                # Bug? ... OR xx = 'yy' AND xxx = 'Tracy' will not filter not-Tracy rows during SQLite execution.
                # Hence add brackets with original `where` predicates before adding newly-created predicate
                where_end_p = self._sql.lower().rfind(' where ') + 7
                self._sql = self._sql[:where_end_p] + '[' + self._sql[where_end_p: insert_p] + ']' + self._sql[insert_p:]
                insert_p += 2
            if not self._parsed_sql['where']: pred = f' WHERE {pred}'
            else: pred = f' AND {pred}'
            rewrite_sql = self._sql[:insert_p] + pred + self._sql[insert_p:]
        # Rule #2 (Projection Redirection)
        from_p = rewrite_sql.lower().find(' from ')
        # Get the result-related columns
        cols_r = re.findall(r'\((.*?)\)', field) if has_agg(field) else [field]
        if has_asterisk(field): cols_r = []
        if field2: cols_r.append(field2)
        # Get the original sql-related columns (WHERE/GROUP/ORDER)
        cols_w_q = get_where_cols(self._parsed_sql)
        cols_g_q = get_group_cols(self._parsed_sql)
        cols_h_q = get_having_cols(self._parsed_sql)
        cols_o_q = get_order_cols(self._parsed_sql)
        # Get the primary keys (columns) of the related tables
        if len(self._parsed_sql['from']['table_units']) > 0:
            tables = extract_parsed_from_structure(self._parsed_sql['from'])
            cols_pk = [f'{t}.{c}' for t in tables for c in self._schema['primaries'][t]]
        # If no primaries found, search for potential pk with following patterns:
        # `id` / `{table_name}_id`/  `name` / `{table_name}_name`
        if not cols_pk: 
            # cols_pk = [f"{t}.{random.choice(schema['tab2cols'][t])}" for t in tables]
            for t in tables:
                visit = False
                for c in self._schema['tab2cols'][t]:
                    if c in ['name', f'{t}_name'] or editdistance.eval(c, f'{t}_name') < 2:
                        visit = True
                        cols_pk.append(f'{t}.{c}')
                if not visit:
                    for c in self._schema['tab2cols'][t]:
                        if c in ['id', f'{t}_id'] or editdistance.eval(c, f'{t}_id') < 2:
                            cols_pk.append(f'{t}.{c}')
        sels = list(set([tc.split('.')[1] if is_one_table_q else use_alias(tc, self._table2alias) \
                for tc in set(cols_r + cols_w_q + cols_g_q + cols_h_q + cols_o_q + cols_pk)]))
        # Re-construct SQL query
        rewrite_sql = 'SELECT ' + ', '.join(sels) + rewrite_sql[from_p:]
        # Rule #3 (Clause Pruning)
        # 1. If 1) GROUPBY exists but no HAVING, or HAVING with sum/max/min operations, remove GROUPBY and following (ORDERBY, LIMIT) clauses if exits
        # Otherwise, add group-related `predicate`
        # E.g., SELECT T1.id ,  T1.name FROM battle AS T1 JOIN ship AS T2 ON T1.id  =  T2.lost_in_battle JOIN death AS T3 ON T2.id  =  T3.caused_by_ship_id GROUP BY T1.id HAVING sum(T3.killed)  >  10
        if self._parsed_sql['groupBy'] and (not self._parsed_sql['having'] or __is_not_count_having__(self._parsed_sql['having'])):
            pos = rewrite_sql.lower().rfind(' group by ')
            p_rewrite_sql, _ = get_sql(self._schema2ids, rewrite_sql)
            p_sql, _ = get_sql(self._schema2ids, self._sql)
            # If `group` and `select` column mismatch, replace the `group` column with `select` column.
            p_rewrite_sql = __fix_group_select_mismatch__(p_rewrite_sql, p_sql, field2)
            keep['group'] = p_rewrite_sql
            rewrite_sql = rewrite_sql[: pos]

            if len(self._to_explain_result.field_names) > 1:
                field = self._to_explain_result.field_names[1]
                pred_c = field.split('.')[1] if is_one_table_q else use_alias(field, self._table2alias)
                pred_v = self._to_explain_result.rows[0][1]
                pred = f'{pred_c} = {pred_v}' \
                    if self._schema['col2type'][field] == 'number' else f'{pred_c} = \'{pred_v}\''
                if self._parsed_sql['where'] or 'AND' in rewrite_sql: rewrite_sql += ' AND ' + pred
                else: rewrite_sql += ' WHERE ' + pred
        # `rewrite_sql2` used for correct database execution
        rewrite_sql1 = re.sub(r'[\[|\]]', '', rewrite_sql)
        rewrite_sql2 = re.sub(r'[\[]', '(', rewrite_sql)
        rewrite_sql2 = re.sub(r'[\]]', ')', rewrite_sql2)
                
        return keep, rewrite_sql1, rewrite_sql2
    
    def get_query_result_table(
        self, 
        parsed_rewrite_sql: dict, 
        rewrite_sql: str
    ) -> PrettyTable:
        # Fix single-quote-value sql bug
        rewrite_sql2 = re.sub(r"'(?!s )", '"', rewrite_sql)
        rows, cols, _ = execute_sqlite_query(rewrite_sql2, parsed_rewrite_sql, self._db_file)
        prov = PrettyTable(field_names=cols)
        prov.add_rows(rows)
        
        return prov

    def explain(self):
        # explain only focus on simple queries
        sqls: list(tuple) = break_down_query(self._sql)
        exps = []
        is_main_no_provenance = False
        for (sql, type_) in sqls:
            # update context if existing multiple queries (IUE exists)
            if len(sqls) > 1:
                self._sql = rewrite_asterisk_query(sql, self._schema, self._db_file)
                self._parsed_sql, _ = get_sql(self._schema2ids, self._sql)
                self._annotator.join_semantics, self._annotator.group_semantics = \
                    parse_query_semantics(self._parsed_sql, self._schema, self._relation2tables, self._table2statistics)
                    
            # for except-type query, use SQL2NL to generate explanation directly for except queries, as the semantics does not rely on provenance
            if type_ == Query_type.EXCEPT:
                not_used_tables = extract_parsed_from_structure(self._parsed_sql['from'])
                filtering_modifier: str = clause2dialect(self._parsed_sql, self._schema, self._annotator.group_semantics, cls_type='where', not_used_tables=not_used_tables, prefix='')
                filtering_modifier += clause2dialect(self._parsed_sql, self._schema, self._annotator.group_semantics, cls_type='having', not_used_tables=not_used_tables, prefix=' and ') if filtering_modifier else \
                    clause2dialect(self._parsed_sql, self._schema, self._annotator.group_semantics, cls_type='having', not_used_tables=not_used_tables, prefix='')
                filtering_modifier += clause2dialect(self._parsed_sql, self._schema, self._annotator.group_semantics, cls_type='orderBy', not_used_tables=not_used_tables, prefix='sorted ')
                explanation = f'not in {self._annotator.join_semantics}' if not filtering_modifier else f'has no ' + filtering_modifier
                exps.append(explanation)
                continue
            # 1. Rewrite the SQL query
            keep, rewrite_sql, rewrite_sql2 = self.rewrite_query_for_provenance()
            # print(f"Rewrited SQL query: {Fore.YELLOW}{rewrite_sql}{Style.RESET_ALL}")
            
            # 2. Get the provenance information
            parsed_rewrite_sql, _ = get_sql(self._schema2ids, rewrite_sql)
            provenance = self.get_query_result_table(parsed_rewrite_sql, rewrite_sql2)
            # print(f"Provenance information (table)\n{Fore.YELLOW}{prov}{Style.RESET_ALL}")
            
            # for union-type query, the provenance may only exist in one of the queries (`main` or `union`).
            # we skip the non-provenance one if exists
            if type_ == Query_type.MAIN and any(s[1] == Query_type.UNION for s in sqls) and not provenance.rows:
                is_main_no_provenance = True
                continue
            if type_ == Query_type.UNION and not provenance.rows: continue
            
            # 3. Annotate the rewrited SQL query
            xql = self._annotator.annotate(parsed_rewrite_sql, provenance, self._to_explain_result, self._schema, keep)
            
            # 4. Translate provenance to explanation-purpose dialect
            self._translator.set_context(self._question, self._schema, self._annotator.group_semantics, self._annotator.join_semantics)
            explanation = self._translator.translate(xql, type_, is_main_no_provenance, bracket=False)
            
            if type_ == Query_type.INTERSECT: explanation = 'and ' + explanation
            elif type_ == Query_type.UNION: 
                if not is_main_no_provenance: explanation = 'or ' + explanation
            exps.append(explanation)
            # print(f"Dialect: {Fore.RED}{xnl}{Style.RESET_ALL}")        
            # 5. Refine dialect to natural language text
            # explanation = self._refiner.refine(self.result_str, explanation)
        explanation = ', '.join(exps)
        return append_blank_space(explanation, '.') if explanation[-1] != '.' else explanation
