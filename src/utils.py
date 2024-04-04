import re, random, signal, sqlite3, threading, time
from typing import Union
import networkx as nx
from enum import IntEnum
from prettytable import PrettyTable

from src.word_dict import WORD_DICT
from spider_utils.process_sql import Schema, get_schema, get_sql


SAMPLING_THRESHOLD = 3
SQL_KEYWORDS = [
    'count(', 'sum(', 'avg(', 'max(', 'min(', 'group', 'by', 'select', 'from', 'order', 'limit', 'on', 'and', 'or', \
        'intersect', 'union', 'except', 'join', 'where', 'distinct', 'desc', 'asc', 'not', 'like', 'in', 'having']
EXCLUDE_KEYWORDS = ['count(', 'avg(', 'sum(', 'min(', 'max(', '*']
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')

class Where_opt(IntEnum):
    NOT = 0
    BETWEEN = 1
    EQ = 2
    GT = 3
    LT = 4
    GTE = 5
    LTE = 6
    NEQ = 7
    IN = 8
    LIKE = 9
    IS = 10
    EXISTS = 11

    
class Agg_opt(IntEnum):
    NONE = 0
    MAX = 1
    MIN = 2
    COUNT = 3
    SUM = 4
    AVG = 5

class Query_type(IntEnum):
    MAIN = 0
    INTERSECT = 1
    UNION = 2
    EXCEPT = 3
    
def table_graph_types():
    # Subject-Relationship-Object table graph
    G1 = nx.DiGraph()
    nx.add_path(G1, [1, 2])
    nx.add_path(G1, [3, 2])
    # Object with attributes table graph
    G2 = nx.DiGraph()
    nx.add_path(G2, [1, 2])
    nx.add_path(G2, [1, 3])
    # Object A with B with C table graph
    G3 = nx.DiGraph()
    nx.add_path(G3, [1, 2, 3])
    return [G1, G2, G3]
TABLE_GRAPHS = table_graph_types()

append_blank_space = lambda s, tok=' ': s + tok

def extract_parsed_from_structure(f):
    return [t[1].strip('_') if t[0] == 'table_unit' else t[1] for t in f['table_units']]

def extract_parsed_select_structure(s, idx=0):
    _, sels = s
    agg_id, val_unit = sels[idx]
    _, col_unit, _ = val_unit
    _, col_id, _ = col_unit
    col = col_id.strip('_')
    return agg_id, col

def extract_parsed_cond_structure(w):
    not_op, op_id, val_unit, val1, val2 = w
    _, col_unit, _ = val_unit
    _, col_id, _ = col_unit
    col = col_id.strip('_')
    val1 =  val1.strip('\'"') if isinstance(val1, str) else val1
    
    return not_op, op_id, col, val1, val2
        
def extract_parsed_group_structure(g, idx=0):
    _, col_id, _ = g[idx]
    col = col_id.strip('_')
    return col

def extract_parsed_order_limit_structure(o, l=None, idx=0):
    asc = o[0]
    _, col_unit, _ = o[1][idx]
    agg_id, col_id, _ = col_unit
    col = col_id.strip('_')
    return asc, agg_id, col, l
     
def get_where_cols(p_sql):
    if not p_sql['where']: return []

    conds = [unit for unit in p_sql['where'][::2]]
    return [extract_parsed_cond_structure(cond)[2] for cond in conds]

def get_group_cols(p_sql):
    if not p_sql['groupBy']: return []

    num = len(p_sql['groupBy'])
    return [extract_parsed_group_structure(p_sql['groupBy'], idx=i) for i in range(num)]

def get_having_cols(p_sql):
    if not p_sql['having']: return []

    conds = [unit for unit in p_sql['having'][::2]]
    return [extract_parsed_cond_structure(cond)[2] for cond in conds if extract_parsed_cond_structure(cond)[2] != 'all']

def get_order_cols(p_sql):
    if not p_sql['orderBy']: return []
    
    num = len(p_sql['orderBy'][1])
    return [extract_parsed_order_limit_structure(p_sql['orderBy'], idx=i)[2] for i in range(num) if extract_parsed_order_limit_structure(p_sql['orderBy'], idx=i)[2] != 'all']


def get_full_column_name(idx, col, p_sql):
    has_agg = lambda s: s.find('(') > 0
    _, col_id = extract_parsed_select_structure(p_sql['select'], idx)
    col_id = '*' if col_id == 'all' else col_id
    if has_agg(col):
        col = re.sub(r'(?<=\().*?(?=\))', col_id, col)
        col = re.sub('(.*?)\s*(\()', r'\1\2', col)
    else: col = col_id

    return col

def get_column_name(col, with_table=False, bracket=False):
    t, col = col.split('.')
    t += ' '
    col = col.replace('_', ' ')
    return f'{"[" if bracket else ""}{t if with_table else ""}{col}{"]" if bracket else ""}'

def use_alias(field, alias_dict):
    t, c = field.split('.')
    alias = list(filter(lambda x: alias_dict[x] == t, alias_dict))[0]
    return f'{alias}.{c}'

def break_down_query(
    sql: str
) -> list:
    """
    Break down a complex SQL query into small pieces (simple queries)

    Args:
        sql: A complex sql string
    
    Return:
        A tuple list where first element is a sql string and the second one is corresponding query type (i.e., main/intersect/except/union)

    #TODO support multiple intersects/excepts/unions
    """
    split_indice = []
    toks = sql.strip(';').split()
    left_brackets = 0
    for idx, t in enumerate(toks):
        if '(' in t: left_brackets += 1
        if ')' in t: left_brackets -= 1
        if any(tt in t.lower() for tt in SQL_OPS) and left_brackets == 0: split_indice.append(idx)
    split_indice.append(len(toks))

    start = split_indice[0]
    sqls: list(tuple) = []
    main_sql = ' '.join(toks[: split_indice[0]])
    sqls.append((main_sql, Query_type.MAIN))
    # Finally add IUE-type SQLs
    for i in split_indice[1:]:
        type = toks[start].strip().upper()
        s = ' '.join(toks[start+1: i])
        # if Query_type[type] == Query_type.EXCEPT:
        #     sqls.insert(0, (s, Query_type.EXCEPT))  # For `except`, we put in font of `main` SQL for explaining purpose.
        sqls.append((s, Query_type[type]))
        start = i
        
    return sqls

def rewrite_asterisk_query(
    sql: str, schema: dict, db_file: str
) -> str:
    """Rewrite `SELECT *` SQL to `SELECT col1, col2, ...`

    Args:
        sql: SQL string
        schema: Database schema JSON dict
        db_file: Database file path

    Returns:
        The rewritten SQL string
    """
    p_sql, _ = get_sql(Schema(get_schema(db_file)), sql)
    if sql[7] != '*': return sql
    
    cols = set()
    for t in p_sql['from']['table_units']:
        normalize_t = t[1].strip('_')
        for c in schema['tab2cols'][normalize_t]:
            cols.add(f'{normalize_t}.{c}')

    sel = ', '.join(cols)
    insert_p = sql.lower().find(' from ')
    rewrite_sql = 'SELECT ' + sel + sql[insert_p:]
    
    return rewrite_sql

def find_relationship_tables(schema: dict):
    res = {}

    G = nx.DiGraph()
    for t in schema['table_names_original']:
        G.add_node(t.lower())
    
    for fk in schema['foreigns']:
        end, start = fk.split('-')
        G.add_edge(start, end)
    
    relation_nodes = [node for node in G.nodes if G.out_degree(node) == 0 and G.in_degree(node) == 2]
    for n in relation_nodes:
        res[n] = [nn for nn in G.predecessors(n)]

    return res

def get_db_statistics(
    sql: str, p_sql: dict, alias_dict: dict, schema: dict, _cursor: sqlite3.Connection
) -> dict:
    result = {}
    from_p = sql.lower().find(' from ')
    end_p = min({len(sql), (sql + ' where ').lower().find(' where '), (sql + ' group ').lower().find(' group '), (sql + ' order ').lower().find(' order '), \
        (sql + ' intersect ').lower().find(' intersect '), (sql + ' union ').lower().find(' union '), (sql + ' except ').lower().find(' except ')})
    # Get the primary keys (columns) of the related tables
    tables = [t[1].strip('_') for t in p_sql['from']['table_units']]
    for t in tables:
        if not schema["primaries"][t]: continue

        pk = schema["primaries"][t].pop()
        field = f'{t}.{pk}'
        sql1 = 'SELECT ' + f'COUNT(DISTINCT {pk if len(tables) == 1 else use_alias(field, alias_dict)}) ' + sql[from_p:end_p]
        _cursor.execute(sql1)
        rows = [list(row) for row in _cursor.fetchall()]
        result[t] = rows[0][0]

    return result

def parse_query_semantics(
    p_sql: dict, schema: dict, relation_tables: dict, t2distincts: dict
) -> Union[str, dict]:

    def __CONVERT_TABLE_NAME(orig, schema):
        for idx, t in enumerate(schema['table_names_original']):
            if t.lower() == orig: return schema['table_names'][idx].lower()
    def __CONVERT_TABLE_SEMANTICS(orig, schema):
        # If 1) no underscore in the table name, or 2) no primary key, or 3) exisiting composite primary key, use the annotation directly;
        # Otherwise, use the primary key as the table name or revert back to use the table name.
        if '_' not in orig or \
            not schema['primaries'][orig] or \
                len(schema['primaries'][orig]) > 1: 
                return  __CONVERT_TABLE_NAME(orig, schema)

        pk = list(schema['primaries'][orig]).pop()
        key = f'{orig}.{pk}'
        pk_anno = schema['column_names_mapping'][key]
        pk_anno_norm = pk_anno.replace('id', '').strip()
        if not pk_anno_norm: pk_anno_norm = orig.replace('_', ' ')
        return pk_anno_norm

    label = ""
    group_semantics = {}
    tables = extract_parsed_from_structure(p_sql['from'])
    # Only handle non-nested queries
    if isinstance(tables[0], dict): return label, group_semantics
    # If no table join, return the name of table as entity
    if len(tables) == 1:
        label = __CONVERT_TABLE_NAME(tables[0], schema)
        # Obtain the *star* semantics from the primary key, 
        # or derived from the table name if no meaningful semantics related to the key
        group_semantics[tables[0]] = __CONVERT_TABLE_SEMANTICS(tables[0], schema)
        group_semantics['global'] = group_semantics[tables[0]]
    else:
        # Contruct the corresponding table graph
        G = nx.DiGraph()
        G.add_node(tables[0])
        prev = [tables[0]]
        for t in tables[1:]:
            G.add_node(t)
            add = False      
            for tt in reversed(prev):
                if f'{tt}-{t}' in schema['foreigns']:
                    G.add_edge(t, tt)
                    add = True
                    break
                elif f'{t}-{tt}' in schema['foreigns']:
                    G.add_edge(tt, t)
                    add = True
                    break
            if not add: G.add_edge(prev[-1], t)
            prev.append(t)

        # If three-tables queries, assign the label based on the graph topology
        # We pre-define three types of graph topologies, where each represents a specific join semantics
        if G.number_of_nodes() == 3:
            if nx.is_isomorphic(G, TABLE_GRAPHS[0]):
                relation = [node for node in G.nodes if G.out_degree(node) == 0][0]
                subject_object = [node for node in G.nodes if G.in_degree(node) == 0]
                assert len(subject_object) == 2
                # Which is subject and which is object?
                label = ' with '.join([__CONVERT_TABLE_NAME(n, schema) for n in subject_object])
                group_semantics[subject_object[0]] = __CONVERT_TABLE_SEMANTICS(subject_object[1], schema)
                group_semantics[subject_object[1]] = __CONVERT_TABLE_SEMANTICS(subject_object[0], schema)
                group_semantics[relation] = 'UNK'
                if subject_object[1] in t2distincts.keys() and subject_object[0] in t2distincts.keys():
                    group_semantics['global'] = __CONVERT_TABLE_SEMANTICS(subject_object[0], schema) \
                        if t2distincts[subject_object[1]] < t2distincts[subject_object[0]] else __CONVERT_TABLE_SEMANTICS(subject_object[1], schema)
                else: group_semantics['global'] = __CONVERT_TABLE_SEMANTICS(relation, schema)
            elif nx.is_isomorphic(G, TABLE_GRAPHS[1]):
                subject = [node for node in G.nodes if G.in_degree(node) == 0]
                others = [node for node in G.nodes if G.in_degree(node) != 0]
                assert len(subject) == 1
                label = __CONVERT_TABLE_NAME(subject[0], schema)
                group_semantics[others[0]] = __CONVERT_TABLE_SEMANTICS(subject[0], schema)
                group_semantics[others[1]] = __CONVERT_TABLE_SEMANTICS(subject[0], schema)
                group_semantics[subject[0]] = __CONVERT_TABLE_SEMANTICS(subject[0], schema)
                group_semantics['global'] = __CONVERT_TABLE_SEMANTICS(subject[0], schema)
            elif nx.is_isomorphic(G, TABLE_GRAPHS[2]):
                nodes = []
                while(G.number_of_nodes()):
                    cursor = [node for node in G.nodes if G.in_degree(node) == 0][0]
                    nodes.append(cursor)
                    G.remove_node(cursor)
                label = ' with '.join([__CONVERT_TABLE_NAME(n, schema) for n in nodes])
                group_semantics[nodes[0]] = ' with '.join([__CONVERT_TABLE_NAME(n, schema) for n in nodes[1:]])
                group_semantics[nodes[1]] = ' with '.join([__CONVERT_TABLE_NAME(n, schema) for n in [nodes[0], nodes[2]]])
                group_semantics[nodes[2]] = ' with '.join([__CONVERT_TABLE_NAME(n, schema) for n in nodes[:-1]])
                group_semantics['global'] = ' with '.join([__CONVERT_TABLE_NAME(n, schema) for n in nodes])
            else:
                assert 1 == 0
        else:
            nodes = []
            while(G.number_of_nodes()):
                cursor = [node for node in G.nodes if G.in_degree(node) == 0]
                if not cursor: return label, group_semantics
                cursor = cursor[0]
                nodes.append(cursor)
                G.remove_node(cursor)
            # If two-table queries that include one relationship-type table
            if len(nodes) == 2 and nodes[-1] in relation_tables.keys() and nodes[0] in relation_tables[nodes[-1]]:
                replacement = [t for t in relation_tables[nodes[-1]] if t != nodes[0]][0]
                group_semantics[nodes[-1]] = __CONVERT_TABLE_SEMANTICS(replacement, schema)
                nodes[-1] = replacement
            label = ' with '.join([__CONVERT_TABLE_NAME(n, schema) for n in nodes])
            if nodes[1] in t2distincts.keys() and nodes[0] in t2distincts.keys():
                group_semantics['global'] = __CONVERT_TABLE_SEMANTICS(nodes[1], schema) \
                    if t2distincts[nodes[0]] < t2distincts[nodes[1]] else __CONVERT_TABLE_SEMANTICS(nodes[0], schema)
            elif nodes[1] not in t2distincts.keys(): group_semantics['global'] = __CONVERT_TABLE_SEMANTICS(nodes[1], schema)
            else: group_semantics['global'] = __CONVERT_TABLE_SEMANTICS(nodes[0], schema)
            if len(nodes) == 2:
                group_semantics[nodes[0]] = __CONVERT_TABLE_SEMANTICS(nodes[-1], schema)
                group_semantics[nodes[-1]] = __CONVERT_TABLE_SEMANTICS(nodes[0], schema)
            else:
                for n in nodes:
                    group_semantics[n] = ' with '.join([__CONVERT_TABLE_NAME(nn, schema) for nn in nodes if nn != n])
    
    # Hard-code to use `gobal`-key semantics if group exists
    if p_sql['groupBy']:
        col = extract_parsed_group_structure(p_sql['groupBy'])
        # if foreign key, use the corresponding primary key instead
        if col in schema['foreign_key_columns'] and schema['foreign_column_pairs'][col].split('.')[0] in tables:
            gcol_t = schema['foreign_column_pairs'][col].split('.')[0]
        else: gcol_t = col.split('.')[0]
        group_semantics['global'] = group_semantics[gcol_t]
        # If GROUP exisits, change join semantics as group semantics instead
        label = group_semantics['global']

    return label, group_semantics

def is_agg_group_query(p_sql):
    if p_sql['groupBy']:
        _, sels = p_sql['select']
        for sel in sels:
            agg_id, _ = sel
            if agg_id > 0:
                return True
    return False

def is_only_select_query(p_sql):
    if not p_sql['where'] and not p_sql['groupBy'] and not p_sql['orderBy'] and \
        not p_sql['intersect'] and not p_sql['union'] and not p_sql['except']: 
         return True
    
    return False

def execute_sqlite_query(
    sql: str, p_sql: dict, db_file: str
) -> Union[list, list, list]:
    """Execute a SQL string against on database (Multi-thread version to avoid long-running queries)
    

    Args:
        sql: SQL string
        p_sql: The parsed sql structure
        db_file: SQLite database file path

    Return:
        The rows and columns in the result set
    """
    
    def interrupt(signum, frame):
            pass
    def execute_sql(sql, db_file):
        conn = sqlite3.connect(db_file)
        conn.text_factory = lambda b: b.decode(errors = 'ignore')
        cursor = conn.cursor()
        try: cursor.execute(sql)
        except: return
        global rows, cols, types
        rows = [list(row) for row in cursor.fetchall()]
        # row exists and make sure not all `None` value
        if rows: types = [type(c).__name__ for c in rows[0]]
        if not types: types = ['NoneType' for _ in cursor.description]
        cols = [get_full_column_name(i, column[0], p_sql) \
                for i, column in enumerate(cursor.description)]
    
    global rows, cols, types
    rows, cols, types = [], [], []
    signal.signal(signal.SIGINT, interrupt)
    mainthread = threading.Thread(target=execute_sql, args=[sql, db_file])
    mainthread.start()
    cnt = 0
    while mainthread.isAlive() and cnt < 100:
        time.sleep(0.05)
        cnt += 1

    return rows, cols, types

def get_one_query_result(
    result_set: Union[list, list, list], p_sql: dict, exclude_key: str=None
) -> PrettyTable:
    rows, columns, _ = result_set
    r_i = random.choice(range(len(rows)))
    # For `SELECT agg(), col ... GROUP BY col ...` query, 
    # make sure the explanation is meaningful (choose agg-realted column)
    if is_agg_group_query(p_sql):# or __is_only_select_query__(p_sql):
        c_i, c_j = 0, -1
        for idx, col in enumerate(cols):
            if any(kd in col.lower() for kd in EXCLUDE_KEYWORDS): c_i = idx
            else: c_j = idx
        x_col = cols[c_i]
        if c_j == -1:
            xres = PrettyTable(field_names=[x_col])
            xres.add_row([rows[r_i][c_i]])
        else:
            xres = PrettyTable(field_names=[x_col, cols[c_j]])
            xres.add_row([rows[r_i][c_i], rows[r_i][c_j]])
    else:
        c_i = -1
        if exclude_key:
            for ii, col in enumerate(cols):
                if exclude_key.lower() == col.lower(): c_i = ii
        else: c_i = random.choice(range(len(cols)))
        if c_i == -1: return None
        
        c_i = random.choice(range(len(cols)))
        # c_i = 0
        # r_i = 3
        x_col = cols[c_i]
        xres = PrettyTable(field_names=[x_col])
        xres.add_row([rows[r_i][c_i]])

    return xres

def sql_format(sql: str):
    
    table_alias_pattern = re.compile(r'[a-zA-Z_]+[\w]*\.')
    sql = re.sub(table_alias_pattern, '', sql)

    join_pattern = [" INNER ", " OUTER ", " LEFT ", " inner ", " outer ", " left "]
    join_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, join_pattern)) + r')\b')
    sql = re.sub(join_pattern, ' ', sql)

    alias_pattern = re.compile(r'[a|A][s|S] [t|T][\d]* ')
    sql = re.sub(alias_pattern, '', sql)
    alias_pattern = re.compile(r'[a|A][s|S] [\w]* ')
    sql = re.sub(alias_pattern, '', sql)
    # *** Place \( in a capture group (...) and then use \2 to refer to it ***
    agg_space_pattern = re.compile(r'([\w]*)([ ]*)(\()')
    sql = re.sub(agg_space_pattern, r'\1\3 ', sql)
    agg_space_pattern1 = re.compile(r'([\w]*)(\))')
    sql = re.sub(agg_space_pattern1, r'\1 \2', sql)

    quote_pattern = re.compile(r'[\'](.*?)[\']')
    sql = re.sub(quote_pattern, r'"\1"', sql)
    
    # Lowercase entire string except for substrings in quotes
    sql = re.sub(r"\b(?<!\")(\w+)(?!\")\b", lambda match: match.group(1).lower(), sql)
    toks = [t.upper() if t in SQL_KEYWORDS else t for t in sql.strip().split()]
    
    return ' '.join(toks)

def remove_join_conditions(c):
    if ' on ' not in c.lower(): return c

    toks = []
    join = -1
    sql_toks = c.split()
    for i, t in enumerate(sql_toks):
        if t in ['ON', 'on']:
            join = 3
            if i + 4 < len(sql_toks) and sql_toks[i + 4] in ['and', 'AND', 'or', 'OR']: join = 7
            continue
        elif join > 0: 
            join -= 1
        else: 
            toks.append(t)
        
    return ' '.join(toks)
   
def subq2text(p_sql: dict, schema: dict, group_semantics: dict, outer_col: str, bracket=False):
    """Translate a WHERE-nested-subquery to corresponding natural language text.
        We consider the following subquery types:
        1) aggregation-type: where age > ( select avg( age ) from singer )
        1) inconsistent-type: where nationality not in (select name from conductor join orchestra where nationality = "USA")'
        3) order-limit-type: where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)
        4) where-type: where stuid not in (select t1.stuid from student as t1 join has_pet as t2 join pets as t3 where t3.pettype = "cat")
        5) only-from-type: where stadium_id not in (select stadium_id from concert)
        6) iue-type: where airportCode not in (select sourceairport from flights union select destAirport from flights)
    Args:
        p_sql: The parsed sql structure
        group_semantics: The semantics mapping used for GROUP-related operations
        outer_col: To check if the outer operand is the the same with the inner one
        bracket: Add brackets surrounding the column string

    Return:
        The text for the subquery 
    """
    def __is_column_consistent__(outer_col, inner_col, schema):
        """Check if the outer operand is not consistent (either the same or having fk relation) with the operand in subquery"""
        if inner_col not in schema['foreign_column_pairs'].keys(): return inner_col == outer_col
        return (outer_col == inner_col) or schema['foreign_column_pairs'][inner_col] == outer_col
    
    text = ""
    not_used_tables = extract_parsed_from_structure(p_sql['from'])
    agg_id, col = extract_parsed_select_structure(p_sql['select'])
    is_outer_inner_consistent = __is_column_consistent__(outer_col, col, schema=schema)
    
    # first add any aggregation semantics, either from `select` itself or `order limit` on the select column
    if agg_id != Agg_opt.NONE: text += f'{WORD_DICT[Agg_opt(agg_id).name.lower()]} '
    elif p_sql['orderBy'] and p_sql['limit']:
        asc, _, o_col, limit = extract_parsed_order_limit_structure(p_sql['orderBy'], p_sql['limit'], 0)
        if __is_column_consistent__(col, o_col, schema=schema): 
            key = f'{asc}_limit_{limit}'
            text += f'{WORD_DICT[key]} '
    # next add `select` semantics as subject semantics
    # 1. if not consistent with outer column, use select-column in subquery as subject
    # 2. if consistent, use `the ones` as subject
    if not is_outer_inner_consistent:
        with_table = False if col == "all" else True
        text += f'{get_column_name(col, with_table=with_table, bracket=bracket) if col != "all" else group_semantics["global"]}'
        if with_table: not_used_tables.remove(col.split('.')[0])
    else: text += f'{"the " if not text else ""}ones'
    # then add `where` semantics if exists
    conjs = [''] + [f' {c} ' for c in p_sql['where'][1::2]]
    conds = [c for c in p_sql['where'][::2]]
    for cond, conj in zip(conds, conjs):
        if conj == '': text += ' that ' # add `that` to begin condition semantics
        text += conj
        text += cond2text(cond, schema, group_semantics, not_used_tables, bracket=bracket)
    # then add `order limit` semantics if exists
    if p_sql['orderBy'] and p_sql['limit']:
        asc, agg_id, o_col, limit = extract_parsed_order_limit_structure(p_sql['orderBy'], p_sql['limit'], 0)
        if not __is_column_consistent__(col, o_col, schema=schema):
            text += " with "
            with_table = True if o_col != all and o_col.split('.')[0] in not_used_tables else False
            key = f'{asc}_limit_{p_sql["limit"]}'
            text += f"{WORD_DICT[key]} "
            text += f'{WORD_DICT[Agg_opt(agg_id).name.lower()]} {get_column_name(o_col, with_table=with_table, bracket=bracket) if o_col != "all" else group_semantics["global"]} ' 
            if with_table: not_used_tables.remove(o_col.split('.')[0])
    # add `from` semantics if no above semantics captured (e.g., where stadium_id not in (select stadium_id from concert))
    if text == 'the ones':
        tables = list(map(lambda x: x.replace('_', ' '), extract_parsed_from_structure(p_sql['from'])))
        text += f' in {" with ".join(tables)}'
    # last add `iue` semantics if exists
    if p_sql['union']: 
        text, iue_text = iue2text(p_sql['union'], schema, group_semantics, outer_col, text, col, bracket=bracket)
        text += ' or ' + iue_text
    if p_sql['intersect']: 
        text, iue_text = iue2text(p_sql['intersect'], schema, group_semantics, outer_col, text, col, bracket=bracket)
        text += ' and ' + iue_text
    if p_sql['except']: 
        text, iue_text = iue2text(p_sql['except'], schema, group_semantics, outer_col, text, col, bracket=bracket)
        text += ' except ' + iue_text
        
    text = ' '.join(text.split())
    return text

def iue2text(p_sql: dict, schema: dict, group_semantics: dict, outer_col: str, text: str='', col: str=None, bracket: bool=False):
    iue_text = subq2text(p_sql, schema, group_semantics, outer_col, bracket=bracket)
    # use select-column semantics instead. (e.g., where airportCode not in (select sourceairport from flights union select destAirport from flights))
    if text == iue_text and 'ones' in text:
        text = text.replace('ones', get_column_name(col))
        iue_text = iue_text.replace('ones', get_column_name(extract_parsed_select_structure(p_sql['select'])[1]))
        
    return text, iue_text

def cond2text(cond: tuple, schema: dict, group_semantics: dict, not_used_tables: list=None, ignore_col=False, bracket=False):
    """Translate a WHERE condition to corresponding natural language text.

    Args:
        p_sql: The parsed sql structure
        group_semantics: The semantics mapping used for GROUP-related operations
        ignore_col: If ignoring the operand semantics in the nested-subquery condition
        bracket: Add brackets surrounding the column string

    Return:
        The text for the subquery 
    """
    text = ''
    not_op, op_id, col, val1, val2 = extract_parsed_cond_structure(cond)
    with_table = True if col != all and not_used_tables and col.split('.')[0] in not_used_tables else False
    if not ignore_col:
        text += f'{get_column_name(col, with_table=with_table, bracket=bracket)} ' if col != "all" else f'{group_semantics["global"]} '
    if op_id == Where_opt.IN:
        assert isinstance(val1, dict)
        text += f'{"not " if not_op else ""}{WORD_DICT[Where_opt.IN.name.lower()]} {subq2text(val1, schema, group_semantics, outer_col=col, bracket=bracket)}'
    elif op_id == Where_opt.BETWEEN and val2:
        text += f'{WORD_DICT[Where_opt.BETWEEN.name.lower()]} {val1} and {val2}'
    else:
        key = Where_opt(op_id).name.lower()
        text += f'{WORD_DICT[key]} '
        if isinstance(val1, dict):
            text += f'{subq2text(val1, schema, group_semantics, outer_col=col, bracket=bracket)}'
        elif isinstance(val1, tuple):
            text += f'{get_column_name(val1[1].strip("_"), with_table=with_table, bracket=bracket)}'
        else: text += f'{val1}'
    
    text = ' '.join(text.split())
    return text

def order2text(order: tuple, keyword: str, limit: int, group_semantics: dict, bracket=False):
    text = ""
    agg_id, col_id, _ = order[1]
    if limit:
        # Superlative semantics
        if limit == 1:
            key = f'{keyword}_limit_1'
            text += f"is {WORD_DICT[key]} "
        # `Top k` semantics
        else:
            key = f'{keyword}_limit_n'
            text += f'is {WORD_DICT[key]} {limit}'
        text += f' {WORD_DICT[Agg_opt(agg_id).name.lower()]} {get_column_name(col_id, bracket=bracket) if col_id != "__all__" else group_semantics["global"]}' 
    else:
        text += f'{WORD_DICT[keyword.lower()]} {get_column_name(col_id, bracket=bracket)}'
    
    text = ' '.join(text.split())
    return text

def clause2dialect(
    p_sql: dict, schema: dict, group_semantics: dict, cls_type: str, not_used_tables: list = None, prefix: str = '', bracket = False
) -> str:
    """Translate a specific SQL clause into corresponding dialect expression.

    Args:
        p_sql: The parsed sql structure
        group_semantics: The semantics mapping used for GROUP-related operations
        cls_type: SQL clause type (mainly for `where`, `having`, `order`)
        prefix: Prefix string added on the dialect
        bracket: Add brackets surrounding column string

    Return:
        The dialect text expression for the SQL clause
    """   
    dialect = ""
    # if not p_sql[cls_type]: return dialect
    
    if cls_type == 'where' and p_sql[cls_type]:
        conjs = [''] + [f' {c} ' for c in p_sql[cls_type][1::2]]
        conds = [c for c in p_sql[cls_type][::2]]
        for cond, conj in zip(conds, conjs):
            dialect += conj
            dialect += cond2text(cond, schema, group_semantics, not_used_tables, bracket=bracket)
    elif cls_type == 'orderBy' and p_sql[cls_type]:
        keyword, orders = p_sql[cls_type]
        agg_id, col_id, _ = orders[0][1]
        dialect += f'{WORD_DICT[keyword.lower()]} '
        dialect += f'{get_column_name(col_id, bracket=bracket)}' if agg_id == Agg_opt.NONE else \
            f'{WORD_DICT[Agg_opt(agg_id).name.lower()]} {get_column_name(col_id, bracket=bracket) if col_id != "__all__" else group_semantics["global"]}'
        if p_sql['limit']:
            limit = p_sql['limit']
            key = f'{keyword}_limit_{"1" if limit == 1 else "n"}'
            dialect += f' with a limit {limit} ({WORD_DICT[key]} {"" if limit == 1 else limit}) returned'
    elif cls_type == 'having' and p_sql[cls_type]:
        dialect += cond2text(p_sql[cls_type][0], schema, group_semantics, not_used_tables, bracket=bracket)
    # if IUE exists, add related conditions
    if cls_type in ['where', 'having']:
        for type in SQL_OPS:
            if not p_sql[type]: continue
            conjs = [''] + [f' {c} ' for c in p_sql[type][cls_type][1::2]]
            conds = [c for c in p_sql[type][cls_type][::2]]
            for cond, conj in zip(conds, conjs):
                if not conj: 
                    if type == 'intersect': dialect += ' or '
                    elif type == 'union': dialect += ' and '
                    else: dialect += ' excepting '
                dialect += conj
                dialect += cond2text(cond, schema, group_semantics, not_used_tables, bracket=bracket)
                
    dialect = ' '.join(dialect.split()) # Remove duplicate spaces
    return prefix + dialect if dialect else dialect