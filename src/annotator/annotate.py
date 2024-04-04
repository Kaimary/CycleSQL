
import re
from typing import Any
from enum import IntEnum
from prettytable import PrettyTable
from colorama import Fore, Style

from src.word_dict import WORD_DICT
from src.utils import EXCLUDE_KEYWORDS, Where_opt, cond2text, iue2text, order2text, get_column_name, \
    extract_parsed_cond_structure, extract_parsed_group_structure, extract_parsed_order_limit_structure, extract_parsed_select_structure


COND_OPS = ['=', 'in']

def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
class Annotation_type(IntEnum):
    RESULT_ASTERISK = 0
    RESULT_AGG = 1
    ROW_COUNT = 2
    EMPTY_PROVENANCE = 3
    WHERE = 4
    GROUP = 5
    ORDER  = 6
    HAVING = 7
    INTERSECT = 8
    UNION = 9
    EXCEPT = 10
    
class XQL:
    def __init__(self):
        self._table: XTable = None
        self._conditions: list[XCondition] = []
        self.p = 0
        self.field2cond = {}
    
    @property
    def table(self): 
        return self._table
    
    @property
    def conditions(self): 
        # We sort the array before returning the conditions
        # 1. Prioritize primary-key-related conditions
        # 2. De-prioritize conditions with [ANNO] annotation
        self._conditions.sort(key=lambda x: (not x.primary, any(a._content == 2 for a in x.annotations)))
        return self._conditions
    
    def add_condition(
        self, 
        field: str, 
        values: list, 
        schema: dict
    ):
        t, c = field.split('.')
        self._conditions.append(XCondition(field, values, schema['col2type'][field], c in schema['primaries'][t]))
        self.field2cond[field] = self.p
        self.p += 1

    def add_table(
        self, 
        table_units: list
    ):
        tables = [table_unit[1].strip('_') for table_unit in table_units]
        self._table = XTable(tables)

    def add_annotation(
        self, 
        parsed_rewrite_sql: dict,
        result_info: PrettyTable, 
        provenace: PrettyTable
    ):  
        row_count = len(provenace.rows)
        field = result_info.field_names[0]
        # If explaining an arithmatical result, add result information to the corresponding database fact
        if any(kw in field.lower() for kw in EXCLUDE_KEYWORDS):
            key = re.findall(r'\((.*?)\)', field)[0]
            value = result_info.rows[0][0]
            content = [field, value]
            if key == '*':
                self._table.add_annotation(Annotation(content, type=Annotation_type.RESULT_ASTERISK))
            else:
                for c in self._conditions:
                    if c.key == key: c.add_annotation(Annotation(content, type=Annotation_type.RESULT_AGG))
        # If the total returned rows greater than 1, or it is `group`-related query, add RowCount annotation. 
        if row_count > 1 or (parsed_rewrite_sql['groupBy'] and len(result_info.field_names) > 1):
            # Add RowCount annotation to table
            self._table.add_annotation(Annotation(content=row_count, type=Annotation_type.ROW_COUNT))
        # # Add grouping-related annotations to the table
        # for g in parsed_rewrite_sql['groupBy']:
        #     self._table.add_annotation(Annotation(content=g, type='group'))
        # Add predicate-related annotations to the conditions
        parsed_rewrite_sql['where'].reverse()
        bfind = False
        for w in parsed_rewrite_sql['where'][::2]:
            _, op_id, key, val, _ = extract_parsed_cond_structure(w)
            for c in self._conditions:
                if c.key == key:
                    # If `empty` provenance, add special annotation
                    if not c.values: c.add_annotation(Annotation(content=w, type=Annotation_type.WHERE, extra='*empty-provenance*'))
                    else:
                        # Extra check if the value in condition satisfy the values in provenance, 
                        # since the value in `OR` condition may not in provenance. 
                        extra1 = None
                        # Only check if `number`-type values
                        if (isinstance(val, int) or isinstance(val, float)) and all(is_num(v) for v in c.values):
                            values = [float(v) if '.' in v else int(v) for v in c.values] if isinstance(c.values[0], str) else c.values
                            if (op_id == Where_opt.EQ and all(v != val for v in values)) or \
                                (op_id == Where_opt.GT and all(v <= val for v in values)) or \
                                    (op_id == Where_opt.LT and all(v >= val for v in values)) or \
                                        (op_id == Where_opt.GTE and all(v < val for v in values)) or \
                                            (op_id == Where_opt.LTE and all(v > val for v in values)): 
                                                extra1 = val
                        # Annotation = predicate (result-related predicate?)
                        if op_id == Where_opt.EQ:
                            if parsed_rewrite_sql['groupBy'] and key == parsed_rewrite_sql['groupBy'][0][1].strip('_'):
                                c.add_annotation(Annotation(content=op_id, type=Annotation_type.WHERE, extra='*group-related*'))
                            elif not bfind:
                                c.add_annotation(Annotation(content=op_id, type=Annotation_type.WHERE, extra='*result-related*', extra1 = extra1))
                                bfind = True
                        else:
                            c.add_annotation(Annotation(content=w, type=Annotation_type.WHERE, extra1 = extra1))
        # Add ordering-related (if exists) annotations to the conditions/table 
        if parsed_rewrite_sql['orderBy'] and parsed_rewrite_sql['limit']:
            asc, _, key, limit = extract_parsed_order_limit_structure(parsed_rewrite_sql['orderBy'], parsed_rewrite_sql['limit'])
            if key == 'all':
                self._table.add_annotation(Annotation(content=[parsed_rewrite_sql['orderBy'][1][0], asc, limit], type=Annotation_type.ORDER))
            else:
                for c in self._conditions:
                    if c.key == key: c.add_annotation(Annotation(content=[parsed_rewrite_sql['orderBy'][1][0], asc, limit], type=Annotation_type.ORDER))
        # Add having-related (if exists) annotations to the conditions/table 
        for h in parsed_rewrite_sql['having'][::2]:
            _, op_id, key, _, _ = extract_parsed_cond_structure(h)
            if key == 'all': self._table.add_annotation(Annotation(content=h, type=Annotation_type.HAVING))
            else:
                for c in self._conditions:
                    if c.key == key: c.add_annotation(Annotation(content=h, type=Annotation_type.HAVING))
        
        # # TODO Only consider the first select-column
        # _, outer_col = extract_parsed_select_structure(parsed_original_sql['select'])
        # if parsed_rewrite_sql['intersect']:
        #     self._table.add_annotation(Annotation(content=[parsed_rewrite_sql['intersect'], outer_col], type=Annotation_type.INTERSECT))
        # if parsed_rewrite_sql['union']:
        #     self._table.add_annotation(Annotation(content=[parsed_rewrite_sql['union'], outer_col], type=Annotation_type.UNION))
        # if parsed_rewrite_sql['except']:
        #     self._table.add_annotation(Annotation(content=[parsed_rewrite_sql['except'], outer_col], type=Annotation_type.EXCEPT))
            
    def get_string(
        self, 
        group_semantics: dict, 
        join_semantics: dict,
        bracket: bool =False
    ):
        self._table.get_string(group_semantics, join_semantics, bracket=bracket)
        for cond in self._conditions:
            cond.get_string(group_semantics, bracket=bracket)

class Annotation:
    def __init__(
        self, 
        content: Any, 
        type: str, 
        extra=None, 
        extra1=None
    ):
        self._content = content
        self._type = type
        self._extra = extra
        self._extra1 = extra1
    
    @property
    def type(self):
        return self._type
    
    @property
    def content(self):
        return self._content
    
    def get_string(
        self,
        schema: dict,
        group_semantics: dict=None, 
        join_semantics: dict=None,
        ignore_col: bool=True,
        bracket: bool=False
    ): # sems: dict for join semantics
        text = 'that '
        if self._type == Annotation_type.WHERE:
            if isinstance(self._content, tuple):
                text += cond2text(self._content, schema, group_semantics, ignore_col=ignore_col, bracket=bracket)
            # Special token for semantic equivalence beteween annotation and predicate
            # e.g., predicate: paragraph_text = 'Brazil' (annotation: paragraph_text = 'Brazil')
            else: text = ''
        elif self._type == Annotation_type.GROUP:
            col = extract_parsed_group_structure([self._content])
            text += f'for each {get_column_name(col)}'
        elif self._type == Annotation_type.HAVING:
            text += cond2text(self._content, schema, group_semantics, ignore_col=ignore_col, bracket=bracket)
        elif self._type == Annotation_type.ORDER:
            order, asc, limit = self._content
            text += order2text(order, asc, limit, group_semantics=group_semantics)
        elif self._type == Annotation_type.ROW_COUNT:
            text = f'there are {self._content} {join_semantics}'
        elif self._type == Annotation_type.RESULT_AGG:
            agg = self._content[0].split('(')[0]
            field = re.findall(r'\((.*?)\)', self._content[0])[0]
            value = self._content[1]
            text = f'{WORD_DICT[agg.lower()]} {get_column_name(field)} is '+ (f'{value:.2f}' if isinstance(value, float) else f'{value}')
        elif self._type == Annotation_type.RESULT_ASTERISK:
            agg = self._content[0].split('(')[0]
            value = self._content[1]
            text = f'{WORD_DICT[agg.lower()]} {group_semantics["global"]} is {"only" if value == 1 else ""} ' + (f'{value:.2f}' if isinstance(value, float) else f'{value}')
        elif self._type in [Annotation_type.INTERSECT, Annotation_type.UNION, Annotation_type.EXCEPT]:
            p_iue_sql, outer_col = self._content
            _, iue_text = iue2text(p_iue_sql, schema, group_semantics, outer_col, text='', col=None, bracket=bracket)
            if self._type == Annotation_type.INTERSECT: text = 'and '
            elif self._type == Annotation_type.UNION: text = 'or '
            else: text = 'except '
            text += iue_text
            
        text = ' '.join(text.split())
        return text
    
class XTable:
    def __init__(self, tables):
        self._tables = tables
        self._annotations: list(Annotation) = []
    
    @property
    def tables(self):
        return self._tables
    
    @property
    def annotation(self):
        return self._annotations
    
    def get_string(self, group_semantics, join_semantics, bracket=False):
        print(f"{Fore.BLUE}table name: {self._tables} -> annotations:{Style.RESET_ALL}", end="")
        if not self._annotations: print(f"{Fore.RED}None{Style.RESET_ALL}", end = "")
        for ann in self._annotations:
            print(f"{Fore.RED}{ann.get_string(group_semantics, join_semantics, bracket)}{Style.RESET_ALL}", end = ";;;")
            
    def add_annotation(self, annotation: Annotation):
        self._annotations.append(annotation)

class XCondition:
    def __init__(self, field, values, type, isprimary = False):
        if len(values) == 1: self.op = COND_OPS[0]
        else: self.op = COND_OPS[1]
        self._field = field
        self._values = values
        self._type = type
        self._primary = isprimary
        self._annotations: list(Annotation) = []

    @property
    def key(self):
        return self._field
    
    @property
    def values(self):
        return self._values
    
    @property
    def primary(self):
        return self._primary

    @property
    def annotations(self):
        return self._annotations
    
    def add_annotation(self, annotation: Annotation):
        self._annotations.append(annotation)

    def get_string(self, group_semantics, bracket=False):
        print(f"{Fore.BLUE}content: {self._field} {self.op} {self._values} (primary: {self._primary}) -> annotations:{Style.RESET_ALL}", end = "")
        if not self._annotations: print(f"{Fore.RED}None{Style.RESET_ALL}", end = "")
        for ann in self._annotations:
            print(f"{Fore.RED}{ann.get_string(group_semantics, bracket)}{Style.RESET_ALL}", end = ";;;")

class Annotator:
    # Annotate the SQL query
    # Obtains the database facts
    @staticmethod
    def prov2nl(provenace: PrettyTable, schema: dict):
        xnl = XQL()
        for idx, field in enumerate(provenace.field_names):
            values = [row[idx] for row in provenace.rows]
            xnl.add_condition(field, values, schema)
        return xnl

    def annotate(
        self, 
        parsed_rewrite_sql: dict, 
        provenace: PrettyTable, 
        result_info: PrettyTable,
        schema: dict, 
        keep: dict
    ) -> XQL:
        def __restore_rewrite_sql__(sql, keep):
            if not keep: return sql
            if 'group' in keep.keys():
                sql['groupBy'] = keep['group']['groupBy']
                sql['having'] = keep['group']['having']
                sql['orderBy'] = keep['group']['orderBy']
                sql['limit'] = keep['group']['limit']
            return sql
        
        xnl = self.prov2nl(provenace, schema)
        # We remove some clauses from original SQL for execution purpose (to make sure the result provenance is correct);
        # Hence, restore back to the original one for annotation purpose.
        parsed_rewrite_sql = __restore_rewrite_sql__(parsed_rewrite_sql, keep)
        xnl.add_table(parsed_rewrite_sql['from']['table_units'])
        xnl.add_annotation(parsed_rewrite_sql, result_info, provenace)

        return xnl