from src.annotator.annotate import XQL, Annotation_type
from src.utils import SAMPLING_THRESHOLD, Query_type, get_column_name, append_blank_space

class Translator:
    def set_context(
        self, 
        question: str, 
        schema: dict, 
        group_semantics: dict, 
        join_semantics: dict
    ):
        self._question = question
        self._schema = schema
        self._group_semantics = group_semantics
        self._join_semantics = join_semantics

    def translate(
        self, 
        xql: XQL,
        type_: str,
        is_main_no_provenance: bool,
        bracket: bool = False
    ):
        xnl, modifier = '', ''
        no_subject = False
        description: list[tuple] = []

        t = xql.table
        # Table name as the subject of the explanation
        priority = 10
        subject = '' if 'with' in self._join_semantics else self._join_semantics
        for a in t.annotation:
            # Check the table annotations that are `result`-related, de-prioritize to 0 level (Put last)
            if a.type in [Annotation_type.RESULT_ASTERISK, Annotation_type.RESULT_AGG]:
                phrase = f'so {a.get_string(self._schema, self._group_semantics, self._join_semantics, bracket=bracket)}.'
                description.append((phrase, 0))
            # Check the table annotations that are not `result`-related, prioritize to 10 level (Put ahead)
            # And clear `subject` as the modifier exists
            else:
                subject = ''
                no_subject = True
                ignore_col = False if a._type == Annotation_type.HAVING else True
                modifier += a.get_string(self._schema, self._group_semantics, self._join_semantics, ignore_col=ignore_col, bracket=bracket)
                if type_ not in [Query_type.INTERSECT, Query_type.UNION] or a._type == Annotation_type.HAVING: priority = 4 # explain after `*result-related*` phrase
        if subject and modifier: subject = append_blank_space(subject)
        if subject or modifier: description.append((subject + modifier, priority))

        # with_table = True if not subject else False
        for cond in xql.conditions:
            priority1 = 1
            field = cond.key
            phrase = f'{get_column_name(field)} is '
            values = [str(v) for v in set(cond.values)]
            if not values or len(values) > SAMPLING_THRESHOLD: phrase = ''
            # Combine same values
            elif (values.count(values[0]) == len(values)): phrase += f'{values[0]} '
            else:
                s_values = ', '.join(values)
                phrase += s_values
                phrase = append_blank_space(phrase)
            for a in cond.annotations:
                # Prioritize result-related condition so that mention in the explanation ahead.
                if a._extra == '*result-related*':
                    priority1 = 100 if no_subject else 5
                    phrase = phrase.replace('is ', '')
                    # if `IU`query, remove `subject`-related annotation to avoid duplication.
                    if type_ in [Query_type.INTERSECT, Query_type.UNION] and not is_main_no_provenance: phrase = ''
                elif a._extra == '*group-related*':
                    priority1 = 11
                    phrase = 'for ' + phrase
                    phrase = phrase.replace('is ', '')
                elif a._type == Annotation_type.RESULT_AGG: priority1 -= 0.5
                ignore_col = True if phrase else False
                phrase += a.get_string(self._schema, self._group_semantics, self._join_semantics, ignore_col=ignore_col, bracket=bracket)
                if a._extra == '*empty-provenance*' and type_ not in [Query_type.INTERSECT, Query_type.UNION, Query_type.EXCEPT]:
                    priority1 = 11
                    # Post-edit phrase if no provenance information 1. remove `that` 2. negate the subject
                    phrase = phrase.replace('that ', '')
                    if (subject + modifier, priority) in description: description.remove((subject + modifier, priority))
                    else: subject = self._join_semantics
                    subject = 'has no ' + subject
                    description.append((subject + modifier, priority))
                if a._extra1: phrase += f' (or {a._extra1})'
            if cond.primary: priority1 += 1
            if phrase: description.append((phrase.strip(), priority1))
        # Sort the phrases with priorities, and hence get a `quasi-question-explanation`
        description = sorted(description, key=lambda x: len(x[0]))
        description = sorted(description, key=lambda x: x[1], reverse=True)
        xnl += ', '.join([d[0] for d in description])
        
        return xnl