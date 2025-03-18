import os, json, click
from tqdm import tqdm
from collections import defaultdict

from spider_utils.evaluation import build_foreign_key_map_from_json
from spider_utils.utils import check_valid, find_exec_match
    
# @click.command()
# @click.argument("test_file", type=click.Path(exists=True, dir_okay=False))
# @click.argument("model_name", type=click.Path(exists=True, dir_okay=True))
# @click.argument("beam_output_file", type=click.Path(exists=True, dir_okay=False))
# @click.argument("beam_size", default=5)
# @click.argument("nli_model_dir", type=click.Path(exists=True, dir_okay=True))
# @click.argument("tables_file", type=click.Path(exists=True, dir_okay=False))
# @click.argument("db_dir", type=click.Path(exists=True, dir_okay=True))
# @click.argument("output_dir", type=click.Path(exists=False, dir_okay=True))
# @click.argument("enable_logging", default=False)
def main(model_name, test_file, nli_output_file, nli_prob_output_file, beam_output_file, raw_beam_output_file, beam_size, tables_file, db_dir, output_file):
    """
    Entry point

    :param test_file: the testing json file that includes natural language queries and ground-truth sqls (optional)
    :param model_name: the nl2sql model name (currently support `chatgpt`, `picard`, `resdsql`, `smbop`)
    :param beam_output_file: the corresopoding predictions (pure text) from the default beams from the model
    :param beam_size: the default beam size
    :param nli_model_dir: the saved nli model directory
    :param table_file: the table json file that describes the table schema information
    :param db_dir: the directory of sqlite database files
    :param output_dir: output directory
    :return: outputs from nlXidb
    """
    data = json.load(open(test_file))
    nlis = [l.strip() for l in open(nli_output_file).readlines()]
    nli_probs = [l.strip('[] \n').split() for l in open(nli_prob_output_file).readlines()]
    beam_preds = [l.strip(';\n ') if 'resdsql' in model_name else l.split('\t')[0].strip() \
                  for l in open(raw_beam_output_file).readlines()]
    if 'resdsql' in model_name:
        if model_name == 'resdsql': beam_preds = [l for idx, l in enumerate(beam_preds) if idx % 9 != 0]
        beam_preds = [l.split('|')[1].strip() if '|' in l else l for l in beam_preds]

    beam_outputs = json.load(open(beam_output_file))
    assert len(nlis) == len(beam_outputs)

    nl2preds, nl2entailment, nl2probs, nl2beam, nl2exp = \
        defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for ex, nli, probs in zip(beam_outputs, nlis, nli_probs):
        nl2preds[ex['question']].append(ex['query'])
        nl2entailment[ex['question']].append(nli)
        nl2probs[ex['question']].append(probs)
        nl2beam[ex['question']].append(ex['beam'])
        nl2exp[ex['question']].append(ex['explanation'])

    hit = 0 # accuracy indicator for CycleSQL
    beam_k_hit_succeed, beam_1_hit_but_miss, beam_k_hit_but_entailment_fail = 0, 0, 0
    beam_1_hit_cnt, beam_k_hit_cnt, beam_k_hit_but_early_stop = 0, 0, 0
    preds, originals = [], []
    kmaps = build_foreign_key_map_from_json(tables_file)
    for i, ex in tqdm(enumerate(data), total = len(data), desc ="Processing on Explaining"):
        db_id = ex['db_id']
        nl = ex['question']
        gold = ex['query'].strip(';')
        beam1 = beam_preds[i*beam_size]
        originals.append(beam1)
        all_beams = beam_preds[i * beam_size: (i + 1) * beam_size]
        db_file = os.path.join(db_dir, db_id, f'{db_id}.sqlite')
        valids = check_valid(all_beams, db_file, kmaps[db_id])
        global_matches = find_exec_match(gold, all_beams, db_file, kmaps[db_id])
        if global_matches:
            beam_k_hit_cnt += 1
            if 0 in global_matches: beam_1_hit_cnt += 1
        # If all the beams are either invalid or not able to generate explanation, 
        # use the first beam output as the prediction directly.
        if nl not in nl2preds.keys():
            preds.append(beam1)
            if global_matches and 0 in global_matches: hit += 1
            continue
        
        beam_nums = nl2beam[nl]
        entaiments, exps, probs = [], [], []
        for beam in range(1, beam_size + 1):
            if beam not in beam_nums:
                # If no explanation for beam 1 output, consider it as correct output; Otherwise, consider as incorrect.
                if beam == 1: entaiments.append('entailment')
                else: 
                    entaiments.append('contradiction')
                probs.append([])
                exps.append('')
            else: 
                entaiments.append(nl2entailment[nl][beam_nums.index(beam)])
                probs.append(nl2probs[nl][beam_nums.index(beam)])
                exps.append(nl2exp[nl][beam_nums.index(beam)])
        beam_nums = [i for i in range(1, beam_size + 1)]
        
        entailment, contradiction = False, False
        for j, (pred, valid, ent, prob, order, exp) in enumerate(zip(all_beams, valids, entaiments, probs, beam_nums, exps)):
            if j == 0 and ent == 'contradiction' and float(prob[1]) - float(prob[0]) < 0:
                preds.append(pred)
                entailment = True
                if j in global_matches:
                    hit += 1
                    # if eval_hardness(pred, db_file, kmaps[db_id]) in ['extra']: p+= 1
                elif global_matches:
                    if entaiments[global_matches[0]] == 'contradiction': 
                        beam_k_hit_but_entailment_fail += 1
                        # print('=' * 100)
                        # print(f"NL: {nl}\nbeam: {global_matches[0]}\n")
                        # print(f"beam: {all_beams[global_matches[0]]}, entailment: {entaiments[global_matches[0]]}, prob: {probs[global_matches[0]]}")
                    else: 
                        beam_k_hit_but_early_stop += 1
                        # print('=' * 100)
                        # print(f"NL: {nl}\ngold: {gold}\nhit: {global_matches[0]}")
                        # for b, e, p in zip(all_beams[:global_matches[0]+1], entailments[:global_matches[0]+1], probs[:global_matches[0]+1]):
                        #     print(f"beam: {b}, entaiment: {e}, prob: {p}")
                    # if eval_hardness(all_beams[global_matches[-1]], db_file, kmaps[db_id]) in ['extra']: n += 1
                break
                # if 0 in global_matches:
            # Use as prediction for first-match `entailment`
            elif ent == 'entailment' and valid and not entailment:
                entailment = True
                preds.append(pred)
                if j in global_matches:
                    hit += 1
                    if j > 0 and all(o not in global_matches for o in range(j)): 
                        beam_k_hit_succeed += 1
                        # print('='*100)
                        # print(f"NL: {nl}\nhit: {j}\ngold: {gold}\n")
                        # for b, e, p in zip(all_beams[:j+1], entaiments[:j+1], probs[:j+1]):
                        #     print(f"beam: {b}, entaiment: {e}, prob: {p}")
                        break
                elif 0 in global_matches:
                    beam_1_hit_but_miss += 1
                    # print('='*100)
                    # print(f"NL: {nl}\nhit: {j}\ngold: {gold}\nexplanation: {exps[0]}\n entaiment:{entaiments[0]}, prob: {probs[0]}")
            elif ent == 'entailment' and valid and not contradiction:
                if global_matches and 0 not in global_matches and j in global_matches:
                    contradiction = True
                    beam_k_hit_but_early_stop += 1
                    # print('=' * 100)
                    # print(f"NL: {nl}\ngold: {gold}\nhit: {j}")
                    # for b, e, p in zip(all_beams[:j+1], entaiments[:j+1], probs[:j+1]):
                    #     print(f"beam: {b}, entaiment: {e}, prob: {p}")
            elif ent == 'contradiction' and valid and not contradiction:
                if global_matches and 0 not in global_matches and j in global_matches:
                    contradiction = True
                    beam_k_hit_but_entailment_fail += 1
                    # print('=' * 100)
                    # print(f"NL: {nl}\nbeam: {j}\n")
                    # print(f"beam: {all_beams[j]}, entaiment: {entaiments[j]}, prob: {probs[j]}")

        if not entailment:
            preds.append(beam1)
            if 0 in global_matches: 
                hit += 1
    
    print(f"beam1_hit_but_miss: {beam_1_hit_but_miss}, beam_k_hit_succeed: {beam_k_hit_succeed}, beam_k_hit_but_earlystop: {beam_k_hit_but_early_stop}, beam_k_hit_but_entail_fail: {beam_k_hit_but_entailment_fail}")
    acc = hit/len(data)
    before_acc = beam_1_hit_cnt/len(data)
    beam_acc = beam_k_hit_cnt/len(data)
    print(f"{model_name}+CycleSQL acc: {acc:.3f} ({hit}/{len(data)})")
    print(f"{model_name} acc: {before_acc:.3f} ({beam_1_hit_cnt}/{len(data)})")
    print(f"{model_name} beam acc: {beam_acc:.3f} ({beam_k_hit_cnt}/{len(data)})")

    with open(output_file, "w") as out:
        for pred in preds:
            out.write(f"{pred}\n")
    # output_file = output_file.replace('cycle', 'beam1')
    # with open(output_file, "w") as out:
    #     for pred in originals:
    #         out.write(f"{pred}\n")

if __name__ == "__main__":
    dataset = "spider"
    test_file = f"data/{dataset}/dev.json"
    model_name = "resdsql-3b"
    beam_size = 8
    checkpoint = 500
    nli_output_file = f"output/inference/{dataset}/{model_name}/{model_name}.predictions.ckpt={checkpoint}.txt"
    nli_prob_output_file = f"output/inference/{dataset}/{model_name}/{model_name}.probs.ckpt={checkpoint}.txt"
    beam_output_file = f"output/inference/{dataset}/{model_name}/{model_name}.dev.beam{beam_size}.json"
    raw_beam_output_file = f"data/beam_outputs/raw/{dataset}/dev/{model_name}.dev.beam{beam_size}.txt" 
    output_file = f"output/inference/{dataset}/{model_name}/{model_name}.dev.cycle.predictions.txt"
    tables_file = "data/spider/tables.json" 
    db_dir = "data/spider/database/" 
    main(model_name, test_file, nli_output_file, nli_prob_output_file, beam_output_file, raw_beam_output_file, beam_size, tables_file, db_dir, output_file)