import json

train_file_7000 = "data/spider/train_spider.json"
train_file_6984 = "data/spider/train_spider_6984.json"
smbop_train_beam_outputs = "data/beam_outputs/raw/train/smbop.train.beam8.txt"
lines = [l.strip() for l in open(smbop_train_beam_outputs).readlines()]

train = json.load(open(train_file_7000))
train2 = json.load(open(train_file_6984))

questions = [ex['question'] for ex in train2]

missing_cnt = 0
outputs = []
idx = 0
for ex in train:
    if ex['question'] in questions:
        beams = lines[idx*8: (idx+1)*8]
        idx += 1
    else:
        beams = ['sql'] * 8
    outputs.extend(beams)

    if ex['question'] not in questions: missing_cnt += 1

with open("output.txt", "w") as f:
    for o in outputs:
        f.write(f"{o}\n")