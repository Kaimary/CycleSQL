# <span style="font-size:0.8em;">G</span>rounding Natural Language to SQL Translation with Data-Based Self-Explanations
> Improve NL2SQL with Natural Language Explanations as Self-provided Feeback
> 
The official repository contains the code and pre-trained models for our paper [Grounding Natural Language to SQL Translation with Data-Based Self-Explanations](https://arxiv.org/) (The paper will be public after acceptance😊).

<p align="center">
   <a href="https://github.com/kaimary/CycleSQL/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/kaimary/CycleSQL.svg?color=blue">
   </a>
   <a href="https://github.com/kaimary/CycleSQL/stargazers">
       <img alt="stars" src="https://img.shields.io/github/stars/kaimary/CycleSQL" />
  	</a>
  	<a href="https://github.com/kaimary/CycleSQL/network/members">
       <img alt="FORK" src="https://img.shields.io/github/forks/kaimary/CycleSQL?color=FF8000" />
  	</a>
    <a href="https://github.com/kaimary/CycleSQL/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/kaimary/CycleSQL?color=0088ff"/>
    </a>
    <br />
</p>

## 📖 Overview

This code implements:

* A plug-and-play iterative framework built upon <strong>self-provided feedback</strong> to enhance the translation accuracy of existing end-to-end models.

### 🚀 About CycleSQL
> **TL;DR:** We introduce CycleSQL --  a plug-and-play framework that enables flexible integration into existing end-to-end NL2SQL models.
> Inspired by the *feedback mechanisms* used in modern recommendation systems and *iterative refinement* methods introduced in LLMs, CycleSQL introduces data-grounded NL explanations of query
results as a form of internal feedback to create a self-contained feedback loop within the end-to-end translation process, facilitating iterative self-evaluation of translation correctness.

The objective of NL2SQL translation is to convert a natural language query into an SQL query. 

While significant advancements in enhancing overall translation accuracy, current end-to-end models face persistent challenges in producing desired quality output during their initial attempt, owing to the treatment of language translation as a "one-time deal".

To tackle the problem, Cyclesql introduces natural language explanations of query results as self-provided feedback and uses the feedback to validate the correctness of the translation iteratively, hence improving the overall translation accuracy. 

This is the approach used in the CycleSQL method.

### ❓ How it works

CycleSQL uses the following four steps to establish the feedback loop for the NL2SQL translation process:

1. **Provenance Tracking**: Track provenance of the to-explained query result to retrieve data-level information from the database.
2. **Semantics Enrichment**: Enhance the provenance by associating it with operation-level semantics derived from the translated SQL.
3. **Explanation Generation**: Generate a natural language explanation by interpreting the enriched provenance information.
4. **Translation Verification**: The generated NL explanation is utilized to verify the correctness of the underlying NL2SQL translation.
Iterating through the above steps until a validated correct translation is achieved.

This process is illustrated in the diagram below:

<div style="text-align: center">
<img src="assets/overview.png" width="800">
</div>


## ⚡️ Quick Start

### 🙇 Prerequisites
First, you should set up a Python environment. This code base has been tested under Python 3.8.

1. Install the required packages
```bash
pip install -r requirements.txt
```

2. Download the [Spider](https://yale-lily.github.io/spider) and the other three robustness variants ([Spider-Realistic](https://drive.google.com/file/d/19tsgBGAxpagULSl9r85IFKIZb4kyBGGu/view?usp=sharing),  [Spider-Sync](https://drive.google.com/file/d/19tsgBGAxpagULSl9r85IFKIZb4kyBGGu/view?usp=sharing), and [Spider-DK](https://drive.google.com/file/d/19tsgBGAxpagULSl9r85IFKIZb4kyBGGu/view?usp=sharing)), and put the data into the <strong>data</strong> folder. Unpack the datasets and create the following directory structure:
```
/data
├── database
│   └── ...
├── dev.json
├── dev_gold.sql
├── tables.json
├── train_gold.sql
├── train.json
└── train.json
```


## 🏋️‍♀️ Training

**📃 Natural Language Inference Model:**
We implemented the natural language inference model based on the T5-large model. We utilize various NL2SQL models (i.e., SmBoP, PICARD, RESDSQL, and ChatGPT) to generate the training data for the model training. You can use the following command to train the model from scratch:

```
$ python scripts/run_classification.py --model_name_or_path t5-large --shuffle_train_dataset --do_train --do_eval --num_train_epochs 5 --learning_rate 5e-6 --per_device_train_batch_size 8 --per_device_eval_batch_size 1 --evaluation_strategy steps --train_file data/nli/train.json  --validation_file data/nli/dev.json --output_dir tmp/ --load_best_model_at_end --save_total_limit 5
```

### 👍 Download the checkpoint

The natural language inference model checkpoint will be uploaded in the following link:

Model  | Download Model
----|----
`nli-classifier`  | [nli-classifier.tar.gz](https://drive.google.com/file/d/13oOEkAHwF7i0iiWgVBdcMBR8lijEuF4o/view?usp=share_link)

put the model checkpoint put the data into the <strong>saved_models</strong> folder.

## 👀 Inference
The evaluation script is located in the root directory `run_inference.sh`.
You can run it with:
```
$ bash run_infer.sh <dataset_name> <model_name> <test_file_path> <model_raw_beam_output_file_path> <table_path> <db_dir> <test_suite_db_dir>
```

The evaluation script will create the directory `outputs` in the current directory and generate the result outcomes.
