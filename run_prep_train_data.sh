#!/bin/bash

train_data='data/spider/train_spider.json'
train_data2='data/spider/train_spider_6984.json'
table_data='data/spider/tables.json'
db_dir='data/spider/database'
beam_output_dir='data/beam_outputs/raw/train'
output_dir='output/beam_outputs/train'

if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

python -m scripts.process_raw_train_beam_outputs --train_file_path ${train_data} --train_file_path2 ${train_data2} \
    --table_file_path ${table_data} --db_dir ${db_dir} --beam_output_dir ${beam_output_dir} --output_dir ${output_dir}