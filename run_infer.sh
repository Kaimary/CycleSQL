#!/bin/bash
[ -z "$1" ] && echo "First argument is the dataset name." && exit 1
DATASET_NAME="$1"

[ -z "$2" ] && echo "Second argument is the NL2SQL model name." && exit 1
MODEL_NAME="$2"

[ -z "$3" ] && echo "Third argument is the test JSON file." && exit 1
TEST_FILE="$3"

[ -z "$4" ] && echo "Fourth argument is the raw beam output txt file" && exit 1
RAW_BEAM_OUTPUT_FILE="$4"

[ -z "$5" ] && echo "Fifth argument is the datset table schema file." && exit 1
TABLES_FILE="$5"

[ -z "$6" ] && echo "Sixth argument is the directory of the databases of the dataset." && exit 1
DB_DIR="$6"

[ -z "$7" ] && echo "Seventh argument is the directory of the test suite databases of the dataset (optional)." && exit 1
TS_DB_DIR="$7"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "===================================================================================================================================="
echo "INFO     ****** CycleSQL Inference Pipeline Start ******"

# Define some variables
OUTPUT_DIR=outputs/$DATASET_NAME/$MODEL_NAME
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
NLI_MODEL_DIR=saved_models/checkpoint-500
case $MODEL_NAME in
    "chatgpt")
    BEAM_SIZE=5
    ;;
    "gpt-4")
    BEAM_SIZE=5
    ;;
    "chess")
    BEAM_SIZE=5
    ;;
    "dailsql")
    BEAM_SIZE=8
    ;;
    "smbop")
    BEAM_SIZE=8
    ;;
    "picard")
    BEAM_SIZE=8
    ;;
    "resdsql")
    BEAM_SIZE=6
    ;;
    "resdsql-3b")
    BEAM_SIZE=8
    ;;
    *)
    echo "unknown NL2SQL model!"
    exit;
    ;;
esac

OUTPUT_FILE=$OUTPUT_DIR/preds.txt
if [ ! -f $OUTPUT_FILE ]; then
python -m scripts.run_infer --model_name $MODEL_NAME --beam_size $BEAM_SIZE --test_file $TEST_FILE \
    --beam_output_file $RAW_BEAM_OUTPUT_FILE  --nli_model_dir $NLI_MODEL_DIR \
    --table_file_path $TABLES_FILE --db_dir $DB_DIR  --output_file_path $OUTPUT_FILE || exit $?
else
    echo "WARNING     \`$OUTPUT_FILE\` already exists."
fi

# Final Evaluation
EVALUATE_OUTPUT_FILE=$OUTPUT_DIR/eval_result.txt
python -m spider_utils.eval.evaluation --gold $TEST_FILE --pred "$OUTPUT_FILE" \
--db "$DB_DIR" --ts_db "$TS_DB_DIR" > "$EVALUATE_OUTPUT_FILE"
echo "Spider evaluation complete! Results are saved in \`$EVALUATE_OUTPUT_FILE\`"
echo "===================================================================================================================================="