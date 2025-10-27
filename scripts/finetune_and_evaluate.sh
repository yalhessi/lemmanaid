# BASE_MODEL="meta-llama/Llama-3.2-1B"
# MODEL_LABEL="Llama-3.2-1B"
# CONFIGS_FILE="configs/multi_gpu_llama.yaml"
BASE_MODEL="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_LABEL="deepseek-coder-1.3b-base"
CONFIGS_FILE="configs/multi_gpu_yousef.yaml"

DATASET="yalhessi/lemexp-task1-v3"

INPUT_CONFIG=""
# INPUT_CONFIG="_nodefs"
# INPUT_CONFIG="_notypes"

OUTPUT_CONFIG="template"
# OUTPUT_CONFIG="lemma_object"

# TRAINING_SET="small"
TRAINING_SET="full"

TRAIN_CONFIG="$OUTPUT_CONFIG"_"$TRAINING_SET$INPUT_CONFIG"
echo "Training config: $TRAIN_CONFIG"

learning_rate=8e-4
epochs=12
LABEL="8lr-12epochs-normal-eos"
# LABEL="8lr-12epochs-no-eos"

MODEL_NAME="$DATASET-$TRAIN_CONFIG-$MODEL_LABEL-$LABEL"

torchrun --nproc_per_node=8 lemma_exploration/parallel_finetune.py --model-name="$BASE_MODEL" --dataset-name="$DATASET" --dataset-config="$TRAIN_CONFIG"  --label="$LABEL" --training-args num_train_epochs=$epochs eval_steps=1/60 learning_rate=$learning_rate  ;

EVAL_CONFIG="$OUTPUT_CONFIG"_"small$INPUT_CONFIG"
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=greedy --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=beam-search --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;

EVAL_CONFIG="$OUTPUT_CONFIG"_"octonions$INPUT_CONFIG"
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=greedy --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=beam-search --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;

EVAL_CONFIG="$OUTPUT_CONFIG"_"afp$INPUT_CONFIG"
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=greedy --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=beam-search --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;

# # # ==============================================================================
# # # ==============================================================================
# # # ==============================================================================

# BASE_MODEL="meta-llama/Llama-3.2-1B"
# MODEL_LABEL="Llama-3.2-1B"
# CONFIGS_FILE="configs/multi_gpu_llama.yaml"
BASE_MODEL="deepseek-ai/deepseek-coder-1.3b-base"
MODEL_LABEL="deepseek-coder-1.3b-base"
CONFIGS_FILE="configs/multi_gpu_yousef.yaml"

DATASET="yalhessi/lemexp-task1-v3"

# INPUT_CONFIG=""
INPUT_CONFIG="_nodefs"
# INPUT_CONFIG="_notypes"

OUTPUT_CONFIG="template"
# OUTPUT_CONFIG="lemma_object"

# TRAINING_SET="small"
TRAINING_SET="full"

TRAIN_CONFIG="$OUTPUT_CONFIG"_"$TRAINING_SET$INPUT_CONFIG"
echo "Training config: $TRAIN_CONFIG"

learning_rate=8e-4
epochs=12
LABEL="8lr-12epochs-normal-eos"
# LABEL="8lr-12epochs-no-eos"

MODEL_NAME="$DATASET-$TRAIN_CONFIG-$MODEL_LABEL-$LABEL"

torchrun --nproc_per_node=8 lemma_exploration/parallel_finetune.py --model-name="$BASE_MODEL" --dataset-name="$DATASET" --dataset-config="$TRAIN_CONFIG"  --label="$LABEL" --training-args num_train_epochs=$epochs eval_steps=1/60 learning_rate=$learning_rate  ;

EVAL_CONFIG="$OUTPUT_CONFIG"_"small$INPUT_CONFIG"
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=greedy --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=beam-search --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;

EVAL_CONFIG="$OUTPUT_CONFIG"_"octonions$INPUT_CONFIG"
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=greedy --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=beam-search --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;

EVAL_CONFIG="$OUTPUT_CONFIG"_"afp$INPUT_CONFIG"
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=greedy --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;
accelerate launch --config_file=$CONFIGS_FILE lemma_exploration/parallel_predictions.py --generation=beam-search --model-name="$MODEL_NAME" --base-model-name=$BASE_MODEL --dataset-name="$DATASET" --dataset-config="$EVAL_CONFIG" ;
