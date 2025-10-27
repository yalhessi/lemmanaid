#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = lemma-exploration
ENV_NAME = lemexp
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(ENV_NAME)"

## Activate Python interpreter environment
.PHONY: activate_environment
activate_environment:
	conda activate $(ENV_NAME)
	@echo ">>> conda env activated. Use:\nconda deactivate to deactivate it."



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Set up the project environment
.PHONY: setup
setup: create_environment requirements
	$(PYTHON_INTERPRETER) scripts/setup.py


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) lemma_exploration/dataset.py

BASE_MODEL="deepseek-ai/deepseek-coder-1.3b-base"
LABEL="deepseek-coder-1.3b-base-ddp-8lr-v2"
TASK_NAME=lemexp-task1-v2
FINETUNE_CONFIG="template_small_old_defs"
EVAL_CONFIG="template_afp"
## Make predictions
.PHONY: parallel-predictions
parallel-predictions: 
	make greedy-predictions;
	make beam-search-predictions
	
greedy-predictions:
	@echo ">>> Running greedy predictions for the following:"
	@echo ">>> base model: $(BASE_MODEL)"
	@echo ">>> task: $(TASK_NAME)"
	@echo ">>> finetuned on $(FINETUNE_CONFIG)"
	@echo ">>> evaluating on $(EVAL_CONFIG)"

	accelerate launch --config_file=configs/multi_gpu_yousef.yaml lemma_exploration/parallel_predictions.py --generation=greedy --model-name="yalhessi/${TASK_NAME}-${FINETUNE_CONFIG}-${LABEL}" --base-model-name=${BASE_MODEL} --dataset-name="yalhessi/${TASK_NAME}" --dataset-config="${EVAL_CONFIG}"

beam-search-predictions:
	@echo ">>> Running beam search predictions for the following:"
	@echo ">>> base model: $(BASE_MODEL)"
	@echo ">>> task: $(TASK_NAME)"
	@echo ">>> finetuned on $(FINETUNE_CONFIG)"
	@echo ">>> evaluating on $(EVAL_CONFIG)"

	accelerate launch --config_file=configs/multi_gpu_yousef.yaml lemma_exploration/parallel_predictions.py --generation=beam-search --model-name="yalhessi/${TASK_NAME}-${FINETUNE_CONFIG}-${LABEL}" --base-model-name=${BASE_MODEL} --dataset-name="yalhessi/${TASK_NAME}" --dataset-config="${EVAL_CONFIG}"

temperature-predictions:
	@echo ">>> Running temperature predictions for the following:"
	@echo ">>> base model: $(BASE_MODEL)"
	@echo ">>> task: $(TASK_NAME)"
	@echo ">>> finetuned on $(FINETUNE_CONFIG)"
	@echo ">>> evaluating on $(EVAL_CONFIG)"

	accelerate launch --config_file=configs/multi_gpu_yousef.yaml lemma_exploration/parallel_predictions.py --generation=temperature-0.5 --model-name="yalhessi/${TASK_NAME}-${FINETUNE_CONFIG}-${LABEL}" --base-model-name=${BASE_MODEL} --dataset-name="yalhessi/${TASK_NAME}" --dataset-config="${EVAL_CONFIG}"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
