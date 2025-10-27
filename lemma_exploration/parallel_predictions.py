import argparse
import time
from datetime import timedelta

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from lemma_exploration.config import OUTPUT_DIR
from lemma_exploration.dataset import EOS_TOKEN, OUTPUT_TOKEN, SEP_TOKEN, SPECIAL_TOKENS, TOKEN_LIST, TYPE_TOKEN
from lemma_exploration.eval import *
from lemma_exploration.utils import get_run_id


greedy_kwargs = {
    "batch_size": 4,
    "num_return_sequences": 1,
    "max_new_tokens": None,
}
# beam search, top 5 sequences
beam_search_top5_kwargs = {
    "num_beams": 4,
    "batch_size": 1,
    "num_return_sequences": 4,
    # "diversity_penalty": 0.5,
    # "low_memory": True,
    # "early_stopping": True,
}

generation_strategies = {
    "greedy": greedy_kwargs,
    "beam-search": beam_search_top5_kwargs,
} | {
    f"temperature-{i}": {
        "do_sample": True,
        "temperature": i,
        "num_return_sequences": 4,
    }
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="yalhessi/lemexp-processed-task1_min_symbols_template_small-deepseek-coder-1.3b-base",
        required=True,
    )
    parser.add_argument("--load-model-from", type=str, default="web-peft")
    parser.add_argument(
        "--base-model-name",
        type=str,
        default="deepseek-ai/deepseek-coder-1.3b-base",
        required=True,
    )
    parser.add_argument(
        "--dataset-name", type=str, default="yalhessi/lemexp-processed", required=True
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="task1_min_symbols_template_small",
        required=True,
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument(
        "--debug", type=bool, default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--upload", type=bool, default=False, action=argparse.BooleanOptionalAction
    )

    parser.add_argument(
        "--root-dir",
        type=str,
        default="/home/ubuntu/lemma-exploration/isabelle/afp-small",
    )
    parser.add_argument(
        "--imports",
        type=str,
        nargs="+",
        default=[
            "/home/ubuntu/lemma-exploration/thys/ExtractLemmas",
            "/home/ubuntu/lemma-exploration/thys/RoughSpec",
        ],
    )

    parser.add_argument("--generation", type=str, default="greedy")
    args = parser.parse_args()

    if not args.output_dir:
        run_id = get_run_id(args.model_name, args.dataset_name, args.dataset_config)
        args.output_dir = str(OUTPUT_DIR / run_id)

    return args


# batch, left pad (for inference), and tokenize
def prepare_prompts(prompts, tokenizer, batch_size=16):
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding="longest",
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False,
            ).to("cuda")
        )
    tokenizer.padding_side = "right"
    return batches_tok


if __name__ == "__main__":
    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=60 * 60 * 12))
        ]
    )

    configs = parse_args()
    accelerator.print(f"Configs: {configs}")

    # Prep the model
    tokenizer = AutoTokenizer.from_pretrained(
        configs.base_model_name, padding_side="left"
    )
    # tokenizer.add_special_tokens({"eos_token": EOS_TOKEN, "sep_token": SEP_TOKEN})
    # tokenizer.add_tokens([OUTPUT_TOKEN, TYPE_TOKEN])
    tokenizer.add_tokens([EOS_TOKEN])
    # tokenizer.add_special_tokens(SPECIAL_TOKENS)
    # tokenizer.add_tokens(TOKEN_LIST)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        configs.base_model_name,
        device_map={"": accelerator.process_index},
    )
    ### FOR MODELS TRAINED BEFORE REMOVING THE SPECIAL TOKENS
    ###
    base_model.resize_token_embeddings(len(tokenizer))  # resize for special tokens
    ###
    ###

    model = PeftModel.from_pretrained(
        model=base_model,
        model_id=configs.model_name,
        device_map={"": accelerator.process_index},
    )

    # Prep the evaluation dataset
    test_set = load_dataset(configs, split="test")
    # if "output_key" not in test_set.column_names:
    #     test_set["output_key"] = "template"
    #     test_set = preprocess_dataset(test_set)

    if configs.debug:
        n_samples = 100
        accelerator.print(f"DEBUG MODE: using only {n_samples} samples")
    else:
        n_samples = len(test_set)

    test_set = test_set.select(range(n_samples))
    test_set : Dataset = prepare_test_set(test_set, tokenizer)

    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    generation_kwargs = generation_strategies[configs.generation]
    generation_kwargs["tokenizer"] = tokenizer
    accelerator.print(f"Generating predictions using {configs.generation} strategy")

    predictor = get_predictor(
        configs,
        model,
        tokenizer,
        eos_token_id=[tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids(EOS_TOKEN)],
        # stop_strings=[EOS_TOKEN],
    )

    # divide the prompt list onto the available GPUs
    with accelerator.split_between_processes(test_set) as test_set_split:
        predictions = get_predictions(
            test_set_split, predictor, generation_kwargs, configs, recache=True
        )

    predictions = gather_object(predictions)
    predictions = [[pred.split(EOS_TOKEN)[0] for pred in sublist] for sublist in predictions]

    if accelerator.is_main_process:
        timediff = time.time() - start
        num_tokens = sum(
            len(prediction)
            for prediction_list in predictions
            for prediction in prediction_list
        )

        accelerator.print(
            f"predictions: {len(predictions)}, tokens/sec: {num_tokens // timediff}, time elapsed: {timediff}, num_tokens {num_tokens}"
        )

        import json
        import os

        os.makedirs(configs.output_dir, exist_ok=True)
        file_path = (
            f"{configs.output_dir}/predictions-{configs.generation}.json"
            if not configs.debug
            else f"{configs.output_dir}/predictions-{configs.generation}-debug.json"
        )
        with open(file_path, "w") as f:
            json.dump(predictions, f)

        accelerator.print(f"predictions saved to {configs.output_dir}")
        
        # if configs.upload:
        accelerator.print("Uploading predictions to the hub...")
        
        results_set = test_set.add_column(
            "predictions", predictions
        )
        results_set.push_to_hub(
            "yalhessi/lemexp-predictions",
            config_name=f"{configs.model_name}/{configs.dataset_config}/{configs.generation}",
        )