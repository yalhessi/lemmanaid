import argparse
import time
from datetime import timedelta

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lemma_exploration.eval import *
from lemma_exploration.utils import get_run_id


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
        args.output_dir = f"/home/ubuntu/lemma-exploration/output/{run_id}"

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
    configs = parse_args()
    print(f"Configs: {configs}")

    tokenizer = AutoTokenizer.from_pretrained(configs.base_model_name)

    test_set = load_dataset(configs, split="test")

    if configs.debug:
        n_samples = 100
        print(f"DEBUG MODE: using only {n_samples} samples")
    else:
        n_samples = len(test_set)

    test_set = test_set.select(range(n_samples))
    test_set = prepare_test_set(test_set, tokenizer)
    prompts_all = test_set["input"]

    base_model = AutoModelForCausalLM.from_pretrained(
        configs.base_model_name,
        device_map="auto",
    )

    ### FOR MODELS TRAINED BEFORE REMOVING THE SPECIAL TOKENS
    ###
    base_model.resize_token_embeddings(len(tokenizer) + 5)
    ###
    ###

    model = PeftModel.from_pretrained(
        model=base_model,
        model_id=configs.model_name,
        device_map="auto",
    )

    start = time.time()

    greedy_kwargs = {
        "batch_size": 4,
        "num_return_sequences": 1,
    }
    # beam search, top 5 sequences
    beam_search_top5_kwargs = {
        "num_beams": 5,
        "batch_size": 1,
        "num_return_sequences": 5,
        # "early_stopping": True,
    }

    generation_kwargs = (
        greedy_kwargs if configs.generation == "greedy" else beam_search_top5_kwargs
    )

    print(f"Generating predictions using {configs.generation} strategy")

    results = dict(outputs=[], num_tokens=0)

    if "batch_size" in generation_kwargs:
        batch_size = generation_kwargs["batch_size"]
        generation_kwargs.pop("batch_size")
    else:
        batch_size = 1

    if "num_return_sequences" in generation_kwargs:
        num_return_sequences = generation_kwargs["num_return_sequences"]
    else:
        num_return_sequences = 1

    # have each GPU do inference in batches
    prompt_batches = prepare_prompts(prompts_all, tokenizer, batch_size=batch_size)

    for prompts_tokenized in prompt_batches:
        outputs_tokenized = model.generate(
            **prompts_tokenized,
            max_length=1024,
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )

        # remove prompt from gen. tokens
        # import pdb

        # pdb.set_trace()
        outputs_tokenized = [
            tok_out[len(tok_in) :]
            for tok_in, tok_out in zip(
                prompts_tokenized["input_ids"].repeat(num_return_sequences, 1),
                outputs_tokenized,
            )
        ]

        # count and decode gen. tokens
        num_tokens = sum([len(t) for t in outputs_tokenized])
        outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)

        # store in results{} to be gathered by accelerate
        results["outputs"].extend(outputs)
        results["num_tokens"] += num_tokens

    results = [
        results
    ]  # transform to list, otherwise gather_object() will not collect correctly

    results_gathered = results

    timediff = time.time() - start
    num_tokens = sum([r["num_tokens"] for r in results_gathered])

    print(
        f"tokens/sec: {num_tokens // timediff}, time elapsed: {timediff}, num_tokens {num_tokens}"
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
        json.dump(results_gathered, f)

    print(f"predictions saved to {configs.output_dir}")
