import time
from lemma_exploration.eval import *
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import gather_object

from argparse import Namespace


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


model_configs = Namespace(
    load_model_from="web-peft",
    model_name="yalhessi/lemexp-processed-task1_min_symbols_template_small-deepseek-coder-1.3b-base",
    base_model_name="deepseek-ai/deepseek-coder-1.3b-base",
)

configs = Namespace(
    dataset_name="yalhessi/lemexp-processed",
    dataset_config="task1_min_symbols_template_small",
    # dataset_config="hol-thms-by-file",
    # dataset_config="afp-small-thms",
    output_dir=None,
    max_seq_length=1024,
    use_cache=True,
    debug=False,
    upload=False,
    n_proc=1,
    # device=7,
    root_dir="/public/yousef/lemma-exploration/isabelle/afp-small",
    imports=[
        "/public/yousef/lemma-exploration/thys/ExtractLemmas",
        "/public/yousef/lemma-exploration/thys/RoughSpec",
    ],
)

run_id = get_run_id(
    model_configs.model_name, configs.dataset_name, configs.dataset_config
)
configs.output_dir = f"/public/yousef/lemma-exploration/output/{run_id}"
print(configs)


accelerator = Accelerator()

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_configs.base_model_name)

test_set = load_dataset(configs, split="test")
test_set = prepare_test_set(test_set, tokenizer)

base_model = AutoModelForCausalLM.from_pretrained(
    model_configs.base_model_name,
    device_map={"": accelerator.process_index},
)
base_model.resize_token_embeddings(len(tokenizer) + 5)

model = PeftModel.from_pretrained(
    model=base_model,
    model_id=model_configs.model_name,
    device_map={"": accelerator.process_index},
)

prompts_all = test_set["input"][:1000]


# sync GPUs and start the timer
accelerator.wait_for_everyone()
start = time.time()


# divide the prompt list onto the available GPUs
with accelerator.split_between_processes(prompts_all) as prompts:
    results = dict(outputs=[], num_tokens=0)

    # have each GPU do inference in batches
    prompt_batches = prepare_prompts(prompts, tokenizer, batch_size=4)

    for prompts_tokenized in prompt_batches:
        outputs_tokenized = model.generate(
            **prompts_tokenized, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id
        )

        # remove prompt from gen. tokens
        outputs_tokenized = [
            tok_out[len(tok_in) :]
            for tok_in, tok_out in zip(
                prompts_tokenized["input_ids"], outputs_tokenized
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

results_gathered = gather_object(results)

if accelerator.is_main_process:
    timediff = time.time() - start
    num_tokens = sum([r["num_tokens"] for r in results_gathered])

    print(
        f"tokens/sec: {num_tokens // timediff}, time elapsed: {timediff}, num_tokens {num_tokens}"
    )

    accelerator.print(results_gathered)
