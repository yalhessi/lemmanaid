import argparse
from datetime import timedelta
import os
import pandas as pd
import seaborn as sns
import torch
from transformers import Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM

from lemma_exploration.config import OUTPUT_DIR
from lemma_exploration.dataset import OUTPUT_TOKEN, SEP_TOKEN, SPECIAL_TOKENS, TOKEN_LIST, TYPE_TOKEN, load_dataset, tokenize_dataset, EOS_TOKEN
from lemma_exploration.model import HuggingFaceModel, test_model
from lemma_exploration.utils import ParseKwargs, get_run_id
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft.utils.other import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model

DEFAULT_TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "num_train_epochs": 6,
    "learning_rate": 8e-4,
    "fp16": True,
    "eval_strategy": "steps",
    "eval_steps": 1 / (6 * 5),
    "save_strategy": "epoch",
    "hub_strategy": "all_checkpoints",
    "push_to_hub": True,
    "hub_always_push": True,
}


def parse_args(**kwargs):
    # load_dotenv()
    parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument(
        "--load-model-from",
        type=str,
        default="web",
        choices=["web", "cache", "web-peft"],
    )
    parser.add_argument("--model-name", type=str, help="model name", required=True)

    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--use-quantization", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-peft", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dataset-name", type=str, help="dataset name", required=True)
    parser.add_argument(
        "--dataset-config", type=str, help="dataset config", required=True
    )
    parser.add_argument("--training-args", nargs="*", action=ParseKwargs, default={})
    parser.add_argument("--label", type=str, help="label for the run", default="")
    args = parser.parse_args()
    args.training_args = {**DEFAULT_TRAINING_ARGS, **args.training_args}
    if not args.output_dir:
        args.output_dir = os.path.join(
            OUTPUT_DIR,
            get_run_id(
                args.model_name, args.dataset_name, args.dataset_config, args.label
            ),
        )  # type: ignore
    return args


def get_trainer(args, model, tokenizer, dataset):
    tokenized_dataset = tokenize_dataset(
        dataset, tokenizer, max_length=args.max_seq_length
    )
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["valid"]

    # # shuffle + remove index mapping to avoid slow training
    # shuffled_dataset = tokenized_dataset.shuffle()
    # shuffled_dataset.flatten_indices()

    response_template = OUTPUT_TOKEN

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        ddp_find_unused_parameters=False,
        **args.training_args,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    return trainer


def save_trained_state(trainer):
    # trainer.save_state()
    # trainer.save_model()
    trainer.push_to_hub()
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)


def plot_loss_data(trainer: Trainer):
    loss_data = pd.DataFrame(
        [
            {
                "step": log["step"],
                "training_loss": log["loss"]
                if "loss" in log and log["step"] % 10 == 0
                else None,
                "eval_loss": log["eval_loss"] if "eval_loss" in log else None,
            }
            for log in trainer.state.log_history
        ]
    )

    raw_loss_data = pd.melt(loss_data, ["step"])
    g = sns.relplot(raw_loss_data, x="step", y="value", hue="variable", kind="line")
    g.savefig(f"{trainer.args.output_dir}/loss_plot.png")

# import torch
# import gc

# def add_new_tokens(model, tokenizer, new_tokens=[], method="mean", interpolation=0.5):
#     assert isinstance(new_tokens, (list, tuple))
#     assert len(new_tokens) > 0
#     assert method in ["mean", "interpolation"]
#     assert 0 <= interpolation <= 1

#     overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
#     if overlapping_tokens:
#         print(f"Unsloth: Skipping overlapping tokens: {list(overlapping_tokens)}")
#         new_tokens = [x for x in new_tokens if x not in overlapping_tokens]

#     # Add new tokens to tokenizer
#     old_length = len(tokenizer)
#     tokenizer.add_tokens(new_tokens)

#     # Fix — resize before accessing embedding matrix
#     model.resize_token_embeddings(len(tokenizer))

#     # Get mean embedding
#     embedding_matrix = model.get_input_embeddings().weight.clone()
#     lm_head_matrix = model.get_output_embeddings().weight.clone()
#     eps = 1e-16
#     indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
#     where_untrained = torch.where(indicator_untrained)[0]
#     n_untrained = where_untrained.shape[0]
#     n_trained = embedding_matrix.shape[0] - n_untrained
#     sum_embedding = embedding_matrix.sum(dim=0) - embedding_matrix[where_untrained].sum(dim=0)
#     sum_lm_head = lm_head_matrix.sum(dim=0) - lm_head_matrix[where_untrained].sum(dim=0)
#     mean_embedding = (sum_embedding / n_trained).to(torch.float32)
#     mean_lm_head = (sum_lm_head / n_trained).to(torch.float32)

#     embedding_matrix = model.get_input_embeddings().weight
#     lm_head_matrix = model.get_output_embeddings().weight

#     if method == "interpolation":
#         print("Using interpolation for initializing new tokens.")
#         for j, token in enumerate(new_tokens):
#             input_ids = tokenizer(token, add_special_tokens = False).input_ids
#             mean_embedding_token = embedding_matrix[input_ids].mean(axis = 0, dtype = torch.float32)
#             mean_lm_head_token   = lm_head_matrix  [input_ids].mean(axis = 0, dtype = torch.float32)

#             # Interpolate
#             mean_embedding_token = mean_embedding*(1-interpolation) + mean_embedding_token*interpolation
#             mean_lm_head_token   = mean_lm_head  *(1-interpolation) + mean_lm_head_token  *interpolation

#             # Set the new vector
#             with torch.no_grad():
#                 embedding_matrix[old_length+j] = mean_embedding_token
#                 lm_head_matrix  [old_length+j] = mean_lm_head_token
#     else:
#         embedding_matrix.data[old_length:] = mean_embedding
#         lm_head_matrix.data[old_length:] = mean_lm_head

#     model.config.vocab_size = len(tokenizer)
#     if hasattr(model, "tie_weights"):
#         model.tie_weights()

#     for _ in range(3):
#         gc.collect()
#         torch.cuda.empty_cache()
#     print(f"✅ Added {len(new_tokens)} new tokens to the tokenizer and model.") 

def main():
    accelerator = Accelerator(
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(seconds=60 * 60 * 12))
        ]
    )

    args = parse_args()
    print(f"Running with args: {args}")

    # model = HuggingFaceModel(
    #     load_model_from="web",
    #     is_trainable=True,
    #     use_quantization=False,
    #     use_peft=True,
    #     model_name=args.model_name,
    # )
    # model, tokenizer = model.model, model.tokenizer
    # model = accelerator.prepare(model)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16,
        # bnb_4bit_quant_storage=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.add_tokens([EOS_TOKEN])

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        # load_in_8bit=True,
        # torch_dtype=torch.bfloat16,
    )  # .to(accelerator.device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # add_new_tokens(model, tokenizer, new_tokens=[EOS_TOKEN], method="interpolation", interpolation=0.3)
    model.resize_token_embeddings(len(tokenizer))
    # model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        # modules_to_save= ["embed_tokens", "lm_head"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    # model = accelerator.prepare(model)

    # # print_trainable_parameters(model)

    if args.debug:
        test_model(args, model, tokenizer, input_text="What are we having for dinner?")

    dataset_dict = load_dataset(args=args, split=None)
    trainer = get_trainer(args, model, tokenizer, dataset_dict)
    # if getattr(trainer.accelerator.state, "fsdp_plugin", None):
    #     from peft.utils.other import fsdp_auto_wrap_policy

    #     fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    #     fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # if trainer.is_fsdp_enabled:
    #     print("Setting FSDP state dict type to FULL_STATE_DICT")
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    plot_loss_data(trainer)
    save_trained_state(trainer)
    # test_model(model, tokenizer, dataset["test"])


if __name__ == "__main__":
    main()
