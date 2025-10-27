import argparse
import os
import pandas as pd
import seaborn as sns
from transformers import Trainer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM

from lemma_exploration.config import OUTPUT_DIR
from lemma_exploration.dataset import load_dataset, tokenize_dataset
from lemma_exploration.model import HuggingFaceModel, test_model
from lemma_exploration.utils import ParseKwargs, get_run_id


DEFAULT_TRAINING_ARGS = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "num_train_epochs": 6,
    "learning_rate": 2e-4,
    "fp16": True,
    "eval_strategy": "steps",
    "eval_steps": 1 / (6 * 5),
    "save_strategy": "epoch",
    "hub_strategy": "all_checkpoints",
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

    response_template = "###output\n"

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
    trainer.save_state()
    trainer.save_model()
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


def main():
    args = parse_args()
    print(f"Running with args: {args}")

    model = HuggingFaceModel(
        load_model_from="web",
        is_trainable=True,
        use_quantization=args.use_quantization,
        use_peft=args.use_peft,
        model_name=args.model_name,
    )
    model, tokenizer = model.model, model.tokenizer

    if args.debug:
        test_model(args, model, tokenizer, input_text="What are we having for dinner?")

    dataset_dict = load_dataset(args=args, split=None)
    trainer = get_trainer(args, model, tokenizer, dataset_dict)

    if args.resume:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    save_trained_state(trainer)
    plot_loss_data(trainer)
    # test_model(model, tokenizer, dataset["test"])


if __name__ == "__main__":
    main()
