from dataset import load_dataset

from isabelle_commands import eval_template
from lemma_exploration.parse import split_lemma


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extend dataset with additional features"
    )
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument("--dataset_config", type=str, help="dataset config")
    parser.add_argument("--split", type=str, help="dataset split")
    parser.add_argument(
        "--upload", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()

def template(example):
    lemma_name, lemma_string = split_lemma(example["lemma"])
    return eval_template(example["theory_file"], lemma_string)


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset(
        args.dataset_name, args.dataset_config, args.split, preprocess=False
    )
    
    dataset = dataset.map(
        lambda example: {
            "template": template(example)
        },
        batched=False,
        
        load_from_cache_file=False,
    )
    
    dataset = dataset.filter(lambda example: example["template"] is not None)
    
    print(f"Dataset size: {dataset.num_rows}")
    print(f"Dataset columns: {dataset.column_names}")

    if args.upload:
        dataset.push_to_hub(args.dataset_name, f"{args.dataset_config}-template")
