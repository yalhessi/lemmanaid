from datasets import Dataset
from datasets import load_dataset as load_hf_dataset


def load_dataset(args, split=None) -> Dataset:
    split = split if split != "all" else None
    dataset: Dataset = load_hf_dataset(
        args.dataset_name, name=args.dataset_config, split=split
    )  # type: ignore
    return dataset


SYMBOLS_TOKEN = "###symbols\n"
TYPE_TOKEN = "::::"
DEFS_TOKEN = "\n###defs\n"
OUTPUT_TOKEN = "\n###output\n"
SEP_TOKEN = "\n\n"
EOS_TOKEN = "###end"

TOKEN_LIST = [SYMBOLS_TOKEN, TYPE_TOKEN, DEFS_TOKEN, OUTPUT_TOKEN, SEP_TOKEN]

SPECIAL_TOKENS = {"eos_token": EOS_TOKEN}

def stringify_input(example):
    return f"{SYMBOLS_TOKEN}{SEP_TOKEN.join(example['symbols'])}{DEFS_TOKEN}{SEP_TOKEN.join(example['defs'])}"

def stringify_input_with_types_no_defs(example):
    return f"{SYMBOLS_TOKEN}{SEP_TOKEN.join(f'{sym} {TYPE_TOKEN} {typ}' for sym, typ in zip(example['symbols'], example['types']))}"

def stringify_input_with_types(example):
    return f"{SYMBOLS_TOKEN}{SEP_TOKEN.join(f'{sym} {TYPE_TOKEN} {typ}' for sym, typ in zip(example['symbols'], example['types']))}{DEFS_TOKEN}{SEP_TOKEN.join(example['defs'])}"


def stringify_output(example):
    return f"{OUTPUT_TOKEN}{example[example['output_key']]}{EOS_TOKEN}"


def process_dataset(dataset, input_fn=stringify_input, output_fn=stringify_output):
    return dataset.map(
        lambda example: {
            "input": input_fn(example),
            "output": output_fn(example),
        },
        batched=False,
        load_from_cache_file=True,
    )


def tokenize_dataset(dataset, tokenizer, max_length):
    print(f"Tokenizing dataset with max_length={max_length}")
    tokenized_dataset = dataset.map(
        lambda batch: tokenizer(
            batch["input"],
            batch["output"],
            truncation="longest_first",
            max_length=max_length,
        ),
        batched=True,
        load_from_cache_file=True,
    )
    return tokenized_dataset


def truncate_dataset(dataset, fn_kwargs=None):
    return dataset.map(
        truncate_example,
        fn_kwargs=fn_kwargs,
        batched=False,
        load_from_cache_file=True,
    )


def truncate_example(example, key, tokenizer, max_len):
    return {
        key: tokenizer.decode(
            tokenizer.encode(
                example[key],
                max_length=max_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )[0]
        )
    }


def split_dataset(dataset, train_size=0.8, test_size=0.1, valid_size=0.1):
    from datasets import DatasetDict

    train_testvalid = dataset.train_test_split(test_size=test_size + valid_size)
    test_valid = train_testvalid["test"].train_test_split(
        test_size=test_size / (test_size + valid_size)
    )
    # gather everyone if you want to have a single DatasetDict
    return DatasetDict(
        {
            "train": train_testvalid["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )
