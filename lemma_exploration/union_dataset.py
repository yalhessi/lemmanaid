import string
from collections import Counter

from dataset import load_dataset
from datasets import Dataset, DatasetDict, concatenate_datasets
from numpy import isin


def parse_args():
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser(description="Union two datasets")
    parser.add_argument("--upload", action=BooleanOptionalAction, default=False)
    return parser.parse_args()


def project_path(theory_file):
    return theory_file.rsplit("/", 1)[0]


def top_projects_dataset(dataset, n_projects):
    if n_projects is None:
        return dataset

    if isinstance(dataset, DatasetDict):
        theory_files = dataset["train"]["theory_file"]
    else:
        theory_files = dataset["theory_file"]

    allowed_projects, counts = zip(
        *Counter(project_path(x) for x in theory_files).most_common(n_projects)
    )
    print(f"Top projects: {allowed_projects}")
    return dataset.filter(lambda x: project_path(x["theory_file"]) in allowed_projects)


if __name__ == "__main__":
    args = parse_args()
    name1, config1, split1 = "yalhessi/lemexp", "hol", "all"
    name2, config2, split2 = "yalhessi/lemexp", "afp", "all"
    n_projects1, n_project2 = 1, 20

    dataset1 = load_dataset(name1, config1, split1, preprocess=False)
    dataset2 = load_dataset(name2, config2, split2, preprocess=False)

    top_project2 = top_projects_dataset(dataset2, n_project2)

    if isinstance(dataset1, DatasetDict):
        dataset = DatasetDict(
            {
                split: concatenate_datasets([dataset1[split], top_project2[split]])
                for split in dataset1
            }
        )
    else:
        dataset = concatenate_datasets([dataset1, top_project2])

    print(f"Dataset size: {dataset.num_rows}")
    print(f"Dataset1 size: {dataset1.num_rows}")
    print(f"Dataset2 size: {dataset2.num_rows}")
    print(f"Top projects size: {top_project2.num_rows}")

    if args.upload:
        dataset.push_to_hub("yalhessi/lemexp", f"{config1}+{config2}{n_project2}")
