import argparse
import json
import multiprocessing
import os
import re

from lemma_exploration.dataset import EOS_TOKEN


def getenv(*args, **kwargs):
    return os.getenv(*args, **kwargs)


def get_run_id(model_name, dataset_name, dataset_config, label=""):
    def extract_name(name):
        return name.split("/")[-1] if "/" in name else name

    dataset_name = extract_name(dataset_name)
    model_name = extract_name(model_name)
    label_suffix = f"-{label}" if label else ""
    return f"{dataset_name}-{dataset_config}-{model_name}{label_suffix}"


def current_process_offset():
    return multiprocessing.current_process()._identity[0]


def flatten_dict(d_list: list[dict]) -> dict:
    return {k: v for d in d_list for k, v in d.items()}


def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]


def prepare_notebook():
    import nest_asyncio

    import lemma_exploration.reload_recursive

    nest_asyncio.apply()
    import warnings

    warnings.filterwarnings(action="once")
    import multiprocessing

    multiprocessing.set_start_method("spawn")


def path_to_theory_name(path):
    import re

    # remove all special characters
    return re.sub(r"\W+", "_", path).strip("_")


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            try:
                value = eval(value)
            except:
                pass
            getattr(namespace, self.dest)[key] = value


def clean_term(term):
    cleaned = term.split(EOS_TOKEN)[0].replace("?", " ?").replace("( ?", "(?").strip().replace("\\\\", "\\")
    uniform_spaces = re.sub(r"\s+", " ", cleaned)
    return uniform_spaces


def clean_prediction(prediction):
    if "\n" not in prediction:
        term = prediction
    else:
        term, rest = prediction.split("\n", 1)
    return clean_term(term)


def load_local_predictions(
    model_name, dataset_name, dataset_config, generation_strategy="greedy"
):
    run_id = get_run_id(model_name, dataset_name, dataset_config)
    with open(
        f"{getenv('OUTPUT_DIR')}/{run_id}/predictions-{generation_strategy}.json"
    ) as f:
        predictions = json.load(f)
        clean_predictions = [[clean_term(p) for p in p_list] for p_list in predictions]
        return clean_predictions
