from argparse import Namespace

from datasets import load_dataset as load_hf_dataset
import pandas as pd
from textdistance import levenshtein

from lemma_exploration.dataset import load_dataset
from lemma_exploration.utils import (
    clean_term,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate generated predictions on lemma exploration tasks (make sure predictions appear in yalhessi/lemexp-predictions before running this script)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to evaluate (Huggingface name)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="yalhessi/lemexp-task1-v2",
        help="Name of the dataset to evaluate on (Huggingface name)",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Name of the configuration the model was trained on",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        required=True,
        help="Name of the configuration to evaluate the model on",
    )
    parser.add_argument(
        "--generation-strategies",
        nargs="+",
        type=str,
        default=["greedy"],
        help="List of generation strategies to evaluate",
    )
    args = parser.parse_args()
    return args


def normalize_vars(term):
    counter = 1
    d = {}
    new_term = []
    cleaned_term = (
        clean_term(term).replace("(", " ( ").replace(")", " ) ").replace("?", " ?")
    )
    for tok in cleaned_term.split():
        if tok.startswith("?"):
            if tok not in d:
                d[tok] = f"?x{counter}"
                counter += 1

            new_term.append(d[tok])
        else:
            new_term.append(tok)
    return " ".join(new_term)


def eval_template_predictions(generation_strategy, df, template_predictions):
    predictions = [
        [clean_term(prediction) for prediction in pred_list]
        for pred_list in template_predictions
    ]
    template_references = [clean_term(template) for template in df["template"].tolist()]

    template_levenshtein_scores = [
        min([levenshtein(prediction, reference) for prediction in p_list])
        for p_list, reference in zip(predictions, template_references)
    ]
    df[f"template_predictions_{generation_strategy}"] = predictions
    df[f"template_levenshtein_scores_{generation_strategy}"] = (
        template_levenshtein_scores
    )
    df[f"template_success_{generation_strategy}"] = (
        df[f"template_levenshtein_scores_{generation_strategy}"] == 0
    )  # & df["rediscovery_success"]
    print(
        f"Generation strategy: {generation_strategy} | Num of correct predictions: {len(df[df[f'template_success_{generation_strategy}']])} (%{len(df[df[f'template_success_{generation_strategy}']]) / len(df) * 100:.1f})"
    )


def eval_lemma_object_predictions(generation_strategy, df, lemma_object_predictions):
    lemma_object_predictions = [
        [clean_term(prediction) for prediction in pred_list]
        for pred_list in lemma_object_predictions
    ]

    lemma_object_references = [
        clean_term(lemma_object) for lemma_object in df["lemma_object"].tolist()
    ]

    normalized_lemma_objects = [
        normalize_vars(lemma_object) for lemma_object in lemma_object_references
    ]

    normalized_lemma_object_predictions = [
        [normalize_vars(prediction) for prediction in predictions]
        for predictions in lemma_object_predictions
    ]

    normalized_lemma_object_levenstein_scores = [
        min(
            [
                levenshtein(clean_term(prediction), clean_term(reference))
                for prediction in predictions
            ]
        )
        for predictions, reference in zip(
            normalized_lemma_object_predictions, normalized_lemma_objects
        )
    ]
    df[f"lemma_object_predictions_{generation_strategy}"] = lemma_object_predictions
    df[f"lemma_object_levenshtein_scores_{generation_strategy}"] = (
        normalized_lemma_object_levenstein_scores
    )
    df[f"lemma_object_success_{generation_strategy}"] = (
        df[f"lemma_object_levenshtein_scores_{generation_strategy}"] == 0
    )  # & df["rediscovery_success"]
    print(
        f"Generation strategy: {generation_strategy} | Num of correct predictions: {len(df[df[f'lemma_object_success_{generation_strategy}']])} (%{len(df[df[f'lemma_object_success_{generation_strategy}']]) / len(df) * 100:.1f})"
    )
    return lemma_object_predictions


def eval_predictions(
    model_name, dataset_name, train_config, eval_config, generation_strategies
):
    all_predictions = pd.DataFrame()
    all_results = pd.DataFrame()
    print(
        f"Evaluating model: {model_name} | Trained on: {train_config} | Evaluated on: {eval_config}"
    )

    eval_set_configs = Namespace(
        dataset_name=dataset_name,
        dataset_config=eval_config,
    )

    eval_set = load_dataset(
        eval_set_configs,
        split="test",
    )
    if "full" in eval_config:
        eval_set = eval_set.select(range(4740, len(eval_set)))

    df = eval_set.remove_columns(["input", "output", "output_key"]).to_pandas()

    any_strategy_pd = pd.DataFrame()
    for gen_strat in generation_strategies:
        try:
            configs = Namespace(
                dataset_name="yalhessi/lemexp-predictions",
                dataset_config=f"{model_name}/{eval_config}/{gen_strat}",
            )
            dataset = load_hf_dataset(
                configs.dataset_name,
                data_dir=configs.dataset_config,
                split="test",
            )
            if "full" in eval_config:
                dataset = dataset.select(range(4740, len(dataset)))

            if "template" in train_config:
                eval_template_predictions(
                    generation_strategy=gen_strat,
                    df=df,
                    template_predictions=dataset["predictions"],
                )
                result_col = (
                    df[f"template_success_{gen_strat}"]
                    # & df["rediscovery_success"],
                )
            elif "lemma_object" in train_config:
                eval_lemma_object_predictions(
                    generation_strategy=gen_strat,
                    df=df,
                    lemma_object_predictions=dataset["predictions"],
                )
                result_col = (
                    df[f"lemma_object_success_{gen_strat}"]
                    # & df["rediscovery_success"]
                )
            else:
                predictions = []
            name = f"{train_config}_success_{gen_strat}"
            any_strategy_pd[name] = result_col
            all_results[name] = result_col
            predictions = dataset["predictions"]
            all_predictions[f"{train_config}_predictions_{gen_strat}"] = predictions
            # if True:
            #     result_set = Dataset.from_pandas(df)
            #     result_set.push_to_hub("yalhessi/lemexp-task1-v2-eval-results", config_name=f"finetuned_on_{train_config}_eval_on_{eval_config}_generation_{gen_strat}")
        except Exception as e:
            print(f"Error evaluating predictions: {e}")
            continue
    any_strategy_pd["success_count"] = any_strategy_pd.sum(axis=1)
    any_strategy_pd["any_success"] = any_strategy_pd.any(axis=1)
    any_strategy_pd.insert(0, "theory_file", df["theory_file"])
    any_strategy_pd.insert(1, "lemma_name", df["lemma_name"])
    try:
        print(
            f"Total number of correct predictions for {train_config}: {len(any_strategy_pd[any_strategy_pd['any_success']])} (%{len(any_strategy_pd[any_strategy_pd['any_success']]) / len(any_strategy_pd) * 100:.1f})"
            "\n==============="
        )
    except Exception as e:
        print(f"Error printing results: {e}")

    all_results["success_count"] = all_results.sum(axis=1)
    all_results["any_success"] = all_results.any(axis=1)
    all_results.insert(0, "theory_file", df["theory_file"])
    all_results.insert(1, "lemma_name", df["lemma_name"])
    print(
        f"Total number of correct predictions: {len(all_results[all_results['any_success']])} (%{len(all_results[all_results['any_success']]) / len(all_results) * 100:.1f})"
        "\n==============="
    )


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    dataset_name = args.dataset_name
    train_config = args.train_config
    eval_config = args.eval_config
    generation_strategies = args.generation_strategies
    eval_predictions(
        model_name, dataset_name, train_config, eval_config, generation_strategies
    )
