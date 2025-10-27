import multiprocessing
import os
import pickle

from textdistance import hamming, jaccard, levenshtein
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

from lemma_exploration.args import LemExpCLIArgs
from lemma_exploration.dataset import OUTPUT_TOKEN, load_dataset, process_dataset, truncate_dataset
from lemma_exploration.decorators import file_cache
from isabelle_connector.isabelle_connector import IsabelleConnector, temp_theory
from lemma_exploration.model import HuggingFaceModel, get_predictor
from lemma_exploration.utils import (
    get_run_id,
    path_to_theory_name,
)

# from lemma_exploration.parse import parse_lemma

TOKENIZERS_PARALLELISM = False


class EvalArgs(LemExpCLIArgs):
    @staticmethod
    def additional_args(parser):
        from argparse import BooleanOptionalAction

        parser.add_argument("--upload", action=BooleanOptionalAction, default=False)
        parser.add_argument(
            "-j", "--n-proc", type=int, default=multiprocessing.cpu_count() // 2
        )
        parser.add_argument("--generation-strategy", type=str, default=None)


def get_prediction_dir(args):
    if args.generation_strategy:
        predictions_path = f"{args.output_dir}/predictions-{args.generation_strategy}"
    else:
        predictions_path = f"{args.output_dir}/predictions"
    if args.debug:
        predictions_path = f"{predictions_path}-debug"
    return predictions_path


def save_predictions(predictions, args):
    prediction_dir = get_prediction_dir(args)
    if not os.path.exists(prediction_dir):
        os.mkdir(prediction_dir)
    pickle.dump(predictions, open(f"{prediction_dir}/predictions.pkl", "wb"))


def load_predictions(args):
    prediction_dir = get_prediction_dir(args)
    prediction_file = f"{prediction_dir}/predictions.pkl"
    if (
        os.path.exists(prediction_dir)
        and os.path.exists(prediction_file)
        and args.use_cache
    ):
        return pickle.load(open(prediction_file, "rb"))
    return []


def upload_results(result_set, args):
    run_id = get_run_id(args.model_name, args.dataset_name, args.dataset_config)

    # get timestamp
    import time

    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    result_set.push_to_hub("yalhessi/lemexp-results", f"{run_id}-results-{time_stamp}")


def prepare_test_set(test_set, tokenizer):
    truncated_set = truncate_dataset(
        test_set,
        fn_kwargs={
            "key": "input",
            "max_len": 1024 - 200,
            "tokenizer": tokenizer,
        },
    )

    sep = OUTPUT_TOKEN

    final_set = truncated_set.map(
        lambda x: {"input": x["input"] + sep},
        batched=False,
        load_from_cache_file=True,
    )
    return final_set


def get_predictions(test_set, predictor, generation_kwargs, configs, recache=False):
    # There are two options to make predictions:
    # Method 1:
    # Make predictions using the pipeline
    @file_cache(recache=recache)
    def _pipeline_predictions(generation_kwargs, configs):
        predictions = [
            [res["generated_text"] for res in log]
            for log in tqdm(
                predictor(KeyDataset(test_set, "input"), **generation_kwargs),  # type: ignore
                total=len(test_set),
            )
        ]
        return predictions

    return _pipeline_predictions(generation_kwargs, configs)
    # predictions = load_predictions(configs)

    # if not predictions:

    # Method 2:
    # Make predictions using the tokenizer and model directly
    # TODO: This crashes when we scale up to the full dataset. Need to investigate why.
    # test_set = test_set.map(lambda x: {"sep": SPECIAL_TOKENS["template"]})
    # inputs = tokenizer(test_set["input"], padding=True, return_tensors='pt').to(model.device)

    # generated_ids = model.generate(
    #     **inputs, max_length=1024, eos_token_id=tokenizer.encode(SPECIAL_TOKENS["eos"]))

    # predictions = tokenizer.batch_decode(
    #     generated_ids, skip_special_tokens=True)


def eval_template(row, predictions, configs):
    reference = row["template"]
    # prediction = row["prediction"]
    name = row["theory_file"] + "/" + row["lemma_name"]

    imports = [
        # "/public/yousef/lemma-exploration/thys/AbstractLemma",
        "/public/yousef/lemma-exploration/thys/RoughSpec",
        "/public/yousef/lemma-exploration/thys/ExtractLemmas",
    ]

    new_thy_name = "Eval_Template_" + name.replace("/", "_").replace("-", "_").replace(
        ".", "_"
    ).replace("(", "_").replace(")", "_").replace("'", "_")
    query = "\n".join(
        f"""
        ML\<open>
        let
        val ref_str = "{reference}"
        val ref_term = Utils.read_template ref_str
        val prediction_str = "{prediction.replace('"', '\\"').replace("\<open>", "").replace("\<close>", "")}"
        val prediction_term = Utils.read_template prediction_str handle ERROR _ => Term.dummy 
        val result = AbstractLemma.same_term ref_term prediction_term
        in
        result
        end
        \<close>
    """
        for prediction in predictions
    )
    thy = temp_theory(
        working_directory=configs.root_dir,
        queries=[query],
        imports=imports,
        name=new_thy_name,
    )
    return thy


def unrolled_eval_template(row, predictions, configs):
    reference = row["template"]
    # prediction = row["prediction"]
    name = row["theory_file"] + "/" + row["lemma_name"]

    imports = [
        # "/public/yousef/lemma-exploration/thys/AbstractLemma",
        "/public/yousef/lemma-exploration/thys/RoughSpec",
        "/public/yousef/lemma-exploration/thys/ExtractLemmas",
    ]

    thys = []
    for i, prediction in enumerate(predictions):
        new_thy_name = (
            "Eval_Template_"
            + name.replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace("'", "_")
            + f"_{i}"
        )
        query = f"""
            ML\<open>
            let
            val ref_str = "{reference}"
            val ref_term = Utils.read_template ref_str
            val prediction_str = "{prediction.replace('"', '\\"').replace("\<open>", "").replace("\<close>", "")}"
            val prediction_term = Utils.read_template prediction_str handle ERROR _ => Term.dummy 
            val result = AbstractLemma.same_term ref_term prediction_term
            in
            result
            end
            \<close>
        """
        thy = temp_theory(
            working_directory=configs.root_dir,
            queries=[query],
            imports=imports,
            name=new_thy_name,
        )
        thys.append(thy)
    return thys


def score_predictions(test_set, predictions, isabelle, configs):
    tasks = [
        eval_template(row, predictions[i], configs) for i, row in enumerate(test_set)
    ]
    results = isabelle.use_theories(
        tasks, batch_size=10, recache=True, rerun_failed=True
    )
    any_success = [any(result) if result else None for result in results.values()]
    # print(f" results: {Counter(any_success)}")
    return any_success


def compare_lemma_objects_batch(predictions, references, thy, configs):
    thy_name = f"Eval_Template_{path_to_theory_name(thy.name)}"
    queries = ["declare [[ML_catch_all]]"] + [
        f"""
        ML\<open>
        let
        val ref_str = "{reference}"
        val ref_term = Syntax.parse_term @{{context}} ref_str |> HOLogic.mk_Trueprop
        val prediction_str = "{prediction}"
        val prediction_term = (Syntax.parse_term @{{context}} prediction_str |> HOLogic.mk_Trueprop) handle  _ => Term.dummy 
        val result = AbstractLemma.same_term_untyped ref_term prediction_term
        in
        result
        end
        \<close>
    """
        for prediction, reference in zip(predictions, references)
    ]
    thy = temp_theory(
        working_directory=os.path.join(
            configs.cache_dir, os.path.basename(configs.root_dir)
        ),
        queries=queries,
        imports=configs.imports + [thy.name],
        name=thy_name,
    )
    # thys.append(thy)
    return thy


def compare_lemma_objects(prediction, reference, thy, lemma_name, configs):
    thy_name = f"Eval_Template_{path_to_theory_name(thy.name)}_{path_to_theory_name(lemma_name)}"
    query = f"""
    declare [[ML_catch_all]]
        ML\<open>
        let
        val ref_str = "{reference}"
        val ref_term = Syntax.parse_term @{{context}} ref_str |> HOLogic.mk_Trueprop
        val prediction_str = "{prediction}"
        val prediction_term = (Syntax.parse_term @{{context}} prediction_str |> HOLogic.mk_Trueprop) handle _ => Term.dummy 
        val result = AbstractLemma.same_term_untyped ref_term prediction_term
        in
        result
        end
        \<close>
    """
    thy = temp_theory(
        working_directory=os.path.join(
            configs.cache_dir, os.path.basename(configs.root_dir)
        ),
        queries=[query],
        imports=configs.imports + [thy.name],
        name=thy_name,
    )
    # thys.append(thy)
    return thy


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    model_parser = EvalArgs.model_args_parser()
    dataset_parser = EvalArgs.dataset_args_parser()
    configs = EvalArgs.parse_args(
        description="Evaluate model on HOL dataset",
        parents=[model_parser, dataset_parser],
    )
    print(configs)

    task = "template"
    model = HuggingFaceModel(is_trainable=False, configs=configs)

    model, tokenizer = model.model, model.tokenizer
    test_set = load_dataset(configs, split="test")
    test_set = process_dataset(test_set, task)
    if configs.debug:
        size = 100
        test_set = test_set.select(range(size))
        print(f"Debugging on {size} samples")
    else:
        size = 1000
        test_set = test_set.select(range(size))
        print(f"Running on {size} samples")

    predictor = get_predictor(configs, model, tokenizer)

    greedy_kwargs = {
        "batch_size": 4,
    }
    # beam search, top 5 sequences
    beam_search_top5_kwargs = {
        "num_beams": 5,
        "batch_size": 1,
        "num_return_sequences": 5,
    }
    test_set = prepare_test_set(test_set, tokenizer)

    greedy_predictions = get_predictions(
        test_set, predictor, greedy_kwargs, configs, recache=False
    )
    beam_search_top5_predictions = get_predictions(
        test_set, predictor, beam_search_top5_kwargs, configs, recache=False
    )

    isabelle = IsabelleConnector(name="eval", working_directory=configs.root_dir)
    greedy_results = score_predictions(test_set, greedy_predictions, isabelle, configs)
    beam_search_top5_results = score_predictions(
        test_set, beam_search_top5_predictions, isabelle, configs
    )
    any_results = [
        any(result_vec) for result_vec in zip(greedy_results, beam_search_top5_results)
    ]

    min_greedy_hamming = [
        [hamming(prediction, reference) for prediction in predictions]
        for predictions, reference in zip(greedy_predictions, test_set["template"])
    ]
    min_beam_search_top5_hamming = [
        [hamming(prediction, reference) for prediction in predictions]
        for predictions, reference in zip(
            beam_search_top5_predictions, test_set["template"]
        )
    ]
    min_hamming = [
        min(greedy, beam_search_top5)
        for greedy, beam_search_top5 in zip(
            min_greedy_hamming, min_beam_search_top5_hamming
        )
    ]

    min_greedy_levenshtein = [
        [levenshtein(prediction, reference) for prediction in predictions]
        for predictions, reference in zip(greedy_predictions, test_set["template"])
    ]
    min_beam_search_top5_levenshtein = [
        [levenshtein(prediction, reference) for prediction in predictions]
        for predictions, reference in zip(
            beam_search_top5_predictions, test_set["template"]
        )
    ]
    min_levenshtein = [
        min(greedy, beam_search_top5)
        for greedy, beam_search_top5 in zip(
            min_greedy_levenshtein, min_beam_search_top5_levenshtein
        )
    ]

    max_greedy_jaccard = [
        [jaccard(prediction, reference) for prediction in predictions]
        for predictions, reference in zip(greedy_predictions, test_set["template"])
    ]
    max_beam_search_top5_jaccard = [
        [jaccard(prediction, reference) for prediction in predictions]
        for predictions, reference in zip(
            beam_search_top5_predictions, test_set["template"]
        )
    ]
    max_jaccard = [
        max(greedy, beam_search_top5)
        for greedy, beam_search_top5 in zip(
            max_greedy_jaccard, max_beam_search_top5_jaccard
        )
    ]

    result_set = (
        test_set.add_column("greedy_predictions", greedy_predictions)
        .add_column("greedy_success", greedy_results)
        .add_column("beam_search_top5_predictions", beam_search_top5_predictions)
        .add_column("beam_search_top5_success", beam_search_top5_results)
        .add_column("any_success", any_results)
        .add_column("min_greedy_hamming", min_greedy_hamming)
        .add_column("min_beam_search_top5_hamming", min_beam_search_top5_hamming)
        .add_column("min_hamming", min_hamming)
        .add_column("min_greedy_levenshtein", min_greedy_levenshtein)
        .add_column(
            "min_beam_search_top5_levenshtein", min_beam_search_top5_levenshtein
        )
        .add_column("min_levenshtein", min_levenshtein)
        .add_column("max_greedy_jaccard", max_greedy_jaccard)
        .add_column("max_beam_search_top5_jaccard", max_beam_search_top5_jaccard)
        .add_column("max_jaccard", max_jaccard)
    )

    if configs.upload:
        upload_results(result_set, configs)
