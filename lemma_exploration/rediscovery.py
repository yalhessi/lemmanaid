from datetime import datetime
import os
from argparse import Namespace

from lemma_exploration.dataset import load_dataset
from isabelle_connector.isabelle_connector import IsabelleConnector, temp_theory

from lemma_exploration.utils import path_to_theory_name


def rediscover_batch(references, thy, configs):
    thy_name = f"Rediscover_{path_to_theory_name(thy.name)}"
    queries = ["declare [[ML_catch_all]]"] + [
        f"""ML\\<open>
        let
        val thm = @{{thm {reference}}}
        val term = Thm.prop_of thm
        val template = AbstractLemma.abstract_term @{{context}} term
        val prettytemplate = Print_Mode.setmp [] (Syntax.string_of_term @{{context}}) template;
        val consts = RoughSpec_Utils.const_names_of_term @{{context}} term

        val lemmas = Timeout.apply_physical (Time.fromSeconds 60) (RoughSpec.templatePropsStringInputs @{{context}} prettytemplate) consts
        val result = if List.null lemmas then "empty" else (List.exists (AbstractLemma.match_lemma term) lemmas |> Bool.toString)
        in
            result
        end handle
            Timeout.TIMEOUT _ => "timeout"
          | _ => "error"
        \\<close>"""
        for reference in references
    ]
    thy = temp_theory(
        working_directory=os.path.join(
            configs.cache_dir, os.path.basename(configs.root_dir)
        ),
        queries=queries,
        imports=configs.imports + [thy.name],
        name=thy_name,
    )
    return thy


def rediscover(row, configs):
    reference = row["lemma_name"]
    src_theory_file = row["theory_file"]
    path, base_name = (
        src_theory_file.rsplit("/", 1)
        if "/" in src_theory_file
        else ("", src_theory_file)
    )
    name = src_theory_file + "/" + row["lemma_name"]
    new_thy_name = f"Rediscover_{path_to_theory_name(name)}"
    query2 = f"""ML\\<open>
        let
        val thm = @{{thm {reference}}}
        val term = Thm.prop_of thm
        val template = AbstractLemma.abstract_term @{{context}} term
        val prettytemplate = Print_Mode.setmp [] (Syntax.string_of_term @{{context}}) template;
        val consts = RoughSpec_Utils.const_names_of_term @{{context}} term

        val lemmas = Timeout.apply_physical (Time.fromSeconds 60) (RoughSpec.templatePropsStringInputs @{{context}} prettytemplate) consts
        val result = if List.null lemmas then "empty" else (List.exists (AbstractLemma.match_lemma term) lemmas |> Bool.toString)
        in
            result
        end handle
            Timeout.TIMEOUT _ => "timeout"
        \\<close>"""
    thy = temp_theory(
        working_directory=configs.root_dir,
        queries=[query2],
        imports=configs.imports + [os.path.join(configs.root_dir, path, base_name)],
        name=new_thy_name,
    )
    return thy


if __name__ == "__main__":
    configs = Namespace(
        dataset_name="yalhessi/lemexp",
        dataset_config="hol-thms-v2-by-file",
        root_dir="/public/yousef/lemma-exploration/isabelle/src/HOL",
        imports=[
            "/public/yousef/lemma-exploration/thys/ExtractLemmas",
            "/public/yousef/lemma-exploration/thys/RoughSpec",
        ],
        debug=False,
        upload=True,
    )
    print(configs)

    test_set = load_dataset(configs, split="test")
    isabelle = IsabelleConnector(name="lemexp", working_directory=configs.root_dir)
    rediscover_thys = [rediscover(row, configs) for row in test_set]
    rediscover_data = isabelle.use_theories(
        rediscover_thys, batch_size=10, recache=False, rerun_failed=False
    )
    values = [str(x[0]) if x else "failed" for x in rediscover_data.values()]

    if configs.upload:
        result_set = test_set.add_column("rediscoverable", values)
        today = datetime.now().strftime("%Y-%m-%d")
        label = f"{configs.dataset_config}-rediscovery-{today}"
        result_set.push_to_hub("yalhessi/roughspec-results", config_name=label)
        print("Results uploaded to the hub")
