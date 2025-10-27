import datetime
import os
import re
from argparse import Namespace

import pandas
from datasets import Dataset
from isabelle_connector.isabelle_connector import (
    IsabelleConnector,
    Theory,
)
from isabelle_connector.isabelle_utils import get_theory, list_theory_files, temp_theory

from lemma_exploration.config import INTERIM_DATA_DIR, PROJ_ROOT
from lemma_exploration.utils import path_to_theory_name
from typer import Typer

app = Typer()

def parse_args(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Isabelle connector")
    parser.add_argument("--root-dir", type=str, default="/isabelle/src/HOL")
    parser.add_argument(
        "--imports",
        nargs="*",
        default=[
            "/Users/yousef/lemma-exploration/thys/ExtractLemmas",
            "/Users/yousef/lemma-exploration/thys/RoughSpec",
        ],
    )
    parser.add_argument("--split", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--upload", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--dataset-name", default="yalhessi/lemexp")
    parser.add_argument("--dataset-config", default="extracted-thms")
    return parser.parse_args(args)


def theorem_name(lemma):
    command = r"(lemma|theorem)"
    patterns = [
        rf"{command}\s*\"(.*)\"",
        rf"{command}\s+(\(.*\)\s+)?([\w\']+)\s*(\[.*\])?:(.*)",
        rf"{command}\s*(.*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, lemma, re.DOTALL)
        if match:
            if pattern == patterns[0]:
                return None
            elif pattern == patterns[1]:
                name, body = match.group(3), match.group(5).strip()
                if re.match(r"\".*\"", body):
                    body = body[1:-1]
                return name
            elif pattern == patterns[2]:
                return None

    print(f"Failed to parse lemma: {lemma}")
    return None


def definition_name(lemma):
    # name is
    # definition (*+*)? (+)? name = ...
    # definition (*+*)? (+)? name :: ...
    # definition (*+*)? (+)? name where ...
    command = (
        r"(definition|fun|primrec|primcorec|datatype|codatatype|function|typedecl|typedef|"
        r"type_synonym|record|inductive|coinductive|abbreviation|class|inductive_set|consts)"
    )

    patterns = [
        rf"{command}\s+(\(.*\)\s+)?([\w\']+)\s*(\[.*\])?:",
        rf"{command}\s+(\(.*\)\s+)?([\w\']+)\s*(\[.*\])?where",
        rf"{command}\s+(\(.*\)\s+)?([\w\']+)\s*(\[.*\])?=",
        rf"{command}\s+(\(.*\)\s+)?([\w\']+)\s*(\[.*\])?::",
        rf"datatype\s+'a\s+([\w\']+)",
    ]

    for pattern in patterns:
        result = re.search(pattern, lemma, re.DOTALL)
        if result:
            if len(result.groups()) > 2:
                return result.group(3).strip('"')
            else:
                return result.group(1).strip('"')

    return lemma.split()[1].strip('"')


def definition_of_symbol(symbol, defs):
    if symbol.count(".") == 1:
        if symbol in defs:
            return defs[symbol]
    elif symbol.count(".") == 2:
        # keep first.second
        real_symbol = symbol.rsplit(".", 1)[0]
        if real_symbol in defs:
            return defs[real_symbol]
        elif real_symbol.endswith("_class") and real_symbol[:-6] in defs:
            return defs[real_symbol[:-6]]


def dependent_definitions(symbols, defs):
    dependents = []
    for symbol in symbols:
        definition = definition_of_symbol(symbol, defs)
        if definition:
            dependents.append(definition)
    return dependents


def extract_transitions_of_kinds(transitions, kinds, extract_fn) -> dict:
    result = {}
    for name, data in transitions.items():
        for src_name, trans in data:
            result[src_name] = []
            for kind, stmt in trans:
                if kind in kinds:
                    extracted_name = extract_fn(stmt)
                    result[src_name].append((extracted_name, stmt))
    return result


def extract_theorems(transitions) -> dict[str, dict[str, str]]:
    thm_kinds = ["theorem", "lemma"]
    thms = extract_transitions_of_kinds(transitions, thm_kinds, theorem_name)
    return {name: {thm_name: stmt for thm_name, stmt in thms[name]} for name in thms}


def extract_definitions(transitions) -> dict[str, list[tuple[str, str]]]:
    def_kinds = [
        "definition",
        "fun",
        "primrec",
        "primcorec",
        "datatype",
        "codatatype",
        "function",
        "typedecl",
        "typedef",
        "type_synonym",
        "record",
        "inductive",
        "coinductive",
        "abbreviation",
        "class",
        "inductive_set",
        "consts",
    ]
    return extract_transitions_of_kinds(transitions, def_kinds, definition_name)

def flatten_definitions(defs: dict[str, list[tuple[str, str]]]) -> dict[str, str]:
    """
    Flattens a nested dictionary of definitions into a single dictionary with 
    keys formatted as "theory_name.definition_name".

    Args:
        defs (dict): A dictionary where keys are theory names (strings) and 
                     values are lists of tuples. Each tuple contains a 
                     definition name (string) and its corresponding definition 
                     statement.

    Returns:
        dict: A flattened dictionary where keys are formatted as 
              "theory_name.definition_name" and values are the corresponding 
              definition statements.

    Notes:
        - If a theory name contains a "/", only the part after the "/" is used 
          in the resulting key.
    """
    all_defs = {}
    for thy_name in defs:
        for name, def_stmt in defs[thy_name]:
            if "/" in thy_name:
                split_name = thy_name.split("/")
                thy_name = split_name[1]
            all_defs[f"{thy_name}.{name}"] = def_stmt
    return all_defs


def local_symbols_of_file(template_data):
    local_symbols = {}
    for thy in template_data:
        if not template_data[thy] or not isinstance(template_data[thy][0], list):
            continue
        if len(template_data[thy]) != 1:
            print(f"Multiple results for {thy}")
            # print(template_data[thy])
            continue
        
        # if len(template_data[thy][0]) != 6:
        #     print(f"Invalid result for {thy}")
        #     print(template_data[thy][0])
        #     continue
        for (
            file_name,
            theorem_name,
            theorem_statement,
            dep_symbols,
            typs,
            template,
        ) in template_data[thy][0]:
            if file_name in local_symbols:
                local_symbols[file_name] |= set(dep_symbols)
            else:
                local_symbols[file_name] = set(dep_symbols)
    return local_symbols


def template_extraction_theory(src_thy: Theory, configs: Namespace) -> Theory:
    name = src_thy.name
    path, base_name = name.rsplit("/", 1) if "/" in name else ("", name)

    new_thy_name = f"Extract_{path_to_theory_name(name)}"
    query = f"""
        let
            val thms = Extract_Lemmas.get_all_thms "{base_name}" @{{context}}
            val results = map (fn (name, thm) => 
            let 
                val term = Thm.prop_of thm
                val template = AbstractLemma.abstract_term @{{context}} term
                val template_str = Print_Mode.setmp [] (Syntax.string_of_term @{{context}}) template
                val symbols = RoughSpec_Utils.const_names_of_term @{{context}} term
            in
            (
                "{name}",
                name,
                thm,
                symbols,
                template_str
            )
            end) thms;
        in
            results
        end"""
    thy = temp_theory(
        name=new_thy_name,
        imports=configs.imports + [name],
        # working_directory=src_thy.working_directory,
        working_directory=os.path.join(
            INTERIM_DATA_DIR, os.path.basename(configs.root_dir)
        ),
    )
    thy.add_ml_block(query)
    return thy

def template_and_type_extraction_theory(src_thy: Theory, configs: Namespace) -> Theory:
    name = src_thy.name
    path, base_name = name.rsplit("/", 1) if "/" in name else ("", name)
    if path:
        session = "-".join(["HOL"] + path.split("/"))
    else:
        session = "HOL"
    import_name = f"{session}.{base_name}"
    
    new_thy_name = f"Extract_{path_to_theory_name(name)}"
    query = f"""
        let
            fun type_of_const symbol =
              let 
                val t = Syntax.read_term @{{context}} symbol
                val typ = Term.type_of t
              in 
                typ
              end 

            val thms = Extract_Lemmas.get_all_thms "{base_name}" @{{context}}
            val results = map (fn (name, thm) => 
            let 
                val term = Thm.prop_of thm
                val template = AbstractLemma.abstract_term @{{context}} term
                val template_str = Print_Mode.setmp [] (Syntax.string_of_term @{{context}}) template
                val symbols = RoughSpec_Utils.const_names_of_term @{{context}} term
                val typs = map (type_of_const) symbols
            in
            (
                "{name}",
                name,
                thm,
                symbols,
                typs,
                template_str
            )
            end) thms;
        in
            results
        end"""
    thy = temp_theory(
        name=new_thy_name,
        imports=configs.imports + [import_name],
        # working_directory=src_thy.working_directory,
        working_directory=os.path.join(
            INTERIM_DATA_DIR, os.path.basename(configs.root_dir)
        ),
    )
    thy.add_ml_block(query)
    return thy


def transitions_theory(thy: Theory, configs) -> Theory:
    """
    Get theorems from a theory.

    :param theory_name: name of the theory
    :returns: theorems from the theory
    """
    new_thy_name = f"Transitions_{path_to_theory_name(thy.name)}"
    query = f"""
            let
                val filename = "{thy.working_directory}/{thy.name}.thy"
                val stream = TextIO.openIn filename
                val content = TextIO.inputAll stream
                val theory = @{{theory}}
                val transitions = Extract.parse_text theory content
                val results = map (fn (trans, string) => (Toplevel.name_of trans, string)) transitions;
            in
                ("{thy.name}", results)
            end"""

    thy = temp_theory(
        name=new_thy_name,
        imports=configs.imports,
        # working_directory=configs.root_dir,
        working_directory=os.path.join(
            INTERIM_DATA_DIR, os.path.basename(configs.root_dir)
        ),
    )
    thy.add_ml_block(query)
    return thy


def extract_data(
    extraction_fn,
    batch_size: int,
    configs: Namespace,
    **kwargs,
):
    isabelle = IsabelleConnector(
        name="extraction", session_name="HOL", working_directory=configs.root_dir
    )
    thy_files = list_theory_files(configs.root_dir)

    src_thys = [get_theory(theory_file, configs.root_dir) for theory_file in thy_files]
    extraction_thys = [extraction_fn(thy, configs) for thy in src_thys]
    data, errs = isabelle.use_theories(extraction_thys, batch_size=batch_size, **kwargs)

    print(
        f"Successfully extracted data from {len([item for item in data.values() if item])} / {len(data)} theories"
    )
    return data


def extract_transitions(configs: Namespace, **kwargs):
    return extract_data(transitions_theory, batch_size=1000, configs=configs, **kwargs)


def extract_templates(configs: Namespace, **kwargs):
    return extract_data(
        template_extraction_theory, batch_size=1, configs=configs, **kwargs
    )

@app.command()
def new_extract(
    root_dir: str,
    imports: list[str] = [
        PROJ_ROOT/"thys/ExtractLemmas",
        PROJ_ROOT/"thys/RoughSpec",
    ],
    batch_size: int = 1,
    upload: bool = False,
    dataset_name: str = "yalhessi/lemexp",
    dataset_config: str = "extracted-thms",
):
    configs = Namespace(
        root_dir=root_dir,
        imports=imports,
        batch_size=batch_size,
        upload=upload,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )
    templates = extract_templates(configs)
    

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    app()
    # configs = parse_args()
    # transitions = extract_transitions(configs)
    # template_data = extract_templates(configs)

    # local_symbols = local_symbols_of_file(template_data)
    # thms = extract_theorems(transitions)
    # defs = extract_definitions(transitions)

    # all_defs = {}
    # for thy_name in defs:
    #     for name, def_stmt in defs[thy_name]:
    #         if "/" in thy_name:
    #             split_name = thy_name.split("/")
    #             thy_name = split_name[1]
    #         # else:
    #         # thy_name = split_name[0]
    #         all_defs[f"{thy_name}.{name}"] = def_stmt

    # table = []
    # for key in template_data:
    #     if template_data[key]:
    #         if len(template_data[key]) != 1 and len(template_data[key][-1]) != 5:
    #             continue

    #         for src_name, thm_name, thm, symbols, template in template_data[key][-1]:
    #             # ignore lemmas generated from definitions
    #             if thm_name.count(".") > 1:
    #                 continue

    #             thm_short_name = thm_name.split(".")[-1]
    #             if "(" in thm_short_name:
    #                 thm_short_name = thm_short_name.split("(")[0]
    #             thm_command = thms[src_name].get(thm_short_name, None)

    #             used_symbols = set(symbols)
    #             file_local_symbols = (
    #                 local_symbols[src_name] if src_name in local_symbols else set()
    #             )
    #             filtered_symbols = list(file_local_symbols - used_symbols)

    #             used_defs = dependent_definitions(symbols, all_defs)
    #             local_defs = [d[1] for d in defs[src_name]] if src_name in defs else []
    #             const_defs = local_defs + used_defs

    #             table.append(
    #                 {
    #                     "theory_file": src_name,
    #                     "lemma_name": thm_name,
    #                     "lemma_object": thm,
    #                     "lemma_command": thm_command,
    #                     "used_symbols": symbols,
    #                     "local_symbols": filtered_symbols,
    #                     "used_defs": used_defs,
    #                     "local_defs": local_defs,
    #                     "defs": const_defs,
    #                     "template": template.replace("\n", " "),
    #                 }
    #             )

    # df = pandas.DataFrame.from_records(table)
    # dataset = Dataset.from_pandas(df)

    # if configs.upload:
    #     today = datetime.datetime.today().strftime("%Y-%m-%d")
    #     label = f"{configs.dataset_config}-{today}"
    #     dataset.push_to_hub("yalhessi/lemexp-raw", config_name=label, split="train")
    #     dataset.push_to_hub(
    #         "yalhessi/lemexp-raw", config_name=configs.dataset_config, split="train"
    #     )
