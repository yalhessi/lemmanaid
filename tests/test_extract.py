from isabelle_connector.isabelle_connector import IsabelleConnector
from isabelle_connector.isabelle_utils import get_theory, list_theory_files
from loguru import logger
import pytest

from lemma_exploration.config import PROJ_ROOT
from lemma_exploration.extract import extract_transitions, transitions_theory
from lemma_exploration.utils import getenv
from argparse import Namespace

import nest_asyncio
nest_asyncio.apply()

def test_extract_transitions_one_file():
    configs = Namespace(
        root_dir=getenv("HOL"),
        imports=[
            PROJ_ROOT / "thys/ExtractLemmas",
            PROJ_ROOT / "thys/RoughSpec",
        ],    
        batch_size = 1,
    )
    isabelle = IsabelleConnector(
        name="extraction", session_name="HOL", working_directory=configs.root_dir
    )
    thy_files = list_theory_files(configs.root_dir)[0:1]  # Limit to one file for testing

    src_thys = [get_theory(theory_file, configs.root_dir) for theory_file in thy_files]
    extraction_thys = [transitions_theory(thy, configs) for thy in src_thys]
    data, errs = isabelle.use_theories(extraction_thys, batch_size=1000, rm_if_temp=False)
    values = [item for item in data.values() if item]
    logger.info(f"Extracted data: {values}")
    logger.info(f"Errors: {errs}")
    assert len(values) > 0, "No data extracted from theories"

def test_extract_transitions_all_files():
    configs = Namespace(
        root_dir=getenv("HOL"),
        imports=[
            PROJ_ROOT / "thys/ExtractLemmas",
            PROJ_ROOT / "thys/RoughSpec",
        ],    
        batch_size = 1,
    )
    isabelle = IsabelleConnector(
        name="extraction", session_name="HOL", working_directory=configs.root_dir
    )
    thy_files = list_theory_files(configs.root_dir)

    src_thys = [get_theory(theory_file, configs.root_dir) for theory_file in thy_files]
    extraction_thys = [transitions_theory(thy, configs) for thy in src_thys]
    data, errs = isabelle.use_theories(extraction_thys, batch_size=1000, rm_if_temp=False)

    values = [item for item in data.values() if item]
    assert len(values) > 0, "No data extracted from theories"