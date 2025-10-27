import argparse
import os
from abc import abstractmethod

from dotenv import load_dotenv

from lemma_exploration.utils import get_run_id


class LemExpCLIArgs:
    @classmethod
    def parse_args(cls, **kwargs):
        load_dotenv()
        parser = argparse.ArgumentParser(**kwargs)
        LemExpCLIArgs.shared_args(parser)
        cls.additional_args(parser)
        args = parser.parse_args()
        if not args.output_dir:
            output_dir = os.getenv("OUTPUT_DIR", ".")
            args.output_dir = os.path.join(
                output_dir,
                get_run_id(args.model_name, args.dataset_name, args.dataset_config),
            )
        return args

    @staticmethod
    def shared_args(parser):
        parser.add_argument("--output-dir", type=str, help="output directory")
        parser.add_argument("--max-seq-length", type=int, default=1024)
        parser.add_argument(
            "--use-cache", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument(
            "--debug", action=argparse.BooleanOptionalAction, default=False
        )

    @staticmethod
    def model_args_parser():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--model-name",
            type=str,
            help="model name",
        )
        parser.add_argument(
            "--use-quantization", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument(
            "--use-peft", action=argparse.BooleanOptionalAction, default=True
        )
        parser.add_argument(
            "--use-cached-model", action=argparse.BooleanOptionalAction, default=False
        )
        return parser

    @staticmethod
    def dataset_args_parser():
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--dataset-name",
            type=str,
            help="dataset name",
        )
        parser.add_argument(
            "--dataset-config",
            type=str,
            help="dataset config",
        )
        return parser

    @staticmethod
    @abstractmethod
    def additional_args(parser):
        pass
