from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from peft.peft_model import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


@dataclass
class HuggingFaceModel:
    load_model_from: Literal["web", "cache", "web-peft"] = "web"
    is_trainable: bool = False
    use_quantization: bool = True
    use_peft: bool = True

    model: Any = None
    tokenizer: Any = None
    model_name: str = ""
    base_model_name: Optional[str] = None  # For web-peft

    # cached model
    output_dir: str = field(default_factory=str)

    def __post_init__(self):
        self._load_model()

    def _load_model(self):
        if self.load_model_from == "cache":
            self.load_cached_model()
        elif self.load_model_from == "web":
            self.load_web_model()
        elif self.load_model_from == "web-peft":
            self.load_finetuned_web_model()
        else:
            raise ValueError(
                f"Invalid value for load_model_from: {self.load_model_from}"
            )

        if self.is_trainable:
            if self.use_quantization:
                self.prepare_for_quantization()

            if self.use_peft:
                self.prepare_for_peft()

    def load_cached_model(self):
        self.load_web_model()
        # self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        self.model = PeftModel.from_pretrained(
            model=self.model,
            model_id=self.output_dir,
            is_trainable=self.is_trainable,
        )

    def load_web_model(self):
        quantization_config = (
            BitsAndBytesConfig(
                load_in_8bit=True,
                # llm_int8_has_fp16_weight=True,
            )
            if self.use_quantization
            else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            # torch_dtype=torch.bfloat16,
        )

    def load_finetuned_web_model(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
        self.model = PeftModel.from_pretrained(base_model, self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)

    def prepare_for_quantization(self):
        from peft.utils.other import prepare_model_for_kbit_training

        self.model = prepare_model_for_kbit_training(self.model)

    def prepare_for_peft(self):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config).to(self.model.device)
        print_trainable_parameters(self.model)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {
            all_param
        } || trainable%: {100 * trainable_params / all_param}"
    )


def get_predictor(args, model, tokenizer, **kwargs):
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        return_full_text=False,
        truncation=True,
        **kwargs,
    )


def predict(model, tokenizer, input_text):
    input_text_tokenized = tokenizer.encode(
        input_text, truncation=True, padding=True, return_tensors="pt"
    )
    prediction = model.generate(input_text_tokenized)
    return tokenizer.decode(prediction[0])


def test_model(args, model, tokenizer, input_text):
    generator = get_predictor(args, model, tokenizer)
    print(f"Pipeline: {generator(input_text)}")
    print(f"Enc-dec: {predict(model, tokenizer, input_text)}")
