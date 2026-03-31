import torch
import json
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from app.schemas import ReviewAnalysisModel, SentimentEnum, AspectsModel

class ABSAPredictor:
    def __init__(self, base_model_id: str, adapter_path: str, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.mock_mode:
            print("⚠️ WARNING: Running in MOCK MODE. Model is not loaded.")
            return

        print(f"🚀 Initializing model on {self.device}...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config if self.device == "cuda" else None,  # Защита от запуска на Mac без CUDA
            device_map="auto",
            trust_remote_code=True
        )

        print(f"Loading LoRA adapters from {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.eval()

        self.prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

    def predict(self, text: str) -> Optional[ReviewAnalysisModel]:
        if self.mock_mode:
            # Возвращаем фейковый ответ для теста API на Mac
            return ReviewAnalysisModel(
                sentiment=SentimentEnum.NEUTRAL,
                aspects=AspectsModel(
                    delivery=SentimentEnum.NEUTRAL,
                    price=SentimentEnum.NEUTRAL,
                    quality=SentimentEnum.NEUTRAL,
                    functionality=SentimentEnum.NEUTRAL,
                    service=SentimentEnum.NEUTRAL
                ),
                summary="[MOCK] Тестовый ответ, модель не загружена."
            )

        instruction = "Извлеки аспекты (delivery, price, quality, functionality, service) и тональность в формате JSON."
        prompt = self.prompt_template.format(instruction, text)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = decoded_output.split("### Response:")[-1].strip()

        return self._validate_json(response_text)

    def _validate_json(self, raw_json: str) -> Optional[ReviewAnalysisModel]:
        try:
            raw_json = raw_json.replace("```json", "").replace("```", "").strip()

            start = raw_json.find('{')
            end = raw_json.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = raw_json[start:end]
            else:
                json_str = raw_json

            data = json.loads(json_str)
            return ReviewAnalysisModel(**data)

        except Exception as e:
            print(f"Validation Error: {e}\nRaw: {raw_json}")
            return None
