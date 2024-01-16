import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import PeftModel, PeftConfig


def load_model(model_name_or_path):
    try:
        peft_config = PeftConfig.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).cuda()
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path,
                                                  trust_remote_code=True, use_fast=False)

        model = PeftModel.from_pretrained(model, model_name_or_path)
        model.eval()
        base_model_name_or_path = model.peft_config['default'].base_model_name_or_path
    except:

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).cuda()
        model = model.eval()
        base_model_name_or_path = model_name_or_path

    return model, tokenizer, base_model_name_or_path
