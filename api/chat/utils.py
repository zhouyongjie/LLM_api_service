import gc
from typing import List, Tuple, Dict, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import GenerationConfig

from api.interface import ChatMessage


def trim(response):
    if isinstance(input, str):
        return response.strip()
    else:
        return response


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


@torch.inference_mode()
def generate_stream_chatglm3(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    query, role = messages[-1].content, messages[-1].role
    history = [m.dict(exclude_none=True) for m in messages[:-1]]

    inputs = tokenizer.build_chat_input(query, history=history, role=role)
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    if input_echo_len > model.config.seq_length:
        raise ValueError("input length is larger than model.config.seq_length")

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
        tokenizer.get_command("<|observation|>")
    ]

    gen_kwargs = {
        "max_length": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }

    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    history.append(
        {
            "role": role,
            "content": query
        }
    )

    for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids[:-1]
        else:
            output_ids = total_ids[input_echo_len:-1]

        response = tokenizer.decode(output_ids)
        if response and response[-1] != "�":
            yield {
                "text": trim(response),
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": None,
            }
    # Only last stream result contains finish_reason, we set finish_reason as stop
    # ret = {
    #     "text": trim(""),
    #     "usage": {
    #         "prompt_tokens": input_echo_len,
    #         "completion_tokens": total_len - input_echo_len,
    #         "total_tokens": total_len,
    #     },
    #     "finish_reason": "stop",
    # }
    # yield ret

    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_qwen(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", False)

    query, role = messages[-1].content, messages[-1].role

    def get_system_massage(messages: List[ChatMessage]) -> str:

        system_message = "You are a helpful assistant."
        for message in messages:
            if message.role == "system":
                system_message = message.content
                break
        return system_message

    def get_history(messages: List[ChatMessage]) -> List[Tuple[str, str]]:
        start = 0
        if messages[0].role == "system":
            start = 1
        history = []
        for element1, element2 in zip(messages[start:-1][::2], messages[start:-1][::2][1::2]):
            history.append(tuple([element1.content, element2.content]))
        return history

    system = get_system_massage(messages)
    history = get_history(messages)

    input_echo_len = len(tokenizer.encode(query))

    if input_echo_len > model.config.seq_length:
        raise ValueError("input length is larger than model.config.seq_length")

    gen_kwargs = {
        "max_new_tokens": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }

    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    for response in model.chat_stream(tokenizer, query, history, system, **gen_kwargs):
        total_len = len(tokenizer.encode(response))

        yield {
            "text": trim(response),
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": None,
        }

    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_baichuan2(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    query, role = messages[-1].content, messages[-1].role
    history = [m.dict(exclude_none=True) for m in messages[:-1]]

    input_echo_len = len(tokenizer.encode(query))

    if input_echo_len > model.config.model_max_length:
        raise ValueError("input length is larger than model.config.seq_length")

    gen_kwargs = {
        "assistant_token_id": 196,
        "bos_token_id": 1,
        "do_sample": True if temperature > 1e-5 else False,
        "eos_token_id": 2,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": 0,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "top_k": 5,
        "top_p": top_p,
        "user_token_id": 195
    }

    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    history.append(
        {
            "role": role,
            "content": query
        }
    )

    for response in model.chat(tokenizer, history, stream=True, generation_config=GenerationConfig(**gen_kwargs)):
        total_len = len(tokenizer.encode(response))

        yield {
            "text": trim(response),
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": None,
        }

    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_chatglm2(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", False)

    query, role = messages[-1].content, messages[-1].role

    def get_history(messages: List[ChatMessage]) -> List[Tuple[str, str]]:
        start = 0
        if messages[0].role == "system":
            start = 1
        history = []
        for element1, element2 in zip(messages[start:-1][::2], messages[start:-1][::2][1::2]):
            history.append(tuple([element1.content, element2.content]))
        return history

    history = get_history(messages)

    input_echo_len = len(tokenizer.encode(query))

    if input_echo_len > model.config.seq_length:
        raise ValueError("input length is larger than model.config.seq_length")

    gen_kwargs = {
        "max_length": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }

    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    for response, _ in model.stream_chat(tokenizer, query, history, **gen_kwargs):
        total_len = len(tokenizer.encode(response))

        yield {
            "text": trim(response),
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": None,
        }

    gc.collect()
    torch.cuda.empty_cache()


def chat_stream_generator(model: PreTrainedModel,
                          tokenizer: PreTrainedTokenizer,
                          base_model_name: str,
                          params: dict):
    # model_name = model.config.name_or_path.split("/")[-1]
    if 'chatglm3' in base_model_name:
        generate_function = generate_stream_chatglm3

    elif 'chatglm2' in base_model_name or 'codegeex' in base_model_name:
        generate_function = generate_stream_chatglm2

    elif 'Baichuan2' in base_model_name:
        generate_function = generate_stream_baichuan2

    elif 'Qwen' in base_model_name:
        generate_function = generate_stream_qwen

    else:
        raise ValueError("Invalid model name")

    generator = generate_function(model, tokenizer, params)

    return generator


def chat_generator(model: PreTrainedModel,
                   tokenizer: PreTrainedTokenizer,
                   base_model_name: str,
                   params: dict):
    generator = chat_stream_generator(model, tokenizer, base_model_name, params)

    response = {}
    for response in generator:
        pass

    return response


