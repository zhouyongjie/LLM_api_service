import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(root_dir)

from contextlib import asynccontextmanager
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from api.interface import *
from api.chat.utils import chat_stream_generator, chat_generator
from api.logs.log_units import get_logger
from api.chat.load_model import load_model

logger = get_logger(__name__,)


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Starting chat-api server...")
model, tokenizer, base_model_name = None, None, None
root_model_hub_path = "/app/ai_platform/model_hub/"


def switch_model(request: ChatCompletionRequest):
    request_model_name = request.model.split("/")[-1]
    global model, tokenizer, base_model_name

    if model is None or tokenizer is None:

        del model, tokenizer, base_model_name
        torch.cuda.empty_cache()
        logger.info(f"Loading new model from {root_model_hub_path + request.model}")
        model, tokenizer, base_model_name = load_model(root_model_hub_path + request.model)
        logger.info(f"Load model success")

    else:
        running_model_name = model.config.name_or_path.split("/")[-1]
        if request_model_name != running_model_name:
            del model, tokenizer, base_model_name
            torch.cuda.empty_cache()
            logger.info(f"Loading new model from {root_model_hub_path + request.model}")
            model, tokenizer, base_model_name = load_model(root_model_hub_path + request.model)
            logger.info(f"Load model success")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):

    global model, tokenizer, base_model_name

    logger.debug(f"Chat request: {request.model_dump_json(exclude_unset=True)}")

    try:
        switch_model(request)
    except Exception as e:
        logger.error(f"Load model error: {str(e)}")
        model, tokenizer, base_model_name = None, None, None

    if model is None or tokenizer is None:
        raise HTTPException(status_code=404, detail="no model or tokenizer")
    if request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid message, last role shouldn't be 'assistant'")

    # with_function_call = bool(
    #     request.messages[0].role == "system" and request.messages[0].tools is not None
    # )

    # stop settings
    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    request.stop_token_ids = request.stop_token_ids or []

    chat_id = str(uuid.uuid4().hex)
    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        chunk=request.chunk,
        # stop_token_ids=request.stop_token_ids,
        stop=request.stop,
        # repetition_penalty=request.repetition_penalty,
        # with_function_call=with_function_call
    )

    if request.stream:
        generate = stream_predict(chat_id, request.model, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")

    else:
        response = predict(chat_id, request.model, gen_params)
        return response


def predict(chat_id, model_id: str, params: dict) -> ChatCompletionResponse:

    global model, tokenizer, base_model_name

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=404, detail="model and tokenizer not available")

    new_response = chat_generator(model, tokenizer, base_model_name, params)

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=DeltaMessage(role='assistant', content=new_response['text']),
        finish_reason='stop'
    )
    usage = UsageInfo()
    task_usage = UsageInfo.model_validate(new_response['usage'])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    response = ChatCompletionResponse(
        id=chat_id,
        model=model_id,
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )

    logger.info(f"Assistant response: {response.model_dump_json()}")
    return response


async def stream_predict(chat_id, model_id: str, params: dict):
    global model, tokenizer, base_model_name

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=404, detail="model and tokenizer not available")

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )

    chunk = ChatCompletionResponse(
        id=chat_id,
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk"
    )

    logger.info(f"Base model name: {base_model_name}")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    logger.info(f"First chunk: {chunk.model_dump_json(exclude_unset=True)}")

    previous_text = ""

    for new_response in chat_stream_generator(model, tokenizer, base_model_name, params):
        decoded_unicode = new_response["text"]

        content = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode

        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(content=content),
            finish_reason=None
        )
        chunk = ChatCompletionResponse(
            id=chat_id,
            model=model_id,
            choices=[choice_data],
            object="chat.completion.chunk"
        )

        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )

    chunk = ChatCompletionResponse(
        id=chat_id,
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    # yield '[DONE]'
    logger.info(f"Assistant response: {previous_text}")
    logger.info(f"Last chunk: {chunk.model_dump_json(exclude_unset=True)}")


if __name__ == "__main__":
    logger.info("Starting chat-api server...")
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # from transformers import PreTrainedModel, PreTrainedTokenizer
    # logger.info("Load init model: chatglm3-6b")
    # model_path = "/app/ai_platform/model_hub/THUDM/chatglm3-6b"
    # logger.info("Load init model: Qwen/Qwen-7B-Chat")
    # model_path = "/app/ai_platform/model_hub/Qwen/Qwen-7B-Chat"
    # logger.info("Load init model: baichuan-inc/Baichuan2-7B-Chat")
    # model_path = "/app/ai_platform/model_hub/baichuan-inc/Baichuan2-7B-Chat"
    # tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    # model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
    #                                                               torch_dtype=torch.bfloat16).cuda()
    # base_model_name: str = model_path.split("/")[-1]
    model, tokenizer, base_model_name = None, None, None
    uvicorn.run(app, host='0.0.0.0', port=3101)
