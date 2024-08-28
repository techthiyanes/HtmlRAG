import argparse
import re
from typing import List

import loguru
import torch
from prometheus_client import make_asgi_app
from starlette.routing import Mount
import fastapi
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM


class OpenAIBaseModel(BaseModel):
    # OpenAI API does not allow extra fields
    model_config = ConfigDict(extra="forbid")


class HTMLTreeTrimRequest(OpenAIBaseModel):
    htmls: List[List[str]]
    question: List[str]
    model: str


app = fastapi.FastAPI()

# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)

TIMEOUT_KEEP_ALIVE = 5


@app.post("/html_tree_trim")
async def create_chat_completion(request: HTMLTreeTrimRequest, ):
    try:
        htmls = request.htmls
        question = request.question
        html_res = model.generate_html_tree(tokenizer, question, htmls)
    except Exception as e:
        loguru.logger.error(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    return JSONResponse(content=html_res)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--port", type=int, default=8000)
    argparser.add_argument("--host", type=str, default="0.0.0.0")
    argparser.add_argument("--uvicorn_log_level", type=str, default="info")
    argparser.add_argument("--ckpt_path", type=str, default="../../huggingface/glm-4-9b-chat-1m")
    args = argparser.parse_args()
    ckpt_path = args.ckpt_path
    # ckpt_path="/cpfs01/shared/public/guopeidong/models/glm4-9b/glm4-9b-128k-v0701-node2/checkpoint-1554"
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    # model=AutoModelForCausalLM.from_pretrained("../../../huggingface/glm-4-9b-chat-1m",trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path, trust_remote_code=True)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device).eval()

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
