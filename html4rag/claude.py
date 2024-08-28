#!/usr/bin/env python3

from loguru import logger
import json
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import requests
from langchain.llms.base import LLM
import anthropic

from oai import openai_payload, OpenAIApiException, OpenAIClient, OpenAIApiProxy

__all__ = ["Claude", "ClaudeClient", "ClaudeProxy"]


class Claude(LLM):

    model: str = "claude-v1-32k"
    url: str = "https://api.anthropic.com"
    api_keys: List[str] = None
    system_prompt = ""

    repetition_penalty: float = 1.0
    temperature: float = None
    top_p: float = None
    seed: int = None
    json_mode: bool = False
    logprobs: bool = False
    top_logprobs: int = 1

    @property
    def _llm_type(self) -> str:
        return self.model

    def predict(  # type: ignore[override]
        self, text: str, *,
        stop: Optional[Sequence[str]] = None,
        model: str = "claude-3-opus-20240229",
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        json_mode: bool = False,
        seed: Optional[int] = None,
        verbose: bool = True,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> str:
        try:
            if isinstance(self.api_keys, list):
                api_key = random.choice(self.api_keys)
            else:
                api_key = self.api_keys
            assert len(api_key) > 0 and isinstance(api_key, str)
            client = anthropic.Anthropic(base_url=self.url, api_key=api_key)

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": text}
                ]
            )
        except Exception as e:
            logger.info(f"error: {e} with api_key: {api_key}, remove it")
            self.api_keys.remove(api_key)

        answer = response.content[0].text
        return answer

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs):
        return self.predict(prompt, stop=stop, **kwargs)


class ClaudeClient(LLM):
    model: str = "claude-3-opus-20240229"
    url: str = "https://api.anthropic.com/v1/messages"
    api_keys: Union[List[str], str] = "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"
    system_prompt = ""

    repetition_penalty: float = 1.0
    temperature: Optional[float] = None
    max_tokens: int = 2048
    response: Optional[Dict] = None

    @property
    def _llm_type(self) -> str:
        return self.model

    def predict(  # type: ignore[override]
        self, text: Union[List[str], str], *,
        stop: Optional[Sequence[str]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_keys,
        }

        if model is None:
            model = self.model
        if not system_prompt:
            system_prompt = self.system_prompt
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        keyword_args: Dict[str, Any] = {"max_tokens": max_tokens}
        if temperature:
            keyword_args["temperature"] = temperature

        payload = openai_payload(text, model, system_prompt=system_prompt, **keyword_args)

        response = requests.post(self.url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            err_msg = f"access openai error, status code: {response.status_code}，errmsg: {response.text}"
            raise OpenAIApiException(err_msg, response.status_code)

        data = response.json()
        self.response = data
        return data["content"][0]["text"]

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs):
        return self.predict(prompt, stop=stop, **kwargs)


class ClaudeProxy(OpenAIClient):

    model = "claude-3-opus-20240229"
    choices = ["claude-3-opus-20240229"]
    system_prompt = ""
    _proxy = OpenAIApiProxy(
        url="http://claude-oneapi.baichuan.svc/v1/chat/completions",
        api_key="sk-7QSO9coE7T3oOAhV4797799b02C043EeB8C565905024BeAe",
        # api_key="sk-EF0XTgLIhuVxLoPv1c21746695044636Ae177bF895C4Ca15",
    )

    def reduce_concurrency(self, low: int = 3, high: int = 7):
        time.sleep(random.randint(low, high))
