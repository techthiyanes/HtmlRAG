#!/usr/bin/env python3

import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import requests
from langchain.llms.base import LLM
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from retrying import retry

OPENAI_INNER_IP = "http://47.236.144.103/v1/chat/completions"

__all__ = [
    "OpenAIClient",
    "OpenAIApiException",
    "OpenAIApiProxy",
    "openai_payload",
]


def openai_payload(
    prompt: Union[List[str], str],
    model_name: str,
    system_prompt: Union[List[str], str] = "",
    **kwargs
) -> Dict:
    """Generate payload for openai api call."""
    messages = []
    if system_prompt:
        if isinstance(system_prompt, str):
            system_prompt = [system_prompt]
        for sys_prompt in system_prompt:
            messages.append({"role": "system", "content": sys_prompt})
    if isinstance(prompt, str):
        prompt = [prompt]
    for idx, p in enumerate(prompt):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append({"role": role, "content": p})

    payload = {"model": model_name, "messages": messages, **kwargs}
    return payload


class OpenAIApiException(Exception):
    def __init__(self, msg, error_code):
        self.msg = msg
        self.error_code = error_code


class OpenAIApiProxy:

    def __init__(self, url: str = None, api_key: str = None):
        if url is None:
            url = OPENAI_INNER_IP
        retry_strategy = Retry(
            total=1,  # 最大重试次数（包括首次请求）
            backoff_factor=1,  # 重试之间的等待时间因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码列表
            allowed_methods=["POST"],  # 只对POST请求进行重试
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # 创建会话并添加重试逻辑
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.api_key = api_key
        self.url = url

    @retry(stop_max_attempt_number=3)
    def call(
        self, prompt: str, system_prompt: str = "",
        model_name: str = "gpt-4",
        verbose: bool = False,
        post_by_session: bool = True,
        **kwargs,
    ):
        headers = {"Content-Type": "application/json"}
        if isinstance(self.api_key, str):
            headers["Authorization"] = "Bearer " + self.api_key
        elif isinstance(self.api_key, list):
            headers["Authorization"] = "Bearer " + random.choice(self.api_key)

        payload = openai_payload(prompt, model_name, system_prompt=system_prompt, **kwargs)
        if verbose:
            print(f"payload: {payload}")

        if post_by_session:
            response = self.session.post(self.url, headers=headers, data=json.dumps(payload))
        else:
            response = requests.post(self.url, headers=headers, data=payload)

        if response.status_code != 200:
            err_msg = f"access openai error, status code: {response.status_code}，errmsg: {response.text}"
            raise OpenAIApiException(err_msg, response.status_code)
        data = json.loads(response.text)
        return data


class OpenAIClient(LLM):

    model = "gpt-4"
    choices = [
        "gpt-4", "gpt-4-1106-preview", "gpt-4-32k",
        "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct",
        "gpt-4-1106-vision-preview", "gpt-4-0125-preview",
    ]
    system_prompt = ""
    _proxy = OpenAIApiProxy(api_key="sk-iD7PPFqiWwyjG4fU26Bf6f49FdD6423088059c6e7cD8Ca2f")

    repetition_penalty: float = 1.0
    temperature: float = None
    top_p: float = None
    seed: int = None
    json_mode: bool = False
    logprobs: bool = False
    top_logprobs: int = 1

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "openai-api"

    def _call(
        self, prompt: str, stop: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,  # type: ignore[override]
        **kwargs
    ):
        response = self.response_call(prompt, system_prompt=system_prompt, stop=stop, **kwargs)
        return response["choices"][0]["message"]["content"]

    def response_call(
        self, prompt: str, system_prompt: str = None,
        stop: Optional[List[str]] = None, **kwargs
    ):
        if system_prompt is None:
            system_prompt = self.system_prompt
        model_name = kwargs.pop("model_name")
        verbose = kwargs.pop("verbose")

        response = self._proxy.call(
            prompt, system_prompt=system_prompt,
            model_name=model_name, verbose=verbose, **kwargs
        )
        return response

    def predict(  # type: ignore[override]
        self, text: str, *,
        stop: Optional[Sequence[str]] = None,
        model_name: str = None,
        system_prompt: str = None,
        temperature: float = None,
        top_p: float = None,
        json_mode: bool = False,
        seed: int = None,
        verbose: bool = True,
        logprobs: bool = False,
        top_logprobs: int = None,
    ) -> str:  # noqa
        kwargs = {}
        kwargs["verbose"] = verbose
        kwargs["model_name"] = model_name if model_name else self.model
        if temperature or self.temperature:
            kwargs["temperature"] = temperature if temperature else self.temperature
        if top_p or self.top_p:
            kwargs["top_p"] = top_p if top_p else self.top_p
        if seed or self.seed:
            kwargs["seed"] = seed if seed else self.seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if logprobs or self.logprobs:
            kwargs["logprobs"] = logprobs if logprobs else self.logprobs
            num_logprobs = top_logprobs if top_logprobs else self.top_logprobs
            assert num_logprobs >= 0 and num_logprobs <= 5
            kwargs["top_logprobs"] = num_logprobs
            return self.response_call(text, system_prompt=system_prompt, stop=stop, **kwargs)

        if isinstance(text, list):
            return self._call(text, system_prompt=system_prompt, stop=stop, **kwargs)
        return super().predict(text, stop=stop, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    def set_keys(self, key: str):
        self._proxy.api_key = key
