"""llm.py – LLM integration layer with auto-detection of best local model.

Provides a unified interface for calling large language models (LLMs)
via Ollama's REST API.  On initialisation, probes the local Ollama
instance to discover available models and selects the best one
according to a preference ranking.

**Default model**: ``qwen3-coder:30b``  (recommended for local compute)
**Fallback**:      ``qwen2.5:7b-instruct``

Model preference ranking (highest to lowest)::

    1. qwen3:235b         – strongest reasoning, needs ~142 GB
    2. qwen3-coder:30b    – strong coding+math, fits in ~19 GB  ★ default
    3. deepseek-r1:8b     – good chain-of-thought reasoning
    4. qwen3-vl:8b        – multimodal, geometry diagram understanding
    5. qwen2.5:7b-instruct – reliable baseline
    6. (any other model)  – last resort

Usage::

    from geometry_agent.llm import get_llm, LLMClient

    llm = get_llm()                       # auto-detect best local model
    llm = get_llm("qwen2.5:7b-instruct") # force specific model
    reply = llm.chat("Prove that AB ∥ EF given AB ∥ CD and CD ∥ EF.")
    print(reply)

The LLM is used by pipeline agents for:
  • **ParserAgent** – natural-language problem parsing
  • **CriticReflectAgent** – failure diagnosis and proof repair suggestions
  • **PlannerAgent** – strategy hinting for complex problems

Author:  Jiangsheng Yu
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# Model preference: higher index = lower priority.
# The auto-detector picks the model with the *lowest* index that is
# available locally.
MODEL_PREFERENCE: List[str] = [
    "qwen3:235b",              # 1 – strongest, huge memory
    "qwen3-coder:30b",         # 2 – ★ recommended default
    "deepseek-r1:8b",          # 3 – chain-of-thought reasoning
    "qwen3-vl:8b",             # 4 – multimodal
    "qwen2.5:7b-instruct",     # 5 – reliable baseline
]

DEFAULT_MODEL = "qwen3-coder:30b"

# ── Data types ───────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """Response from an LLM call.

    Attributes
    ----------
    content : str
        The generated text.
    model : str
        Which model produced this response.
    total_duration_ms : float
        Total round-trip time in milliseconds.
    prompt_tokens : int
        Number of tokens in the prompt (if reported).
    completion_tokens : int
        Number of tokens in the completion (if reported).
    raw : dict
        The full JSON response from the API.
    """
    content: str
    model: str = ""
    total_duration_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Metadata for an available Ollama model."""
    name: str
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    size_bytes: int = 0


# ── Ollama API helpers ───────────────────────────────────────────────


def _ollama_request(
    endpoint: str,
    payload: Optional[dict] = None,
    *,
    timeout: int = 120,
    stream: bool = False,
) -> dict:
    """Make a request to the Ollama REST API.

    Parameters
    ----------
    endpoint : str
        API path, e.g. ``"/api/tags"`` or ``"/api/chat"``.
    payload : dict, optional
        JSON body for POST requests.
    timeout : int
        Request timeout in seconds.
    stream : bool
        Whether to use streaming mode.

    Returns
    -------
    dict
        Parsed JSON response.
    """
    url = f"{OLLAMA_BASE_URL}{endpoint}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
    else:
        req = Request(url, method="GET")

    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")

    if not body.strip():
        return {}

    # Ollama streaming mode returns newline-delimited JSON;
    # we only care about the final message.
    if stream:
        lines = body.strip().split("\n")
        return json.loads(lines[-1])

    return json.loads(body)


def list_local_models() -> List[ModelInfo]:
    """Query Ollama for all locally available models.

    Returns
    -------
    list[ModelInfo]
        Available models, or empty list if Ollama is unreachable.
    """
    try:
        resp = _ollama_request("/api/tags", timeout=5)
    except (URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot reach Ollama at %s: %s", OLLAMA_BASE_URL, exc)
        return []

    models: List[ModelInfo] = []
    for m in resp.get("models", []):
        details = m.get("details", {})
        models.append(ModelInfo(
            name=m.get("name", m.get("model", "")),
            family=details.get("family", ""),
            parameter_size=details.get("parameter_size", ""),
            quantization=details.get("quantization_level", ""),
            size_bytes=m.get("size", 0),
        ))
    return models


def detect_best_model(
    preference: Optional[List[str]] = None,
    fallback: str = DEFAULT_MODEL,
) -> str:
    """Auto-detect the best locally available LLM.

    Scans Ollama for available models and returns the highest-ranked
    one according to ``MODEL_PREFERENCE``.

    Parameters
    ----------
    preference : list[str], optional
        Custom preference order.  Defaults to ``MODEL_PREFERENCE``.
    fallback : str
        Model name to return if no preferred model is found.

    Returns
    -------
    str
        Model name (e.g. ``"qwen3-coder:30b"``).
    """
    pref = preference or MODEL_PREFERENCE
    available = list_local_models()
    if not available:
        logger.warning("No Ollama models detected; using fallback: %s", fallback)
        return fallback

    available_names = {m.name for m in available}
    logger.debug("Ollama models available: %s", sorted(available_names))

    for model_name in pref:
        if model_name in available_names:
            logger.info("Auto-detected best LLM: %s", model_name)
            return model_name

    # None of the preferred models found — pick the first available
    # that isn't a remote/cloud model
    for m in available:
        if m.size_bytes > 0:  # skip remote-only models
            logger.info("Using first available LLM: %s", m.name)
            return m.name

    logger.warning("No suitable local model; using fallback: %s", fallback)
    return fallback


# ═══════════════════════════════════════════════════════════════════════
#  LLMClient
# ═══════════════════════════════════════════════════════════════════════


class LLMClient:
    """Client for interacting with a local LLM via Ollama.

    Provides simple ``chat()`` and ``generate()`` methods with
    conversation history, system prompts, and retry logic.

    Parameters
    ----------
    model : str, optional
        Ollama model name.  If ``None``, auto-detects the best
        available model (default: ``qwen3-coder:30b``).
    system_prompt : str, optional
        System-level instruction prepended to every conversation.
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
    timeout : int
        Per-request timeout in seconds.
    max_retries : int
        Number of retries on transient failures.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        self.model = model or detect_best_model()
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._history: List[Dict[str, str]] = []

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "你是一个专业的几何证明助手。你能够：\n"
            "1. 将自然语言的几何问题解析为结构化的前提和目标\n"
            "2. 分析证明失败的原因并提出修复建议\n"
            "3. 为复杂问题推荐证明策略\n"
            "4. 用自然语言解释证明过程\n\n"
            "You are a professional geometry proof assistant. You can:\n"
            "1. Parse natural-language geometry problems into structured premises and goals\n"
            "2. Analyze proof failures and suggest repairs\n"
            "3. Recommend proof strategies for complex problems\n"
            "4. Explain proofs in natural language\n\n"
            "回答时请简洁精确。如果需要输出结构化数据，请使用JSON格式。"
        )

    @property
    def available(self) -> bool:
        """Check if the LLM backend is reachable."""
        try:
            _ollama_request("/api/tags", timeout=3)
            return True
        except (URLError, OSError):
            return False

    def chat(
        self,
        message: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        keep_history: bool = False,
    ) -> LLMResponse:
        """Send a chat message and get a response.

        Parameters
        ----------
        message : str
            User message.
        system : str, optional
            Override system prompt for this call only.
        temperature : float, optional
            Override temperature for this call only.
        keep_history : bool
            If True, append to conversation history for multi-turn.

        Returns
        -------
        LLMResponse
        """
        messages: List[Dict[str, str]] = []

        # System prompt
        sys_prompt = system or self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # History (if multi-turn)
        if keep_history:
            messages.extend(self._history)

        # Current message
        messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
            },
        }

        t0 = time.time()
        resp = self._call_with_retry("/api/chat", payload)
        elapsed_ms = (time.time() - t0) * 1000

        content = resp.get("message", {}).get("content", "")

        # Update history
        if keep_history:
            self._history.append({"role": "user", "content": message})
            self._history.append({"role": "assistant", "content": content})

        return LLMResponse(
            content=content,
            model=resp.get("model", self.model),
            total_duration_ms=elapsed_ms,
            prompt_tokens=resp.get("prompt_eval_count", 0),
            completion_tokens=resp.get("eval_count", 0),
            raw=resp,
        )

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Raw text generation (non-chat mode).

        Parameters
        ----------
        prompt : str
            The prompt text.
        system : str, optional
            Override system prompt.
        temperature : float, optional
            Override temperature.

        Returns
        -------
        LLMResponse
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
            },
        }
        if system or self.system_prompt:
            payload["system"] = system or self.system_prompt

        t0 = time.time()
        resp = self._call_with_retry("/api/generate", payload)
        elapsed_ms = (time.time() - t0) * 1000

        return LLMResponse(
            content=resp.get("response", ""),
            model=resp.get("model", self.model),
            total_duration_ms=elapsed_ms,
            prompt_tokens=resp.get("prompt_eval_count", 0),
            completion_tokens=resp.get("eval_count", 0),
            raw=resp,
        )

    def clear_history(self) -> None:
        """Reset conversation history."""
        self._history.clear()

    def _call_with_retry(self, endpoint: str, payload: dict) -> dict:
        """Call Ollama API with retry logic."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return _ollama_request(
                    endpoint, payload, timeout=self.timeout,
                )
            except (URLError, OSError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s  – retrying in %ds",
                        attempt + 1, self.max_retries + 1, exc, wait,
                    )
                    time.sleep(wait)
        raise ConnectionError(
            f"LLM request failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    def __repr__(self) -> str:
        return f"LLMClient(model={self.model!r}, temp={self.temperature})"

    def chat_with_rag(
        self,
        message: str,
        *,
        rag_query: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat message augmented with RAG retrieval context.

        If the RAG module is available, retrieves relevant geometry
        knowledge and prepends it to the message.  Falls back to
        plain ``chat()`` if RAG is unavailable.

        Parameters
        ----------
        message : str
            User message / question.
        rag_query : str, optional
            Custom RAG query.  Defaults to ``message``.
        system : str, optional
            Override system prompt.
        temperature : float, optional
            Override temperature.

        Returns
        -------
        LLMResponse
        """
        try:
            from .rag import get_rag
            rag = get_rag()
            query = rag_query or message
            ctx = rag.retrieve(query)
            if ctx.has_results:
                augmented = ctx.augmented_prompt(message)
                return self.chat(augmented, system=system, temperature=temperature)
        except Exception as exc:
            logger.debug("RAG augmentation skipped: %s", exc)

        return self.chat(message, system=system, temperature=temperature)


# ═══════════════════════════════════════════════════════════════════════
#  Geometry-specific prompt templates
# ═══════════════════════════════════════════════════════════════════════

PARSE_NL_PROMPT = """\
将以下几何问题解析为JSON格式。提取所有前提条件(assumptions)和证明目标(goal)。

每个条件用如下格式表示：
{{"predicate": "谓词名", "args": ["参数1", "参数2", ...]}}

支持的谓词：
- Parallel(A, B, C, D) 表示直线AB平行于直线CD
- Perpendicular(A, B, C, D) 表示直线AB垂直于直线CD
- Collinear(A, B, C) 表示点A、B、C共线
- Cyclic(A, B, C, D) 表示点A、B、C、D共圆
- Midpoint(M, A, B) 表示M是线段AB的中点
- EqAngle(A, B, C, D, E, F) 表示∠ABC = ∠DEF

请只输出JSON，不要有其他文字。格式：
{{"assumptions": [...], "goal": {{...}}}}

几何问题：
{problem}"""

CRITIC_PROMPT = """\
以下几何证明搜索失败了。请分析失败原因并给出建议。

题目：
  前提: {assumptions}
  目标: {goal}

搜索结果：
  探索节点数: {nodes}
  诊断: {diagnosis}
  当前可用规则: {rules}

请回答：
1. 失败的可能原因（缺少什么规则或引理？）
2. 建议的辅助构造或中间步骤
3. 是否需要添加新的推理规则

用中文简洁回答。"""

STRATEGY_PROMPT = """\
为以下几何证明推荐搜索策略。

题目：
  前提: {assumptions}
  目标: {goal}
  前提数量: {n_facts}
  涉及的谓词: {predicates}

请推荐：
1. beam_width (搜索宽度，默认8)
2. max_depth (最大深度，默认6)
3. 推荐的证明方向（正向推理/反向推理/混合）
4. 简要理由

请只输出JSON：
{{"beam_width": N, "max_depth": N, "strategy": "forward|backward|hybrid", "reason": "..."}}"""

NARRATE_THEOREM_PROMPT = """\
你是一位优秀的数学老师。下面是一个已经被 Lean4 形式化验证通过的几何定理及其证明。
请用通俗易懂的自然语言，面向高中生，重新表述这个定理和它的证明过程。

要求：
1. 先用一句话概括定理的含义（"这个定理说的是……"）
2. 然后逐步解释证明过程，每一步说清楚用了什么性质、为什么成立
3. 最后用一句话总结证明的核心思路
4. 语言风格：简洁、准确、亲切，像在给学生讲课
5. 使用中文

定理：
  前提条件：{assumptions}
  结论：{goal}

证明步骤：
{proof_steps}

Lean4 形式化代码：
```lean
{lean_code}
```

验证状态：{verification_status}

请开始你的讲解："""


def narrate_theorem(
    assumptions_nl: str,
    goal_nl: str,
    proof_steps_nl: str,
    lean_code: str,
    verified: bool,
    llm: "LLMClient | None" = None,
) -> str:
    """Use the local LLM to narrate a verified theorem in natural language.

    Parameters
    ----------
    assumptions_nl : str
        Human-readable description of the assumptions.
    goal_nl : str
        Human-readable description of the goal.
    proof_steps_nl : str
        Human-readable proof steps (one per line).
    lean_code : str
        The Lean4 source code of the theorem.
    verified : bool
        Whether Lean4 verification passed.
    llm : LLMClient, optional
        LLM client to use.  Defaults to the global singleton.

    Returns
    -------
    str
        LLM-generated natural-language narration of the theorem and proof.

    Raises
    ------
    ConnectionError
        If the LLM is not reachable.
    """
    client = llm or get_llm()
    status = "✅ Lean4 验证通过" if verified else "❌ 验证未通过"
    prompt = NARRATE_THEOREM_PROMPT.format(
        assumptions=assumptions_nl,
        goal=goal_nl,
        proof_steps=proof_steps_nl,
        lean_code=lean_code,
        verification_status=status,
    )
    resp = client.chat(prompt, temperature=0.4)
    return resp.content.strip()


# ═══════════════════════════════════════════════════════════════════════
#  Module-level singleton
# ═══════════════════════════════════════════════════════════════════════

_global_llm: Optional[LLMClient] = None
_global_lock = threading.Lock()


def get_llm(model: Optional[str] = None) -> LLMClient:
    """Return (or create) the process-wide singleton LLMClient.

    On first call, auto-detects the best available model unless
    a specific model is requested.

    Parameters
    ----------
    model : str, optional
        Force a specific model.  If ``None``, auto-detect.

    Returns
    -------
    LLMClient
    """
    global _global_llm
    if _global_llm is None or (model is not None and _global_llm.model != model):
        with _global_lock:
            if _global_llm is None or (model is not None and _global_llm.model != model):
                _global_llm = LLMClient(model=model)
                logger.info("LLM initialised: %s", _global_llm)
    return _global_llm
