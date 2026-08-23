"""双档 LLM 构造（规划文档 任务1.1）。

强模型做规划/写作（temperature 低-中、超时长），弱模型做路由/摘要/记忆判断
（temperature 0、超时短）。全部**显式传参**构造 HelloAgentsLLM——不依赖框架内部
读环境变量，便于测试与多提供商。
"""

from __future__ import annotations

from hello_agents import HelloAgentsLLM

from ..config import CortexConfig, CortexConfigError
from .model_resolver import resolve_model_id


def build_llms(cfg: CortexConfig, *, model_id: str | None = None) -> tuple[HelloAgentsLLM, HelloAgentsLLM]:
    """返回 (strong_llm, weak_llm)。两档默认同一模型，env 可分别覆盖。"""
    if not cfg.llm_api_key:
        raise CortexConfigError("LLM_API_KEY 未配置：请写入 cortex-app/.env（该文件已 gitignore）")
    mid = model_id or resolve_model_id(cfg)

    strong = HelloAgentsLLM(
        model=cfg.strong_model_id or mid,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        temperature=cfg.strong_temperature,
        timeout=cfg.strong_timeout,
    )
    weak = HelloAgentsLLM(
        model=cfg.weak_model_id or mid,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        temperature=cfg.weak_temperature,
        timeout=cfg.weak_timeout,
    )
    return strong, weak


def ping(llm: HelloAgentsLLM) -> bool:
    """轻量连通性检查（P1 验收：两个 LLM 均能对 ping 得到响应）。"""
    try:
        llm.invoke([{"role": "user", "content": "ping"}])
        return True
    except Exception:
        return False
