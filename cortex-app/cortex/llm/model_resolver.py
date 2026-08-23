"""模型 ID 解析：OpenRouter /models → 免费 "alpha" 模型，带磁盘缓存。

用户未显式配置 ``LLM_MODEL_ID`` 时启动调用；结果缓存 24h 到
``data/model_cache.json``，避免每次启动都打 API。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Iterable, Optional

import httpx

from ..config import CortexConfig, CortexConfigError

CACHE_TTL_SECONDS = 86400


def pick_model_id(model_ids: Iterable[str]) -> Optional[str]:
    """从候选中选择模型：含 "alpha"（优先 :free）> 其它免费 > 任意。"""
    ids = sorted(set(model_ids))
    if not ids:
        return None
    alpha = [i for i in ids if "alpha" in i.lower()]
    free = [i for i in ids if i.endswith(":free")]
    alpha_free = [i for i in alpha if i.endswith(":free")]
    for pool in (alpha_free, alpha, free, ids):
        if pool:
            return pool[0]
    return None


def _read_cache(cfg: CortexConfig) -> Optional[str]:
    path = cfg.data_dir / "model_cache.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("base_url") != cfg.llm_base_url:
        return None
    resolved_at = data.get("resolved_at", 0)
    try:
        if time.time() - float(resolved_at) > CACHE_TTL_SECONDS:
            return None
    except (TypeError, ValueError):
        return None
    return data.get("model_id") or None


def _write_cache(cfg: CortexConfig, model_id: str) -> None:
    cfg.ensure_dirs()
    path = cfg.data_dir / "model_cache.json"
    payload = {
        "model_id": model_id,
        "base_url": cfg.llm_base_url,
        "resolved_at": time.time(),
        "resolved_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_model_id(
    cfg: CortexConfig,
    *,
    transport: Optional[httpx.BaseTransport] = None,
    force_refresh: bool = False,
) -> str:
    """解析最终使用的模型 ID。

    顺序：显式 LLM_MODEL_ID > 新鲜缓存 > GET {base_url}/models 现查。
    """
    if cfg.llm_model_id:
        return cfg.llm_model_id
    if not force_refresh:
        cached = _read_cache(cfg)
        if cached:
            return cached
    if not cfg.llm_api_key:
        raise CortexConfigError("LLM_API_KEY 未配置（写入 cortex-app/.env），或显式设置 LLM_MODEL_ID")

    headers = {"Authorization": f"Bearer {cfg.llm_api_key}"}
    try:
        with httpx.Client(base_url=cfg.llm_base_url, timeout=15, headers=headers, transport=transport) as client:
            resp = client.get("/models")
    except httpx.HTTPError as exc:
        raise CortexConfigError(
            f"无法访问模型列表 {cfg.llm_base_url}/models：{exc}。"
            "可设置 LLM_MODEL_ID 跳过自动解析，或检查网络。"
        ) from exc

    if resp.status_code in (401, 403):
        raise CortexConfigError(
            f"LLM_API_KEY 无效（HTTP {resp.status_code}）。请检查 .env 中的 LLM_API_KEY，"
            "或显式设置 LLM_MODEL_ID 后重试。"
        )
    resp.raise_for_status()
    try:
        model_ids = [m["id"] for m in resp.json().get("data", [])]
    except (ValueError, KeyError, TypeError) as exc:
        raise CortexConfigError(f"/models 响应格式异常：{exc}") from exc

    picked = pick_model_id(model_ids)
    if not picked:
        raise CortexConfigError("提供商未返回任何可用模型；请显式设置 LLM_MODEL_ID")
    _write_cache(cfg, picked)
    return picked
