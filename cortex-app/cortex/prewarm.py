"""启动预热（[net-once]，一次性）：解析模型 + tiktoken 缓存 + embedder 下载 + ping。"""

from __future__ import annotations

import argparse
import os


def prewarm(cfg) -> dict:
    steps: dict = {}

    # 1) 模型 ID 解析（写缓存）
    from .llm.model_resolver import resolve_model_id

    mid = resolve_model_id(cfg)
    steps["model_id"] = mid
    print(f"[1/4] model_id = {mid}")

    # 2) tiktoken 编码缓存（后续离线可用）
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cfg.data_dir / "tiktoken_cache"))
    try:
        import tiktoken

        tiktoken.get_encoding("cl100k_base")
        steps["tiktoken"] = "ok"
        print("[2/4] tiktoken cl100k_base 缓存就绪")
    except Exception as exc:
        steps["tiktoken"] = f"降级: {exc}"
        print(f"[2/4] tiktoken 缓存失败（将用 len//3 兜底）: {exc}")

    # 3) 本地 embedder 首次下载/加载
    provider = cfg.embed_provider
    if provider == "local":
        from .services.knowledge.embeddings import LocalEmbedder

        emb = LocalEmbedder(model_name=cfg.embed_model)
        v = emb.embed_one("预热")
        steps["embedder"] = f"{cfg.embed_model} dim={len(v)}"
        print(f"[3/4] embedder 就绪: {cfg.embed_model} (dim={len(v)})")
    else:
        steps["embedder"] = f"provider={provider}（跳过预热）"
        print(f"[3/4] embed provider = {provider}，跳过本地模型预热")

    # 4) 双档 LLM ping
    from .llm.factory import build_llms, ping

    strong, weak = build_llms(cfg, model_id=mid)
    steps["strong_ping"] = ping(strong)
    steps["weak_ping"] = ping(weak)
    print(f"[4/4] ping strong={steps['strong_ping']} weak={steps['weak_ping']}")

    # 5) 可选：BGE 重排器模型下载/加载（仅 enable_rerank 时；失败只降级不报错）
    if getattr(cfg, "enable_rerank", False):
        try:
            from .services.knowledge.rerank import BGEReranker

            rr = BGEReranker(cfg.rerank_model)
            score = rr._model.compute_score([["预热", "重排器预热查询"]], normalize=True)
            steps["reranker"] = f"{cfg.rerank_model} ok (score={float(score):.3f})"
            print(f"[5/5] reranker 就绪: {cfg.rerank_model}")
        except Exception as exc:
            steps["reranker"] = f"降级为纯 RRF: {exc}"
            print(f"[5/5] reranker 初始化失败（检索将用纯 RRF 排序）: {exc}")
    return steps


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m cortex.prewarm", description="Cortex 启动预热")
    parser.parse_args(argv)
    from .config import load_config

    cfg = load_config()
    prewarm(cfg)


if __name__ == "__main__":
    main()
