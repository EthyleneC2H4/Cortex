"""P2 live：真实 LLM 驱动 ReActAgent 调用笔记工具（HAS_LLM_KEY=1 才跑）。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hello_agents import ReActAgent, ToolRegistry

from cortex.config import framework_config, load_config
from cortex.llm.factory import build_llms
from cortex.tools.registry_setup import register_builtin_tools, register_cortex_tools

pytestmark = [pytest.mark.e2e]

requires_key = pytest.mark.skipif(
    not os.getenv("HAS_LLM_KEY"), reason="需要 HAS_LLM_KEY=1 且 .env 配好 LLM_API_KEY"
)


@requires_key
def test_agent_uses_note_tool(tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(env_file=root / ".env")
    cfg.rebase(tmp_path)  # 笔记等数据落 tmp，不污染仓库
    cfg.ensure_dirs()

    strong, weak = build_llms(cfg)
    registry = ToolRegistry()
    register_builtin_tools(registry, project_root=tmp_path)
    register_cortex_tools(registry, cfg)

    agent = ReActAgent(
        name="note-tester",
        llm=strong,
        tool_registry=registry,
        config=framework_config(cfg),
    )
    agent.run(
        "请创建一条标题为「冒烟测试」内容为「这是工具链验证」的笔记，"
        "然后读取它确认存在，最后告诉我版本号。"
    )
    # 验收以工具副作用为准（免费模型最终答复措辞不稳定，不作断言）：
    # ReAct 循环确实调用了 note_create 并落盘
    notes = list(cfg.notes_dir.glob("*.md"))
    assert notes, "agent 未真正调用 note_create 落盘"
    body = notes[0].read_text(encoding="utf-8")
    assert "工具链验证" in body
