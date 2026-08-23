import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ChatComposer } from "./ChatComposer";

/** 键盘契约：Enter 发送 / Shift+Enter 换行 / 空内容不发送 / 忙碌时切换停止。 */
describe("ChatComposer", () => {
  it("Enter 发送并清空输入", async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(<ChatComposer busy={false} onSend={onSend} />);
    await user.type(screen.getByLabelText("消息输入"), "你好{Enter}");
    expect(onSend).toHaveBeenCalledWith("你好");
    expect(screen.getByLabelText("消息输入")).toHaveValue("");
  });

  it("Shift+Enter 换行不发送", async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(<ChatComposer busy={false} onSend={onSend} />);
    await user.type(screen.getByLabelText("消息输入"), "第一行{Shift>}{Enter}{/Shift}第二行");
    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByLabelText("消息输入")).toHaveValue("第一行\n第二行");
  });

  it("纯空白不发送", async () => {
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(<ChatComposer busy={false} onSend={onSend} />);
    await user.type(screen.getByLabelText("消息输入"), "   {Enter}");
    expect(onSend).not.toHaveBeenCalled();
  });

  it("忙碌时显示停止按钮并触发 onStop", async () => {
    const onStop = vi.fn();
    const onSend = vi.fn();
    const user = userEvent.setup();
    render(<ChatComposer busy onSend={onSend} onStop={onStop} />);
    await user.click(screen.getByRole("button", { name: "停止生成" }));
    expect(onStop).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "发送消息" })).toBeNull();
  });
});
