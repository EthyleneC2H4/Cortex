import { describe, expect, it } from "vitest";

import { applyProgressEvent, initialStages } from "./researchState";
import type { ResearchRunState } from "./researchState";

const T0 = 1_000_000;

function runningState(): ResearchRunState {
  return applyProgressEvent(
    { phase: "running", stages: [], mode: "pipeline", topic: "t" },
    { type: "run_start", mode: "pipeline", stages: ["gather", "research", "write", "critique", "revise"] },
    T0,
  );
}

describe("researchState 状态机", () => {
  it("run_start 初始化五个待执行阶段", () => {
    const s = runningState();
    expect(s.stages.map((x) => x.key)).toEqual([
      "gather",
      "research",
      "write",
      "critique",
      "revise",
    ]);
    expect(s.stages.every((x) => x.status === "pending")).toBe(true);
    expect(s.runStartedAtMs).toBe(T0);
  });

  it("stage_start → running 并记录起始时刻；stage_end → done 并落耗时", () => {
    let s = runningState();
    s = applyProgressEvent(s, { type: "stage_start", stage: "gather", run_elapsed: 0 }, T0 + 1000);
    expect(s.stages[0].status).toBe("running");
    expect(s.stages[0].startedAtMs).toBe(T0 + 1000);

    s = applyProgressEvent(
      s,
      { type: "stage_end", stage: "gather", ok: true, timed_out: false, elapsed: 12.5, run_elapsed: 13.5 },
      T0 + 13500,
    );
    expect(s.stages[0].status).toBe("done");
    expect(s.stages[0].elapsed).toBe(12.5);
    expect(s.stages[1].status).toBe("pending");
  });

  it("stage_end 带 timed_out → timeout 状态", () => {
    let s = runningState();
    s = applyProgressEvent(s, { type: "stage_start", stage: "research", run_elapsed: 1 }, T0);
    s = applyProgressEvent(
      s,
      { type: "stage_end", stage: "research", timed_out: true, elapsed: 900, run_elapsed: 901 },
      T0,
    );
    expect(s.stages[1].status).toBe("timeout");
  });

  it("stage_end 带 ok=false → failed（区别于超时降级）", () => {
    let s = runningState();
    s = applyProgressEvent(s, { type: "stage_start", stage: "gather", run_elapsed: 1 }, T0);
    s = applyProgressEvent(
      s,
      { type: "stage_end", stage: "gather", ok: false, timed_out: false, elapsed: 3.2, run_elapsed: 4.2 },
      T0,
    );
    expect(s.stages[0].status).toBe("failed");
    expect(s.stages[0].startedAtMs).toBeUndefined();
  });

  it("revise_skipped note → revise 阶段标记跳过", () => {
    let s = runningState();
    s = applyProgressEvent(s, { type: "note", code: "revise_skipped" }, T0);
    expect(s.stages[4].status).toBe("skipped");
  });

  it("未知 note code 不改变状态", () => {
    const s = runningState();
    const next = applyProgressEvent(s, { type: "note", code: "whatever" }, T0);
    expect(next).toBe(s);
  });

  it("initialStages 对 explore 模式给中文标签回退", () => {
    const stages = initialStages(["explore"]);
    expect(stages[0].label).toContain("自主探索");
  });
});
