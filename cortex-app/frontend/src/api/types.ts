/** 与后端 cortex/api/{schemas,server}.py 对应的手写类型（schema 同仓演进）。 */

export interface KnowledgeStats {
  documents: number;
  chunks: number;
  bm25: number;
  vectors: number;
  graph?: { nodes: number; edges: number };
}

export interface HealthInfo {
  status: string;
  version?: string;
  uptime_seconds?: number;
  auth_required?: boolean;
  research_running?: boolean;
  services?: Record<string, boolean>;
  knowledge?: KnowledgeStats;
}

export interface DocumentItem {
  doc_id: string;
  title: string;
  source: string;
  n_chunks: number;
  created_at: string;
}

export interface DocumentListResult {
  documents: DocumentItem[];
  total: number;
}

export interface IngestResult {
  doc_id: string;
  title?: string;
  n_chunks: number;
}

/** /research SSE progress 载荷（dispatch.py 结构化事件） */
export type StageProgress =
  | { type: "run_start"; mode: string; stages: string[] }
  | {
      type: "stage_start";
      stage: string;
      run_elapsed: number;
    }
  | {
      type: "stage_end";
      stage: string;
      ok?: boolean;
      timed_out?: boolean;
      needs_revision?: boolean;
      adopted?: boolean;
      elapsed: number;
      run_elapsed: number;
    }
  | { type: "note"; code: string; msg?: string }
  | { type: "run_end"; revised: boolean; duration_seconds: number };

export interface ResearchDonePayload {
  report_md: string;
  revised: boolean;
  duration_seconds: number;
  mode: string;
  topic: string;
  report_path: string;
}

export interface ResearchErrorPayload {
  message: string;
  detail?: string;
}

export interface ChatMessagePayload {
  role: string;
  content: string;
  duration_seconds?: number;
}

export interface SessionHistoryEntry {
  role: string;
  content: string;
}

export interface SessionInfoT {
  session_id: string;
  created_at: string;
  turns: number;
  history: SessionHistoryEntry[];
}
