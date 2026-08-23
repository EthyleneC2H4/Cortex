import type { HealthInfo } from "@/api/types";

/** 知识库统计卡：知识库页与状态页共用（数据同源于 GET /health）。 */
export function KnowledgeStatsCards({ stats }: { stats: HealthInfo["knowledge"] }) {
  if (!stats) return null;
  const items: Array<[string, number]> = [
    ["文档", stats.documents],
    ["分块", stats.chunks],
    ["BM25", stats.bm25],
    ["向量", stats.vectors],
  ];
  return (
    <dl className="flex flex-wrap gap-3">
      {items.map(([label, n]) => (
        <div
          key={label}
          className="min-w-24 rounded-md border border-line bg-surface px-4 py-2.5 shadow-card"
        >
          <dt className="text-xs text-ink-2">{label}</dt>
          <dd className="font-display text-xl text-ink">{n}</dd>
        </div>
      ))}
      {stats.graph && (
        <div className="min-w-24 rounded-md border border-line bg-surface px-4 py-2.5 shadow-card">
          <dt className="text-xs text-ink-2">图谱节点</dt>
          <dd className="font-display text-xl text-ink">{stats.graph.nodes}</dd>
        </div>
      )}
    </dl>
  );
}
