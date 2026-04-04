"use client";

import { useState, useEffect } from "react";
import { ChevronDown, ChevronRight, Network, Users, FileText, Hash } from "lucide-react";
import { api } from "@/lib/api";
import { PageIndexNode } from "@/types";

/* ── Summary extraction helper ─────────────────────────────────────────── */
function extractSummary(summary: string | Record<string, unknown>): string {
  if (!summary) return "";
  if (typeof summary === "string") return summary;
  // dict with a "text" or "summary" key (various LLM outputs)
  const s = (summary as Record<string, unknown>);
  return (s.text || s.summary || s.content || JSON.stringify(s)) as string;
}

function truncate(str: string, len = 120): string {
  return str.length > len ? str.slice(0, len) + "…" : str;
}

/* ── Speaker chip ───────────────────────────────────────────────────────── */
const SPEAKER_COLOURS = [
  "bg-violet-500/15 text-violet-300 ring-violet-500/20",
  "bg-teal-500/15 text-teal-300 ring-teal-500/20",
  "bg-rose-500/15 text-rose-300 ring-rose-500/20",
  "bg-sky-500/15 text-sky-300 ring-sky-500/20",
  "bg-orange-500/15 text-orange-300 ring-orange-500/20",
  "bg-purple-500/15 text-purple-300 ring-purple-500/20",
];

function speakerColour(speaker: string, idx: number) {
  return SPEAKER_COLOURS[idx % SPEAKER_COLOURS.length];
}

function SpeakerChip({
  speaker,
  idx,
}: {
  speaker: string;
  idx: number;
}) {
  return (
    <span
      className={`font-mono-editorial inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-semibold ring-1 ${speakerColour(speaker, idx)}`}
    >
      {speaker}
    </span>
  );
}

/* ── Segment range badge ────────────────────────────────────────────────── */
function SegBadge({ start, end }: { start?: number; end?: number }) {
  if (start == null || end == null) return null;
  const count = Math.max(0, end - start);
  return (
    <span
      className="font-mono-editorial ml-auto flex items-center gap-1 rounded bg-white/4 px-1.5 py-0.5 text-[10px] text-slate-500 ring-1 ring-white/5"
    >
      <Hash className="h-2.5 w-2.5" />
      {count} seg{count !== 1 ? "s" : ""}
    </span>
  );
}

/* ── Level-2 Subtopic node ──────────────────────────────────────────────── */
function SubtopicNode({
  node,
  onHighlight,
  highlighted,
}: {
  node: PageIndexNode;
  onHighlight: (start: number, end: number) => void;
  highlighted: boolean;
}) {
  const summary = extractSummary(node.summary);

  return (
    <div
      className={`group cursor-pointer rounded-md border px-3 py-2.5 transition-all ${
        highlighted
          ? "border-indigo-500/40 bg-indigo-500/8"
          : "border-white/4 bg-white/2 hover:border-white/8 hover:bg-white/4"
      }`}
      onClick={() => {
        if (node.start_segment != null && node.end_segment != null) {
          onHighlight(node.start_segment, node.end_segment);
        }
      }}
    >
      <div className="flex items-start gap-2">
        <div className="mt-0.5 flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono-editorial text-xs font-medium text-slate-300">
              {node.node_id}
            </span>
            <span className="truncate text-xs text-slate-400">{node.title}</span>
            <SegBadge start={node.start_segment} end={node.end_segment} />
          </div>
          {summary && (
            <p className="mt-1 text-[11px] leading-relaxed text-slate-600">
              {truncate(summary, 100)}
            </p>
          )}
          {node.speakers && node.speakers.length > 0 && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {node.speakers.map((sp, i) => (
                <SpeakerChip key={sp} speaker={sp} idx={i} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Level-1 Topic card ─────────────────────────────────────────────────── */
function TopicCard({
  node,
  defaultOpen,
  onHighlight,
  highlightRange,
}: {
  node: PageIndexNode;
  defaultOpen?: boolean;
  onHighlight: (start: number, end: number) => void;
  highlightRange: [number, number] | null;
}) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const summary = extractSummary(node.summary);

  const isHighlighted =
    highlightRange != null &&
    node.start_segment != null &&
    node.end_segment != null &&
    node.start_segment === highlightRange[0] &&
    node.end_segment === highlightRange[1];

  return (
    <div className={`rounded-xl border transition-all ${
      isHighlighted ? "border-indigo-500/35" : "border-white/6"
    } bg-[#0d1120]`}>
      {/* Card header — click to expand */}
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-start gap-3 px-4 py-3 text-left"
      >
        <span className="mt-0.5 shrink-0 text-slate-600 transition-colors group-hover:text-slate-400">
          {open
            ? <ChevronDown className="h-4 w-4 text-indigo-400/60" />
            : <ChevronRight className="h-4 w-4 text-slate-600" />
          }
        </span>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-mono-editorial shrink-0 rounded bg-indigo-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-indigo-400/80 ring-1 ring-indigo-500/15">
              {node.node_id}
            </span>
            <span className="truncate text-sm font-medium text-slate-200">{node.title}</span>
            <SegBadge start={node.start_segment} end={node.end_segment} />
          </div>
          {summary && !open && (
            <p className="mt-1 text-[11px] leading-relaxed text-slate-600">
              {truncate(summary, 120)}
            </p>
          )}
        </div>
      </button>

      {/* Expanded content */}
      {open && (
        <div className="tree-expand border-t border-white/4 px-4 pb-4 pt-3">
          {summary && (
            <p className="mb-3 text-xs leading-relaxed text-slate-500">{summary}</p>
          )}
          {node.children && node.children.length > 0 ? (
            <div className="space-y-2">
              {node.children.map((child) => (
                <SubtopicNode
                  key={child.node_id}
                  node={child}
                  onHighlight={onHighlight}
                  highlighted={
                    highlightRange != null &&
                    child.start_segment === highlightRange[0] &&
                    child.end_segment === highlightRange[1]
                  }
                />
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-700 italic">No subtopics</p>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Main component ─────────────────────────────────────────────────────── */

interface PageIndexTreeProps {
  jobId: string;
  /** Called when user clicks a node — segment range to highlight in transcript */
  onHighlightSegments?: (start: number, end: number) => void;
}

export function PageIndexTree({ jobId, onHighlightSegments }: PageIndexTreeProps) {
  const [tree, setTree] = useState<PageIndexNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [highlightRange, setHighlightRange] = useState<[number, number] | null>(null);

  useEffect(() => {
    setLoading(true);
    api.getPageIndexTree(jobId)
      .then((data) => {
        setTree(data);
        setError(null);
      })
      .catch((err) => {
        setError(err.message || "Failed to load PageIndex tree");
      })
      .finally(() => setLoading(false));
  }, [jobId]);

  const handleHighlight = (start: number, end: number) => {
    setHighlightRange([start, end]);
    onHighlightSegments?.(start, end);
  };

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-6 text-sm text-slate-600">
        <Network className="h-4 w-4 animate-pulse text-indigo-400/50" />
        <span className="font-mono-editorial">Loading PageIndex tree…</span>
      </div>
    );
  }

  if (error || !tree) {
    return (
      <div className="rounded-lg border border-red-500/15 bg-red-500/5 px-4 py-3 text-xs text-red-400">
        {error || "Tree unavailable"}
      </div>
    );
  }

  const rootSummary = extractSummary(tree.summary);

  return (
    <div className="space-y-4">
      {/* ROOT header */}
      <div className="rounded-xl border border-white/6 bg-[#0d1120] px-5 py-4">
        <div className="flex items-start gap-3">
          <Network className="mt-0.5 h-4 w-4 shrink-0 text-indigo-400" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-mono-editorial text-xs font-semibold text-indigo-400/70">
                ROOT
              </span>
              <h3 className="font-display text-base font-semibold text-slate-100">
                {tree.title}
              </h3>
            </div>

            {rootSummary && (
              <p className="mt-2 text-xs leading-relaxed text-slate-500">{rootSummary}</p>
            )}

            {/* Meta row */}
            <div className="mt-3 flex flex-wrap items-center gap-3">
              {tree.n_segments != null && (
                <span className="font-mono-editorial flex items-center gap-1 text-[10px] text-slate-600">
                  <FileText className="h-3 w-3" />
                  {tree.n_segments} segments
                </span>
              )}
              {tree.participants && tree.participants.length > 0 && (
                <span className="flex items-center gap-1.5 text-[10px] text-slate-600">
                  <Users className="h-3 w-3" />
                  <span className="flex flex-wrap gap-1">
                    {tree.participants.map((p, i) => (
                      <SpeakerChip key={p} speaker={p} idx={i} />
                    ))}
                  </span>
                </span>
              )}
              {tree.children.length > 0 && (
                <span className="font-mono-editorial text-[10px] text-slate-700">
                  {tree.children.length} topic{tree.children.length !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Topic cards */}
      {tree.children.length === 0 ? (
        <p className="py-4 text-center text-sm text-slate-700 italic">No topics found in this tree.</p>
      ) : (
        <div className="space-y-2">
          {tree.children.map((topic, i) => (
            <TopicCard
              key={topic.node_id}
              node={topic}
              defaultOpen={i === 0}
              onHighlight={handleHighlight}
              highlightRange={highlightRange}
            />
          ))}
        </div>
      )}

      {highlightRange && (
        <div className="flex items-center justify-between rounded-md border border-indigo-500/20 bg-indigo-500/5 px-3 py-2">
          <span className="font-mono-editorial text-xs text-indigo-400/70">
            Highlighting segments {highlightRange[0]}–{highlightRange[1]}
          </span>
          <button
            onClick={() => { setHighlightRange(null); onHighlightSegments?.(-1, -1); }}
            className="text-xs text-slate-600 hover:text-slate-400 transition-colors"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
