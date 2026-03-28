"use client";

import { useState, useRef, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  Send,
  User,
  Bot,
  Loader2,
  FileText,
  ChevronDown,
  ChevronRight,
  Network,
  Settings2,
  ExternalLink,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { useChatWS } from "@/lib/hooks/useChatWS";
import { ChatSource, Job } from "@/types";
import { api } from "@/lib/api";

function fmtTime(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

/* ── Source accordion ────────────────────────────────────────────────────── */
function SourceAccordion({
  sources,
  retrievalMethod = "vector",
}: {
  sources: ChatSource[];
  retrievalMethod?: "vector" | "pageindex";
}) {
  const [isOpen, setIsOpen] = useState(false);
  if (!sources || sources.length === 0) return null;

  const isPageIndex = retrievalMethod === "pageindex";
  const label = isPageIndex
    ? `${sources.length} section${sources.length !== 1 ? "s" : ""}`
    : `${sources.length} segment${sources.length !== 1 ? "s" : ""}`;

  return (
    <div className="mt-3 rounded-lg border border-white/5 bg-white/2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] text-slate-500 transition-colors hover:text-slate-400"
      >
        <FileText className={`h-3 w-3 shrink-0 ${isPageIndex ? "text-amber-500/60" : "text-teal-500/60"}`} />
        <span className="flex-1">Retrieved context — {label}</span>
        <span
          className={`rounded px-1.5 py-0.5 text-[9px] font-semibold ring-1 ${
            isPageIndex
              ? "bg-amber-500/10 text-amber-400/80 ring-amber-500/20"
              : "bg-teal-500/10 text-teal-400/80 ring-teal-500/20"
          }`}
          style={{ fontFamily: "var(--font-mono, monospace)" }}
        >
          {isPageIndex ? "pageindex" : "vector"}
        </span>
        {isOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      </button>

      {isOpen && (
        <div className="border-t border-white/5 px-3 py-2 space-y-3">
          {sources.map((src, i) => (
            <div key={i}>
              <div className="mb-1 flex flex-wrap items-center gap-1.5">
                {isPageIndex && src.node_title ? (
                  <span
                    className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-semibold text-amber-400/80 ring-1 ring-amber-500/20"
                    style={{ fontFamily: "var(--font-mono, monospace)" }}
                  >
                    {src.node_title}
                  </span>
                ) : (
                  <span
                    className="text-[10px] text-slate-600"
                    style={{ fontFamily: "var(--font-mono, monospace)" }}
                    title="Distance (lower is better)"
                  >
                    dist:{src.distance?.toFixed(3) ?? "N/A"}
                  </span>
                )}
                {src.node_id && (
                  <span
                    className="text-[10px] text-slate-700"
                    style={{ fontFamily: "var(--font-mono, monospace)" }}
                  >
                    {src.node_id}
                  </span>
                )}
                <span
                  className="text-[10px] text-violet-400/70"
                  style={{ fontFamily: "var(--font-mono, monospace)" }}
                >
                  {src.speaker}
                </span>
                {typeof src.start === "number" && (
                  <span
                    className="text-[10px] text-slate-700"
                    style={{ fontFamily: "var(--font-mono, monospace)" }}
                  >
                    {fmtTime(src.start)}–{fmtTime(src.end)}
                  </span>
                )}
              </div>
              <p
                className={`border-l-2 pl-2 text-[11px] leading-relaxed text-slate-400 ${
                  isPageIndex ? "border-amber-500/30" : "border-teal-500/30"
                }`}
              >
                &ldquo;{src.text}&rdquo;
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Retrieval pill toggle ────────────────────────────────────────────────── */
function PillToggle({
  value,
  options,
  onChange,
}: {
  value: string;
  options: { label: string; value: string; disabled?: boolean; title?: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex rounded-lg border border-white/6 bg-[#0b0e18] p-0.5">
      {options.map((opt) => (
        <button
          key={opt.value}
          onClick={() => !opt.disabled && onChange(opt.value)}
          disabled={opt.disabled}
          title={opt.title}
          className={`flex-1 rounded-md px-3 py-1.5 text-[11px] font-semibold transition-all ${
            value === opt.value
              ? "bg-amber-500/15 text-amber-400 shadow-sm ring-1 ring-amber-500/25"
              : opt.disabled
              ? "cursor-not-allowed text-slate-700"
              : "text-slate-500 hover:text-slate-300"
          }`}
          style={{ fontFamily: "var(--font-mono, monospace)" }}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

/* ── Main page ───────────────────────────────────────────────────────────── */
type RetrievalMethod = "vector" | "pageindex";

export default function ChatPage() {
  const { id } = useParams() as { id: string };
  const router = useRouter();
  const [input, setInput] = useState("");
  const [llmMode, setLlmMode] = useState<"local" | "hf">("local");
  const [retrievalMethod, setRetrievalMethod] = useState<RetrievalMethod>("vector");
  const [job, setJob] = useState<Job | null>(null);
  const [pageindexAvailable, setPageindexAvailable] = useState(false);

  const { messages, loading, fetchingHistory, error, sendMessage, loadHistory, abortQuery } = useChatWS(id);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    return () => { abortQuery(); };
  }, [abortQuery]);

  useEffect(() => {
    const user = localStorage.getItem("meetbot_user");
    if (!user) { router.push("/login"); return; }
    if (!id) return;

    api.getJobStatus(id).then((j: Job) => {
      setJob(j);
      setPageindexAvailable(j.pageindex_status === "ready");
    }).catch((err: unknown) => {
      console.error("Failed to load job status:", err);
    });

    loadHistory();
  }, [id, router, loadHistory]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  useEffect(() => {
    if (!pageindexAvailable && retrievalMethod === "pageindex") {
      setRetrievalMethod("vector");
    }
  }, [pageindexAvailable, retrievalMethod]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    sendMessage(input, llmMode, retrievalMethod);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const statusLabel = loading
    ? "streaming"
    : fetchingHistory
    ? "loading"
    : "ready";

  return (
    <div className="flex h-[calc(100vh-3.5rem)] overflow-hidden -mt-8 -mx-4 sm:-mx-6">

      {/* ── Left sidebar ──────────────────────────────────────────────────── */}
      <aside className="hidden w-72 shrink-0 flex-col border-r border-white/5 bg-[#080b14] lg:flex">
        <div className="flex flex-1 flex-col gap-6 overflow-y-auto px-5 py-6">

          {/* Job info */}
          <div>
            <p
              className="mb-1 text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-700"
              style={{ fontFamily: "var(--font-mono, monospace)" }}
            >
              Session
            </p>
            <p className="truncate text-sm font-medium text-slate-300">
              {job?.original_filename || "Loading…"}
            </p>
            <div className="mt-2 flex items-center gap-2">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  loading ? "animate-pulse bg-amber-500" : "bg-teal-500"
                }`}
              />
              <span
                className="text-[10px] text-slate-600"
                style={{ fontFamily: "var(--font-mono, monospace)" }}
              >
                {statusLabel}
              </span>
            </div>
          </div>

          <div className="h-px bg-white/5" />

          {/* Retrieval method */}
          <div>
            <div className="mb-2 flex items-center gap-1.5">
              <Network className="h-3 w-3 text-slate-700" />
              <p
                className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-700"
                style={{ fontFamily: "var(--font-mono, monospace)" }}
              >
                Retrieval
              </p>
            </div>
            <PillToggle
              value={retrievalMethod}
              onChange={(v) => setRetrievalMethod(v as RetrievalMethod)}
              options={[
                { label: "Vector", value: "vector" },
                {
                  label: "PageIndex",
                  value: "pageindex",
                  disabled: !pageindexAvailable,
                  title: !pageindexAvailable
                    ? "Build PageIndex on the job detail page first"
                    : "Use LLM tree search",
                },
              ]}
            />
            {!pageindexAvailable && (
              <p className="mt-1.5 text-[10px] text-slate-700 italic">
                PageIndex not built
              </p>
            )}
          </div>

          {/* LLM mode */}
          <div>
            <div className="mb-2 flex items-center gap-1.5">
              <Settings2 className="h-3 w-3 text-slate-700" />
              <p
                className="text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-700"
                style={{ fontFamily: "var(--font-mono, monospace)" }}
              >
                LLM Backend
              </p>
            </div>
            <PillToggle
              value={llmMode}
              onChange={(v) => setLlmMode(v as "local" | "hf")}
              options={[
                { label: "Local", value: "local" },
                { label: "HF API", value: "hf" },
              ]}
            />
          </div>

          <div className="h-px bg-white/5" />

          {/* Job link */}
          <a
            href={`/job/${id}`}
            className="flex items-center gap-2 text-xs text-slate-600 transition-colors hover:text-slate-400"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            View job details
          </a>
        </div>
      </aside>

      {/* ── Main chat area ─────────────────────────────────────────────────── */}
      <div className="flex flex-1 flex-col overflow-hidden">

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-8 space-y-8">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <Bot className="mb-4 h-10 w-10 text-slate-800" />
              <p
                className="text-base font-semibold text-slate-400"
                style={{ fontFamily: "var(--font-display, serif)" }}
              >
                Ask about your transcript
              </p>
              <p className="mt-2 max-w-sm text-sm text-slate-600">
                Summarize key points, extract action items, or ask about specific topics discussed.
              </p>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div
                key={i}
                className={`animate-message-in flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {msg.role === "assistant" && (
                  <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-amber-500/10 ring-1 ring-amber-500/20">
                    <Bot className="h-3.5 w-3.5 text-amber-500/70" />
                  </div>
                )}

                <div
                  className={`max-w-[80%] sm:max-w-[70%] ${
                    msg.role === "user"
                      ? "rounded-2xl rounded-tr-sm bg-[#1a1f35] px-5 py-3.5 text-slate-200 ring-1 ring-white/6"
                      : "rounded-2xl rounded-tl-sm bg-[#0d1020] px-5 py-4 text-slate-300 ring-1 ring-white/4"
                  }`}
                >
                  {msg.role === "assistant" ? (
                    <div className="prose prose-sm prose-invert max-w-none prose-p:leading-relaxed prose-p:text-slate-300 prose-pre:bg-[#080b14] prose-pre:border prose-pre:border-white/5 prose-code:text-amber-300 prose-headings:text-slate-200">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content || (loading && i === messages.length - 1 ? "…" : "")}
                      </ReactMarkdown>
                      {msg.sources && (
                        <SourceAccordion
                          sources={msg.sources}
                          retrievalMethod={msg.retrieval_method ?? retrievalMethod}
                        />
                      )}
                      {msg.retrieval_level_note && (
                        <p
                          className="mt-2 text-[10px] text-slate-700"
                          style={{ fontFamily: "var(--font-mono, monospace)" }}
                        >
                          retrieved via {msg.retrieval_level_note}-level search
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="whitespace-pre-wrap text-sm leading-relaxed">
                      {msg.content}
                    </div>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/5 ring-1 ring-white/8">
                    <User className="h-3.5 w-3.5 text-slate-400" />
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Input bar */}
        <div className="border-t border-white/5 bg-[#080b14]/80 px-6 py-4 backdrop-blur">
          {error && (
            <p className="mb-2 text-xs text-red-400">Error: {error}</p>
          )}

          {loading && (
            <div className="mb-3 flex justify-end">
              <button
                type="button"
                onClick={abortQuery}
                className="flex items-center gap-1.5 rounded-full border border-red-500/20 bg-red-500/8 px-3 py-1.5 text-[11px] font-semibold text-red-400 transition-colors hover:bg-red-500/15"
              >
                <Loader2 className="h-3 w-3 animate-spin" />
                Stop generation
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="relative mx-auto max-w-3xl">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                loading
                  ? "Waiting for response…"
                  : "Ask about this transcript… (Shift+Enter for new line)"
              }
              className="block max-h-[140px] min-h-[52px] w-full resize-none rounded-xl border border-white/6 bg-[#0d1020] py-3.5 pl-4 pr-14 text-sm text-slate-200 placeholder:text-slate-600 focus:border-amber-500/40 focus:outline-none focus:ring-1 focus:ring-amber-500/20 disabled:opacity-50"
              rows={1}
              disabled={loading}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className={`absolute bottom-2 right-2 flex h-9 w-9 items-center justify-center rounded-lg transition-all ${
                input.trim() && !loading
                  ? "bg-amber-500 text-amber-950 shadow-[0_0_12px_rgba(245,158,11,0.3)] hover:bg-amber-400"
                  : "bg-white/4 text-slate-600"
              }`}
            >
              <Send className="ml-0.5 h-4 w-4" />
            </button>
          </form>

          <p
            className="mt-2 text-center text-[10px] text-slate-700"
            style={{ fontFamily: "var(--font-mono, monospace)" }}
          >
            MeetBot may make mistakes — verify important information.
          </p>
        </div>
      </div>
    </div>
  );
}
