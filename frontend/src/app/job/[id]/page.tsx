"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Job } from "@/types";
import { ProgressTimeline } from "@/components/ProgressTimeline";
import { TranscriptEditor } from "@/components/TranscriptEditor";
import { AudioPlayer, AudioPlayerHandle } from "@/components/AudioPlayer";
import { PageIndexTree } from "@/components/PageIndexTree";
import { useJobProgressWS } from "@/lib/hooks/useJobProgressWS";
import {
  Download,
  RotateCcw,
  Loader2,
  XCircle,
  Network,
  MessageSquare,
  FileJson,
  Mic,
  Users,
} from "lucide-react";

/* ── Small section label ─────────────────────────────────────────────────── */
function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p
      className="mb-3 text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-700"
      style={{ fontFamily: "var(--font-mono, monospace)" }}
    >
      {children}
    </p>
  );
}

/* ── Action button ───────────────────────────────────────────────────────── */
function ActionBtn({
  onClick,
  disabled,
  variant = "ghost",
  className = "",
  children,
}: {
  onClick?: () => void;
  disabled?: boolean;
  variant?: "ghost" | "primary" | "danger" | "amber";
  className?: string;
  children: React.ReactNode;
}) {
  const base =
    "flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold transition-all disabled:opacity-40";
  const variants = {
    ghost:
      "border border-white/6 bg-white/2 text-slate-400 hover:bg-white/5 hover:text-slate-200",
    primary:
      "border border-teal-500/30 bg-teal-500/10 text-teal-400 hover:bg-teal-500/20",
    danger:
      "border border-red-500/30 bg-red-500/8 text-red-400 hover:bg-red-500/15",
    amber:
      "bg-amber-500 text-amber-950 shadow-[0_0_14px_rgba(245,158,11,0.25)] hover:bg-amber-400",
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${base} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

/* ── Main page ───────────────────────────────────────────────────────────── */
export default function JobDetailPage() {
  const { id } = useParams() as { id: string };
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [buildingPageIndex, setBuildingPageIndex] = useState(false);
  const [cancellingPageIndex, setCancellingPageIndex] = useState(false);
  const audioPlayerRef = useRef<AudioPlayerHandle>(null);

  useEffect(() => {
    const user = localStorage.getItem("meetbot_user");
    if (!user) { router.push("/login"); }
  }, [router]);

  const ACTIVE_JOB_STATUSES = ["pending", "transcribing", "diarizing", "aligning", "indexing", "reindexing"];
  const wsEnabled = !job || ACTIVE_JOB_STATUSES.includes(job.status);

  const { data: wsData, error: wsError, connected, reconnect: reconnectWS } = useJobProgressWS(id, wsEnabled);

  const fetchJob = async () => {
    if (!id) return;
    try {
      const res = await api.getJobStatus(id);
      setJob(res);
      setError(null);
    } catch (err: unknown) {
      setError((err as Error).message || "Failed to load job.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchJob(); }, [id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (
      wsData?.status === "completed" ||
      wsData?.status === "failed" ||
      wsData?.status === "cancelled"
    ) {
      fetchJob();
    }
  }, [wsData?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleReindex = async () => {
    if (!job?.id || reindexing) return;
    try {
      setReindexing(true);
      await api.reindexJob(job.id);
      await fetchJob();
      reconnectWS();
    } catch (err: unknown) {
      alert((err as Error).message || "Reindex failed");
    } finally {
      setReindexing(false);
    }
  };

  const handleCancelReindex = async () => {
    if (!job?.id || cancelling) return;
    try {
      setCancelling(true);
      await api.cancelJob(job.id);
      await fetchJob();
    } catch (err: unknown) {
      alert((err as Error).message || "Cancel failed");
    } finally {
      setCancelling(false);
    }
  };

  const handleBuildPageIndex = async () => {
    if (!job?.id || buildingPageIndex) return;
    try {
      setBuildingPageIndex(true);
      await api.buildPageIndex(job.id);
      const poll = async () => {
        try {
          const res = await api.getJobStatus(job.id);
          setJob(res);
          if (res.pageindex_status === "building") {
            setTimeout(poll, 3000);
          } else {
            setBuildingPageIndex(false);
          }
        } catch {
          setBuildingPageIndex(false);
        }
      };
      setTimeout(poll, 3000);
    } catch (err: unknown) {
      alert((err as Error).message || "Build PageIndex failed");
      setBuildingPageIndex(false);
    }
  };

  const handleCancelPageIndex = async () => {
    if (!job?.id || cancellingPageIndex) return;
    try {
      setCancellingPageIndex(true);
      await api.cancelPageIndex(job.id);
      await fetchJob();
      setBuildingPageIndex(false);
    } catch (err: unknown) {
      alert((err as Error).message || "Cancel PageIndex failed");
    } finally {
      setCancellingPageIndex(false);
    }
  };

  const handleDownload = (type: "aligned" | "transcription" | "diarization") => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080/api";
    window.open(`${apiBase}/jobs/${id}/download?type=${type}`, "_blank");
  };

  if (loading) {
    return (
      <div className="flex h-48 items-center justify-center gap-2 text-sm text-slate-600">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span style={{ fontFamily: "var(--font-mono, monospace)" }}>Loading…</span>
      </div>
    );
  }

  if (error || wsError) {
    return (
      <div className="mx-auto max-w-2xl rounded-xl border border-red-500/15 bg-red-500/5 p-6 text-sm text-red-400">
        <p className="mb-1 font-semibold text-red-300">Error loading job</p>
        <p>{error || wsError}</p>
      </div>
    );
  }

  const isReindexing = job?.status === "reindexing";
  const isCompleted = job?.status === "completed" || isReindexing;

  const pageindexStatus = job?.pageindex_status;
  const pageindexReady = pageindexStatus === "ready";
  const pageindexBuilding = pageindexStatus === "building" || buildingPageIndex;
  const pageindexFailed = pageindexStatus === "failed" || pageindexStatus === "cancelled";

  return (
    <div className="mx-auto max-w-6xl">

      {/* ── Job header ─────────────────────────────────────────────────────── */}
      <div className="mb-6 flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <h1
            className="truncate text-lg font-semibold text-slate-200"
            style={{ fontFamily: "var(--font-display, serif)" }}
          >
            {job?.original_filename || "Job Details"}
          </h1>
          <div className="mt-1 flex items-center gap-2">
            <span
              className={`h-1.5 w-1.5 rounded-full ${connected ? "animate-pulse bg-teal-500" : "bg-yellow-500"}`}
            />
            <span
              className="text-[10px] text-slate-600"
              style={{ fontFamily: "var(--font-mono, monospace)" }}
            >
              {connected ? "live" : "disconnected"}
            </span>
            {job?.status && (
              <>
                <span className="text-slate-800">·</span>
                <span
                  className="text-[10px] text-slate-600"
                  style={{ fontFamily: "var(--font-mono, monospace)" }}
                >
                  {job.status}
                </span>
              </>
            )}
          </div>
        </div>

        {isCompleted && (
          <a
            href={`/chat/${id}`}
            className="flex items-center gap-2 rounded-lg border border-amber-500/25 bg-amber-500/8 px-4 py-2 text-xs font-semibold text-amber-400 transition-colors hover:bg-amber-500/15"
          >
            <MessageSquare className="h-3.5 w-3.5" />
            Open Chat
          </a>
        )}
      </div>

      {/* ── Progress timeline ───────────────────────────────────────────────── */}
      <ProgressTimeline
        job={job}
        wsData={wsData}
        onUpdate={fetchJob}
        onReconnect={reconnectWS}
      />

      {/* ── Completed layout ────────────────────────────────────────────────── */}
      {isCompleted ? (
        <div className="mt-8 grid gap-8 xl:grid-cols-[1fr_300px]">

          {/* Left: transcript */}
          <div className="min-w-0">
            <AudioPlayer
              ref={audioPlayerRef}
              src={api.getAudioUrl(id)}
              className="mb-4"
            />
            <TranscriptEditor
              jobId={id}
              onTimestampClick={(secs) => audioPlayerRef.current?.seekTo(secs)}
            />
          </div>

          {/* Right: action sidebar */}
          <div className="space-y-4">

            {/* Download */}
            <div className="rounded-xl border border-white/6 bg-[#0d1120] p-5">
              <SectionLabel>Download</SectionLabel>
              <div className="space-y-2">
                <ActionBtn variant="ghost" onClick={() => handleDownload("aligned")}>
                  <FileJson className="h-3.5 w-3.5" />
                  Aligned Transcript
                </ActionBtn>
                <ActionBtn variant="ghost" onClick={() => handleDownload("transcription")}>
                  <Mic className="h-3.5 w-3.5" />
                  Raw Whisper Output
                </ActionBtn>
                <ActionBtn variant="ghost" onClick={() => handleDownload("diarization")}>
                  <Users className="h-3.5 w-3.5" />
                  Speaker Diarization
                </ActionBtn>
              </div>
            </div>

            {/* Reindex */}
            <div className="rounded-xl border border-white/6 bg-[#0d1120] p-5">
              <SectionLabel>Vector Index</SectionLabel>
              {isReindexing ? (
                <div className="space-y-3">
                  <div
                    className="flex items-center gap-2 text-xs text-orange-300"
                    style={{ fontFamily: "var(--font-mono, monospace)" }}
                  >
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Reindexing…
                  </div>
                  <ActionBtn variant="danger" onClick={handleCancelReindex} disabled={cancelling}>
                    {cancelling ? (
                      <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Cancelling…</>
                    ) : (
                      <><XCircle className="h-3.5 w-3.5" /> Cancel</>
                    )}
                  </ActionBtn>
                </div>
              ) : (
                <>
                  <p className="mb-3 text-xs leading-relaxed text-slate-600">
                    Rebuild vector index after editing segments.
                  </p>
                  <ActionBtn variant="ghost" onClick={handleReindex} disabled={reindexing}>
                    {reindexing ? (
                      <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Starting…</>
                    ) : (
                      <><RotateCcw className="h-3.5 w-3.5" /> Rebuild Index</>
                    )}
                  </ActionBtn>
                </>
              )}
            </div>

            {/* PageIndex */}
            {job?.status === "completed" && (
              <div className="rounded-xl border border-white/6 bg-[#0d1120] p-5">
                <div className="mb-3 flex items-center gap-2">
                  <SectionLabel>PageIndex</SectionLabel>
                  {pageindexReady && (
                    <span
                      className="ml-auto rounded-full bg-amber-500/10 px-2 py-0.5 text-[9px] font-semibold text-amber-400 ring-1 ring-amber-500/20"
                      style={{ fontFamily: "var(--font-mono, monospace)" }}
                    >
                      ready
                    </span>
                  )}
                  {pageindexBuilding && (
                    <span
                      className="ml-auto rounded-full bg-white/4 px-2 py-0.5 text-[9px] text-slate-500 ring-1 ring-white/5"
                      style={{ fontFamily: "var(--font-mono, monospace)" }}
                    >
                      building
                    </span>
                  )}
                  {pageindexFailed && (
                    <span
                      className="ml-auto rounded-full bg-red-500/10 px-2 py-0.5 text-[9px] text-red-400 ring-1 ring-red-500/15"
                      style={{ fontFamily: "var(--font-mono, monospace)" }}
                    >
                      {pageindexStatus === "cancelled" ? "cancelled" : "failed"}
                    </span>
                  )}
                </div>

                {pageindexBuilding ? (
                  <div className="space-y-3">
                    <div
                      className="flex items-center gap-2 text-xs text-amber-400/80"
                      style={{ fontFamily: "var(--font-mono, monospace)" }}
                    >
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      Building tree…
                    </div>
                    <ActionBtn variant="danger" onClick={handleCancelPageIndex} disabled={cancellingPageIndex}>
                      {cancellingPageIndex ? (
                        <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Stopping…</>
                      ) : (
                        <><XCircle className="h-3.5 w-3.5" /> Stop</>
                      )}
                    </ActionBtn>
                  </div>
                ) : pageindexReady ? (
                  <div className="space-y-3">
                    <p className="text-xs text-slate-600">
                      Select <span className="font-semibold text-amber-400">PageIndex</span> in chat to use tree search.
                    </p>
                    <ActionBtn variant="ghost" onClick={handleBuildPageIndex}>
                      <RotateCcw className="h-3.5 w-3.5" /> Rebuild
                    </ActionBtn>
                  </div>
                ) : (
                  <>
                    <p className="mb-3 text-xs leading-relaxed text-slate-600">
                      Build an LLM-based tree index for this transcript.
                      {pageindexFailed && (
                        <span className="mt-1 block text-red-400/80">
                          {pageindexStatus === "cancelled" ? "Previous build cancelled." : "Previous build failed."} Try again.
                        </span>
                      )}
                    </p>
                    <ActionBtn variant="amber" onClick={handleBuildPageIndex} disabled={buildingPageIndex}>
                      {buildingPageIndex ? (
                        <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Starting…</>
                      ) : (
                        <><Network className="h-3.5 w-3.5" /> Build PageIndex</>
                      )}
                    </ActionBtn>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="mt-8 flex flex-col items-center justify-center rounded-xl border border-white/5 bg-[#0d1120] py-16 text-center">
          <Loader2 className="mb-3 h-7 w-7 animate-spin text-slate-700" />
          <p className="text-sm font-medium text-slate-500">Processing…</p>
          <p className="mt-1 text-xs text-slate-700">Transcript will appear once complete.</p>
        </div>
      )}

      {/* ── PageIndex tree (full width, below grid) ─────────────────────────── */}
      {isCompleted && pageindexReady && id && (
        <div className="mt-8">
          <div className="mb-4 flex items-center gap-2">
            <Network className="h-4 w-4 text-amber-500/60" />
            <h2
              className="text-sm font-semibold text-slate-400"
              style={{ fontFamily: "var(--font-mono, monospace)" }}
            >
              PageIndex Tree
            </h2>
          </div>
          <PageIndexTree jobId={id} />
        </div>
      )}
    </div>
  );
}
