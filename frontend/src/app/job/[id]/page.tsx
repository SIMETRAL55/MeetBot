"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Job } from "@/types";
import { ProgressTimeline } from "@/components/ProgressTimeline";
import { TranscriptEditor } from "@/components/TranscriptEditor";
import { AudioPlayer, AudioPlayerHandle } from "@/components/AudioPlayer";
import { useJobProgressWS } from "@/lib/hooks/useJobProgressWS";
import { Download, RotateCcw, Loader2, XCircle } from "lucide-react";

export default function JobDetailPage() {
  const { id } = useParams() as { id: string };
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const audioPlayerRef = useRef<AudioPlayerHandle>(null);

  useEffect(() => {
    const user = localStorage.getItem("meetbot_user");
    if (!user) {
      router.push("/login");
    }
  }, [router]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  // Only open a WebSocket for jobs that are still actively processing.
  // For completed / failed / cancelled jobs, the initial REST fetch via
  // fetchJob() already supplies the final state, so a WS connection is
  // unnecessary and causes a "WS 4005: job already completed" error.
  const ACTIVE_JOB_STATUSES = ["pending", "transcribing", "diarizing", "aligning", "indexing", "reindexing"];
  const wsEnabled = !job || ACTIVE_JOB_STATUSES.includes(job.status);

  // FIX: Destructure `reconnect` so we can force a new WS connection after reindex/restart.
  const { data: wsData, error: wsError, connected, reconnect: reconnectWS } = useJobProgressWS(id, wsEnabled);

  const fetchJob = async () => {
    // Guard: id may be undefined during Next.js hydration before params resolve
    if (!id) return;
    try {
       const res = await api.getJobStatus(id);
       setJob(res);
       setError(null);
    } catch(err: unknown) {
       setError((err as Error).message || "Failed to load job profile.");
    } finally {
       setLoading(false);
    }
  };

  useEffect(() => {
    fetchJob();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // FIX: When WS reports completed/failed, refresh job from API so the page
  // shows the completed state (transcript editor, download buttons, etc.)
  useEffect(() => {
    if (
      wsData?.status === "completed" ||
      wsData?.status === "failed" ||
      wsData?.status === "cancelled"
    ) {
      // Terminal state delivered via WS — refresh DB state so the page reflects
      // the real final status rather than relying only on WS payload.
      fetchJob();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsData?.status]);

  const handleReindex = async () => {
    if (!job?.id || reindexing) return;
    try {
      setReindexing(true);
      await api.reindexJob(job.id);
      // After reindex starts, refresh job status and reconnect WS so we
      // stream reindexing progress events from the new active worker.
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
      // Refresh to pick up the new "cancelled" status from the DB.
      await fetchJob();
    } catch (err: unknown) {
      alert((err as Error).message || "Cancel failed");
    } finally {
      setCancelling(false);
    }
  };

  const handleDownload = (type: "aligned" | "transcription" | "diarization") => {
    // FIX: Use API_BASE_URL from environment instead of hardcoded localhost
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080/api";
    window.open(`${apiBase}/jobs/${id}/download?type=${type}`, "_blank");
  };

  if (loading) {
     return <div className="p-8 text-center text-slate-400">Loading Job Details...</div>;
  }

  if (error || wsError) {
     return (
       <div className="mx-auto max-w-4xl space-y-4 rounded-xl border border-red-500/20 bg-red-500/10 p-6 text-red-400 backdrop-blur">
          <h2 className="text-xl font-bold text-red-500">Error Loading Job</h2>
          <p>{error || wsError}</p>
       </div>
     );
  }

  // FIX: Compute display name from job metadata — show original filename
  // instead of raw UUID hash. The hash is only used internally as an ID.
  const displayName = job?.original_filename || "Job Details";
  const isReindexing = job?.status === "reindexing";
  // Show completed-state panels when job is done or actively reindexing
  // (transcript and downloads are still valid while reindex runs).
  const isCompleted = job?.status === "completed" || isReindexing;

  return (
    <div className="mx-auto max-w-5xl">
       <div className="mb-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
          {/* FIX: Show filename instead of hash ID. ID shown only as secondary info. */}
          {displayName}
          {!connected && (
             <span className="ml-2 inline-flex items-center text-yellow-500">
               <span className="mr-1 h-2 w-2 rounded-full bg-yellow-500"></span> Disconnected from real-time updates
             </span>
          )}
          {connected && (
             <span className="ml-2 inline-flex items-center text-teal-500">
               <span className="mr-1 h-2 w-2 rounded-full bg-teal-500 animate-pulse"></span> Streaming real-time
             </span>
          )}
       </div>

       <ProgressTimeline
          job={job}
          wsData={wsData}
          onUpdate={fetchJob}
          onReconnect={reconnectWS}
       />

       {/* ── Completed-job actions: Downloads + Reindex ── */}
       {isCompleted && (
         <div className="mt-8 grid gap-4 sm:grid-cols-2">
           {/* Download Panel */}
           <div className="rounded-xl border border-white/10 bg-slate-900/50 p-6 backdrop-blur">
             <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-200">
               <Download className="h-5 w-5 text-teal-400" />
               Download Transcripts
             </h3>
             <div className="space-y-2">
               <button
                 onClick={() => handleDownload("aligned")}
                 className="w-full rounded-md border border-slate-700 bg-slate-800 py-2 text-sm font-medium text-slate-200 transition-colors hover:border-teal-500/50 hover:bg-slate-700 hover:text-white"
               >
                 📝 Aligned Transcript
               </button>
               <button
                 onClick={() => handleDownload("transcription")}
                 className="w-full rounded-md border border-slate-700 bg-slate-800 py-2 text-sm font-medium text-slate-200 transition-colors hover:border-teal-500/50 hover:bg-slate-700 hover:text-white"
               >
                 🎤 Raw Whisper Output
               </button>
               <button
                 onClick={() => handleDownload("diarization")}
                 className="w-full rounded-md border border-slate-700 bg-slate-800 py-2 text-sm font-medium text-slate-200 transition-colors hover:border-teal-500/50 hover:bg-slate-700 hover:text-white"
               >
                 👥 Speaker Diarization JSON
               </button>
             </div>
           </div>

           {/* Reindex Panel */}
           <div className="rounded-xl border border-white/10 bg-slate-900/50 p-6 backdrop-blur">
             <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-200">
               <RotateCcw className="h-5 w-5 text-orange-400" />
               Reindex
             </h3>
             {isReindexing ? (
               /* While reindexing: show progress indicator + cancel button */
               <div className="space-y-3">
                 <div className="flex items-center gap-2 text-sm text-orange-300">
                   <Loader2 className="h-4 w-4 animate-spin" />
                   Reindexing in progress…
                 </div>
                 <button
                   onClick={handleCancelReindex}
                   disabled={cancelling}
                   className="flex w-full items-center justify-center gap-2 rounded-md border border-red-500/40 bg-red-500/10 py-2 text-sm font-medium text-red-400 transition-colors hover:bg-red-500/20 disabled:opacity-50"
                 >
                   {cancelling ? (
                     <><Loader2 className="h-4 w-4 animate-spin" /> Cancelling…</>
                   ) : (
                     <><XCircle className="h-4 w-4" /> Cancel Reindex</>
                   )}
                 </button>
               </div>
             ) : (
               /* Idle state: rebuild button */
               <>
                 <p className="mb-4 text-sm leading-relaxed text-slate-400">
                   Rebuild the RAG vector index after editing transcript segments.
                   This ensures the AI assistant uses the latest edits.
                 </p>
                 <button
                   onClick={handleReindex}
                   disabled={reindexing}
                   className="flex w-full items-center justify-center gap-2 rounded-md bg-orange-500 py-2 text-sm font-medium text-white transition-colors hover:bg-orange-400 disabled:opacity-50 shadow-[0_0_15px_rgba(249,115,22,0.25)]"
                 >
                   {reindexing ? (
                     <><Loader2 className="h-4 w-4 animate-spin" /> Starting…</>
                   ) : (
                     <><RotateCcw className="h-4 w-4" /> Rebuild Vector Index</>
                   )}
                 </button>
               </>
             )}
           </div>
         </div>
       )}

       {/* Transcript Editor Section — shown for completed jobs; placeholder while processing */}
       {isCompleted ? (
          <div className="mt-8">
             <div className="mb-4 flex items-center justify-between">
               <h2 className="text-xl font-bold tracking-tight text-white drop-shadow-sm">Transcript Editor</h2>
             </div>
             {/* Audio player — click timestamps in the transcript to seek */}
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
       ) : (
          <div className="mt-8 rounded-xl border border-white/10 bg-slate-900/50 p-8 text-center backdrop-blur">
             <Loader2 className="mx-auto mb-3 h-8 w-8 animate-spin text-teal-500/60" />
             <p className="text-sm font-medium text-slate-400">File is currently being processed…</p>
             <p className="mt-1 text-xs text-slate-600">The transcript will appear here automatically once processing completes.</p>
          </div>
       )}
    </div>
  );
}
