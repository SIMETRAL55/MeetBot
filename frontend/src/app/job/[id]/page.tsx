"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Job } from "@/types";
import { ProgressTimeline } from "@/components/ProgressTimeline";
import { TranscriptEditor } from "@/components/TranscriptEditor";
import { AudioPlayer, AudioPlayerHandle } from "@/components/AudioPlayer";
import { useJobProgressWS } from "@/lib/hooks/useJobProgressWS";
import { Download, RotateCcw, Loader2, XCircle, Network } from "lucide-react";

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
  const [buildingPageIndex, setBuildingPageIndex] = useState(false);

  const ACTIVE_JOB_STATUSES = ["pending", "transcribing", "diarizing", "aligning", "indexing", "reindexing"];
  const wsEnabled = !job || ACTIVE_JOB_STATUSES.includes(job.status);

  const { data: wsData, error: wsError, connected, reconnect: reconnectWS } = useJobProgressWS(id, wsEnabled);

  const fetchJob = async () => {
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

  useEffect(() => {
    if (
      wsData?.status === "completed" ||
      wsData?.status === "failed" ||
      wsData?.status === "cancelled"
    ) {
      fetchJob();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsData?.status]);

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
      // Poll for completion by refreshing after a short delay
      setTimeout(async () => {
        await fetchJob();
        setBuildingPageIndex(false);
      }, 3000);
    } catch (err: unknown) {
      alert((err as Error).message || "Build PageIndex failed");
      setBuildingPageIndex(false);
    }
  };

  const handleDownload = (type: "aligned" | "transcription" | "diarization") => {
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

  const displayName = job?.original_filename || "Job Details";
  const isReindexing = job?.status === "reindexing";
  const isCompleted = job?.status === "completed" || isReindexing;

  const pageindexStatus = job?.pageindex_status;
  const pageindexReady = pageindexStatus === "ready";
  const pageindexBuilding = pageindexStatus === "building" || buildingPageIndex;
  const pageindexFailed = pageindexStatus === "failed";

  return (
    <div className="mx-auto max-w-5xl">
       <div className="mb-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
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

       {/* ── Completed-job actions ── */}
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

           {/* PageIndex Panel — only shown when job is completed */}
           {job?.status === "completed" && (
             <div className="rounded-xl border border-white/10 bg-slate-900/50 p-6 backdrop-blur sm:col-span-2 lg:col-span-1">
               <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-200">
                 <Network className="h-5 w-5 text-amber-400" />
                 PageIndex
                 {pageindexReady && (
                   <span className="ml-auto rounded-full bg-amber-500/10 px-2 py-0.5 text-xs font-semibold text-amber-400 ring-1 ring-amber-500/20">
                     Ready
                   </span>
                 )}
                 {pageindexBuilding && (
                   <span className="ml-auto rounded-full bg-slate-700 px-2 py-0.5 text-xs text-slate-400">
                     Building…
                   </span>
                 )}
                 {pageindexFailed && (
                   <span className="ml-auto rounded-full bg-red-500/10 px-2 py-0.5 text-xs text-red-400 ring-1 ring-red-500/20">
                     Failed
                   </span>
                 )}
               </h3>

               {pageindexBuilding ? (
                 <div className="flex items-center gap-2 text-sm text-amber-300">
                   <Loader2 className="h-4 w-4 animate-spin" />
                   Building PageIndex tree… this may take a few minutes.
                 </div>
               ) : pageindexReady ? (
                 <div className="space-y-3">
                   <p className="text-sm text-slate-400">
                     PageIndex tree is built. Select <span className="font-semibold text-amber-400">PageIndex</span> in the chat to use it.
                   </p>
                   <button
                     onClick={handleBuildPageIndex}
                     className="flex w-full items-center justify-center gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 py-2 text-sm font-medium text-amber-400 transition-colors hover:bg-amber-500/20"
                   >
                     <RotateCcw className="h-4 w-4" /> Rebuild PageIndex
                   </button>
                 </div>
               ) : (
                 <>
                   <p className="mb-4 text-sm leading-relaxed text-slate-400">
                     Build a vectorless LLM-based tree index for this transcript.
                     Requires <code className="rounded bg-slate-800 px-1 text-xs text-amber-300">PAGEINDEX_ENABLED=true</code> in your environment.
                     {pageindexFailed && (
                       <span className="mt-1 block text-red-400">Previous build failed. Try again.</span>
                     )}
                   </p>
                   <button
                     onClick={handleBuildPageIndex}
                     disabled={buildingPageIndex}
                     className="flex w-full items-center justify-center gap-2 rounded-md bg-amber-500 py-2 text-sm font-medium text-white transition-colors hover:bg-amber-400 disabled:opacity-50 shadow-[0_0_15px_rgba(245,158,11,0.2)]"
                   >
                     {buildingPageIndex ? (
                       <><Loader2 className="h-4 w-4 animate-spin" /> Starting…</>
                     ) : (
                       <><Network className="h-4 w-4" /> Build PageIndex</>
                     )}
                   </button>
                 </>
               )}
             </div>
           )}
         </div>
       )}

       {/* Transcript Editor Section */}
       {isCompleted ? (
          <div className="mt-8">
             <div className="mb-4 flex items-center justify-between">
               <h2 className="text-xl font-bold tracking-tight text-white drop-shadow-sm">Transcript Editor</h2>
             </div>
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
