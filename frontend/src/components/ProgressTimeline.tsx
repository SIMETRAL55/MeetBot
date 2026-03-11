"use client";

import { useRef, useEffect } from "react";
import { CheckCircle2, Circle, Loader2, StopCircle, RefreshCcw, FileText, ArrowLeft, Trash2 } from "lucide-react";
import { Job, JobStatus } from "@/types";
import { ProgressEvent } from "@/lib/hooks/useJobProgressWS";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { api } from "@/lib/api";
import { useRouter } from "next/navigation";

const STAGES: { id: JobStatus; label: string }[] = [
  { id: "pending", label: "Queued" },
  { id: "transcribing", label: "Transcription" },
  { id: "diarizing", label: "Speaker Diarization" },
  { id: "aligning", label: "Alignment" },
  { id: "indexing", label: "RAG Indexing" },
];

export function ProgressTimeline({
  job,
  wsData,
  onUpdate,
  onReconnect
}: {
  job: Job | null;
  wsData: ProgressEvent | null;
  onUpdate?: () => void;
  /** FIX: Optional callback to reconnect the progress WS after restart */
  onReconnect?: () => void;
}) {
  const router = useRouter();

  // Auto-scroll ref — positioned below the logs declarations so the dependency
  // is valid. The ref itself can be declared here; only the effect needs to be after.
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Prefer WS data if available, otherwise fallback to initial job data
  const currentStatus = (wsData?.status || job?.status || "pending") as JobStatus | "completed" | "failed" | "cancelled";
  const currentStage = (wsData?.stage || job?.status || "pending") as JobStatus;
  const overallProgress = wsData?.overall_progress ?? job?.progress ?? 0;
  const stageProgress = wsData?.stage_progress ?? job?.stage_progress ?? 0;
  const message = wsData?.message ?? job?.progress_message ?? "";
  const logs = wsData?.logs ?? (job as unknown as { logs?: string[] })?.logs ?? [];

  // Scroll the logs panel to the bottom whenever a new line arrives.
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const isCompleted = currentStatus === "completed";
  const isFailed = currentStatus === "failed";
  const isCancelled = currentStatus === "cancelled";
  const isRunning = !isCompleted && !isFailed && !isCancelled;

  const getStageState = (stageId: string) => {
    if (isCompleted) return "completed";
    if (isFailed || isCancelled) {
      // If we failed/cancelled, the current active stage is "failed" or "cancelled", rest are pending or completed
      const currentIndex = STAGES.findIndex(s => s.id === currentStage);
      const thisIndex = STAGES.findIndex(s => s.id === stageId);
      if (thisIndex < currentIndex) return "completed";
      if (thisIndex === currentIndex) return currentStatus; // failed or cancelled
      return "pending";
    }
    
    // Normal running logic
    const currentIndex = STAGES.findIndex(s => s.id === currentStage);
    const thisIndex = STAGES.findIndex(s => s.id === stageId);
    
    if (thisIndex < currentIndex) return "completed";
    if (thisIndex === currentIndex) return "active";
    return "pending";
  };

  const currentStageName = STAGES.find(s => s.id === currentStage)?.label || currentStage;

  const handleCancel = async () => {
    if (!job?.id) return;
    try {
      await api.cancelJob(job.id);
      onUpdate?.();
    } catch(err: unknown) { alert((err as Error).message); }
  };

  const handleRestart = async () => {
    if (!job?.id) return;
    try {
      await api.restartJob(job.id);
      onUpdate?.();
      // FIX: Reconnect the progress WS so we stream progress for the restarted job
      onReconnect?.();
    } catch(err: unknown) { alert((err as Error).message); }
  };

  const handleDelete = async () => {
    if (!job?.id || !confirm("Delete this job?")) return;
    try {
      await api.deleteJob(job.id);
      router.push("/");
    } catch(err: unknown) { alert((err as Error).message); }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
         <div className="space-y-1">
           <button onClick={() => router.push("/")} className="mb-2 flex items-center text-sm font-medium text-slate-400 hover:text-white transition-colors">
              <ArrowLeft className="mr-1 h-4 w-4" /> Back to jobs
           </button>
           <h1 className="text-2xl font-bold tracking-tight text-white">{job?.original_filename || "Job Details"}</h1>
           <div className="flex items-center gap-2">
             <Badge variant="outline" className={`
                px-2 py-0.5
                ${isCompleted ? "border-green-500 text-green-400" : ""}
                ${isFailed ? "border-red-500 text-red-400" : ""}
                ${isCancelled ? "border-yellow-500 text-yellow-400" : ""}
                ${isRunning ? "border-blue-500 text-blue-400" : ""}
             `}>
               {currentStatus.toUpperCase()}
             </Badge>
             {message && <span className="text-sm text-slate-400">{message}</span>}
           </div>
         </div>

         <div className="flex items-center gap-2">
           {isRunning && (
             <button onClick={handleCancel} className="flex h-9 items-center gap-2 rounded-md bg-slate-800 px-4 text-sm font-medium text-slate-200 transition-colors hover:bg-red-500/20 hover:text-red-400">
               <StopCircle className="h-4 w-4" /> Stop
             </button>
           )}
           {(isFailed || isCancelled) && (
             <button onClick={handleRestart} className="flex h-9 items-center gap-2 rounded-md bg-blue-600 px-4 text-sm font-medium text-white transition-colors hover:bg-blue-500">
               <RefreshCcw className="h-4 w-4" /> Restart
             </button>
           )}
           <button onClick={handleDelete} className="flex h-9 w-9 items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-slate-800 hover:text-red-400">
             <Trash2 className="h-4 w-4" />
           </button>
         </div>
      </div>

      {/* Main Grid */}
      <div className="grid gap-6 md:grid-cols-3">
        {/* Left: Timeline & Progress */}
        <Card className="col-span-2 border-white/10 bg-slate-900/50 p-6 backdrop-blur">
           <h3 className="mb-6 text-lg font-semibold text-slate-200">Pipeline Progress</h3>
           
           <div className="mb-8 space-y-2">
             <div className="flex items-center justify-between text-sm">
               <span className="font-medium text-slate-300">Overall: <span className="text-teal-400">{overallProgress.toFixed(1)}%</span></span>
               <span className="text-slate-500">Stage: {stageProgress.toFixed(1)}%</span>
             </div>
             <Progress value={overallProgress} className="h-2 bg-slate-800 *:bg-teal-500" />
           </div>

           <div className="space-y-0">
             {STAGES.map((stage, i) => {
               const state = getStageState(stage.id);
               const isLast = i === STAGES.length - 1;
               
               return (
                 <div key={stage.id} className="relative flex gap-4">
                   {!isLast && (
                      <div className={`absolute left-[11px] top-6 bottom-0 w-[2px] -translate-x-1/2 ${
                        state === "completed" || state === "active" ? "bg-teal-500/50" : "bg-slate-800"
                      }`} />
                   )}
                   
                   <div className="relative z-10 flex h-6 w-6 shrink-0 items-center justify-center bg-slate-900/50">
                     {state === "completed" && <CheckCircle2 className="h-5 w-5 text-teal-400" />}
                     {state === "active" && <Loader2 className="h-5 w-5 animate-spin text-blue-400" />}
                     {state === "failed" && <AlertCircle className="h-5 w-5 text-red-500" />}
                     {state === "cancelled" && <StopCircle className="h-5 w-5 text-yellow-500" />}
                     {state === "pending" && <Circle className="h-5 w-5 text-slate-700" />}
                   </div>
                   
                   <div className="pb-8 pt-0.5">
                     <p className={`text-sm font-medium ${
                        state === "active" ? "text-blue-400" :
                        state === "completed" ? "text-slate-200" : 
                        state === "failed" ? "text-red-400" : 
                        state === "cancelled" ? "text-yellow-400" : 
                        "text-slate-500"
                     }`}>
                       {stage.label}
                     </p>
                     
                     {state === "active" && (
                       <p className="mt-1 text-xs text-slate-400">{message || `Running ${currentStageName.toLowerCase()}...`}</p>
                     )}
                   </div>
                 </div>
               );
             })}
           </div>
        </Card>

        {/* Right: Actions / Info Box */}
        <div className="col-span-1 space-y-6">
          <Card className="border-white/10 bg-slate-900/50 p-6 backdrop-blur">
            <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-slate-200">
              <FileText className="h-5 w-5 text-violet-400" />
              Next Steps
            </h3>
            
            {isCompleted ? (
              <div className="space-y-3">
                <p className="text-sm text-slate-400">Processing finished successfully. You can now view the transcript or ask questions.</p>
                {job?.db_dir && (
                  <button onClick={() => router.push(`/chat/${job.id}`)} className="w-full rounded-md bg-teal-500 py-2 text-sm font-medium text-teal-950 hover:bg-teal-400 transition-colors shadow-[0_0_15px_rgba(20,184,166,0.3)]">
                    Ask AI Assistant
                  </button>
                )}
              </div>
            ) : isFailed ? (
              <div className="space-y-3">
                <p className="text-sm text-red-400">Job failed to complete. Check the logs below or restart to try again.</p>
                <button onClick={handleRestart} className="w-full rounded-md bg-blue-600 py-2 text-sm font-medium text-white hover:bg-blue-500">
                  Restart Pipeline
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-slate-400">The pipeline relies on AI models running in the background. Please wait while processing completes.</p>
                <div className="flex items-center justify-center p-4">
                  <div className="relative flex h-16 w-16 items-center justify-center rounded-full bg-slate-900 ring-1 ring-white/10">
                    <Loader2 className="h-8 w-8 animate-spin text-teal-500" />
                  </div>
                </div>
              </div>
            )}
          </Card>

          {/* Logs — fixed height, scrollable, auto-scrolls to bottom on new entries */}
          <Card className="flex flex-col border-white/10 bg-[#0d1323] p-0 backdrop-blur" style={{ height: "300px" }}>
            <div className="border-b border-white/5 bg-slate-900/80 px-4 py-2 text-xs font-semibold uppercase tracking-wider text-slate-400 rounded-t-xl">
              Execution Logs
            </div>
            <div className="flex-1 overflow-y-auto p-4 font-mono text-xs leading-relaxed text-slate-300">
              {logs && logs.length > 0 ? (
                logs.map((log: string, i: number) => <div key={i} className="mb-1 opacity-80">{log}</div>)
              ) : (
                <div className="text-slate-600 italic">No logs available.</div>
              )}
              {isRunning && <div className="mt-2 flex items-center gap-2 text-teal-500/60"><Loader2 className="h-3 w-3 animate-spin" /> waiting for output...</div>}
              {/* Sentinel element scrolled into view when logs update */}
              <div ref={logsEndRef} />
            </div>
          </Card>
        </div>

      </div>
    </div>
  );
}

// Ensure lucide icon doesn't break
import { AlertCircle } from "lucide-react";
