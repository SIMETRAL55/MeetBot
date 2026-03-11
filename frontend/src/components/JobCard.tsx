"use client";

import { Job } from "@/types";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  PlayCircle, StopCircle, RotateCcw, Search, Eye, 
  Trash2, AlertCircle, FileAudio, Clock, HardDrive, type LucideIcon 
} from "lucide-react";
import { api } from "@/lib/api";
import { formatDistanceToNow } from "date-fns";

const statusConfig: Record<string, { color: string; icon: LucideIcon }> = {
  pending: { color: "bg-slate-500", icon: Clock },
  transcribing: { color: "bg-blue-500", icon: PlayCircle },
  diarizing: { color: "bg-purple-500", icon: PlayCircle },
  aligning: { color: "bg-orange-500", icon: PlayCircle },
  indexing: { color: "bg-cyan-500", icon: PlayCircle },
  reindexing: { color: "bg-teal-500", icon: PlayCircle },
  completed: { color: "bg-green-500", icon: Eye },
  cancelled: { color: "bg-yellow-500", icon: StopCircle },
  failed: { color: "bg-red-500", icon: AlertCircle },
};

export function JobCard({ 
  job, 
  onJobUpdate 
}: { 
  job: Job; 
  onJobUpdate?: () => void 
}) {
  const config = statusConfig[job.status] || statusConfig.pending;
  const isRunning = ["pending", "transcribing", "diarizing", "aligning", "indexing", "reindexing"].includes(job.status);
  const isRestartable = ["cancelled", "failed"].includes(job.status);

  const handleCancel = async () => {
    if (!confirm("Are you sure you want to stop this job?")) return;
    try {
      await api.cancelJob(job.id);
      onJobUpdate?.();
    } catch (err: unknown) {
      alert((err as Error).message || "Failed to cancel job");
    }
  };

  const handleRestart = async () => {
    try {
      await api.restartJob(job.id);
      onJobUpdate?.();
    } catch (err: unknown) {
      alert((err as Error).message || "Failed to restart job");
    }
  };

  const handleDelete = async () => {
    if (!confirm(`Are you sure you want to delete "${job.original_filename}"? This cannot be undone.`)) return;
    try {
      await api.deleteJob(job.id);
      onJobUpdate?.();
    } catch (err: unknown) {
      alert((err as Error).message || "Failed to delete job");
    }
  };

  return (
    <Card className="group relative overflow-hidden border-white/10 bg-slate-900/40 p-5 transition-colors hover:border-slate-700 hover:bg-slate-900/60 backdrop-blur">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        
        {/* Left: Info */}
        <div className="flex flex-1 items-start gap-4">
          <div className={`mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-950 shadow-inner ring-1 ring-white/10`}>
            <FileAudio className={`h-5 w-5 ${config.color.replace('bg-', 'text-')}`} />
          </div>
          
          <div className="min-w-0 pr-4">
            <h3 className="truncate text-base font-semibold text-slate-200" title={job.original_filename}>
              {job.original_filename}
            </h3>
            
            <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-slate-500">
              {job.created_at && (
                <span className="flex items-center gap-1.5" title={new Date(job.created_at).toLocaleString()}>
                  <Clock className="h-3.5 w-3.5" />
                  {formatDistanceToNow(new Date(job.created_at), { addSuffix: true })}
                </span>
              )}
              {job.file_size && (
                <span className="flex items-center gap-1.5">
                  <HardDrive className="h-3.5 w-3.5" />
                  {(job.file_size / (1024 * 1024)).toFixed(1)} MB
                </span>
              )}
              {job.duration_seconds ? (
                <span className="flex items-center gap-1.5">
                  <PlayCircle className="h-3.5 w-3.5" />
                  {Math.floor(job.duration_seconds / 60)}:{Math.floor(job.duration_seconds % 60).toString().padStart(2, '0')}
                </span>
              ) : null}
            </div>
          </div>
        </div>

        {/* Center: Status & Progress */}
        <div className="flex min-w-[180px] flex-col items-start gap-2 sm:items-end">
          <Badge variant="outline" className={`capitalize ${config.color.replace('bg-', 'text-')} border-current bg-transparent px-2.5 py-0.5 text-xs font-semibold tracking-wide`}>
            {job.status}
          </Badge>
          
          {isRunning && job.status !== "pending" && (
            <div className="w-full sm:w-32">
              <Progress value={job.progress} className={`h-1.5 ${config.color.replace('bg-', '*:bg-')}`} />
              {job.progress_message && (
                <p className="mt-1.5 truncate text-right text-xs text-slate-400" title={job.progress_message}>
                  {job.progress_message}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2 border-t border-slate-800 pt-3 sm:border-t-0 sm:pt-0">
          {/* View link: available for all statuses so users can open the job page
              during processing to monitor progress / logs in real time. */}
          <a href={`/job/${job.id}`} className="inline-flex h-8 items-center justify-center rounded-md bg-violet-600/20 px-3 text-sm font-medium text-violet-400 transition-colors hover:bg-violet-600/30 hover:text-violet-300">
            <Eye className="mr-1.5 h-4 w-4" /> View
          </a>

          {job.status === "completed" && job.db_dir && (
            <a href={`/chat/${job.id}`} className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-teal-500/10 text-teal-400 transition-colors hover:bg-teal-500/20 hover:text-teal-300" title="Ask AI">
              <Search className="h-4 w-4" />
            </a>
          )}

          {isRunning && (
            <button
              onClick={handleCancel}
              className="inline-flex h-8 items-center justify-center rounded-md px-3 text-sm font-medium text-red-400 transition-colors hover:bg-red-500/10 hover:text-red-300"
            >
              <StopCircle className="mr-1.5 h-4 w-4" /> Stop
            </button>
          )}

          {isRestartable && (
            <button
              onClick={handleRestart}
              className="inline-flex h-8 items-center justify-center rounded-md px-3 text-sm font-medium text-blue-400 transition-colors hover:bg-blue-500/10 hover:text-blue-300"
            >
              <RotateCcw className="mr-1.5 h-4 w-4" /> Restart
            </button>
          )}

          <button
            onClick={handleDelete}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-500 transition-colors hover:bg-slate-800 hover:text-red-400"
            title="Delete Job"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </Card>
  );
}
