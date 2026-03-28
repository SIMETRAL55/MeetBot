"use client";

import { useJobPolling } from "@/lib/hooks/useJobPolling";
import { UploadDropzone } from "@/components/UploadDropzone";
import { JobCard } from "@/components/JobCard";
import { Loader2, Mic } from "lucide-react";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function DashboardPage() {
  const router = useRouter();
  const { jobs, loading, error, refetch } = useJobPolling();

  useEffect(() => {
    const user = localStorage.getItem("meetbot_user");
    if (!user) {
      router.push("/login");
    }
  }, [router]);

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      
      <div className="flex flex-col items-start justify-between gap-4 md:flex-row md:items-center">
        <div>
          <h1 className="font-display text-3xl font-bold tracking-tight text-white">Dashboard</h1>
          <p className="mt-1 text-slate-400">Manage your transcriptions and AI jobs.</p>
        </div>
      </div>

      <UploadDropzone onUploadSuccess={refetch} />

      <div className="space-y-4">
        <h2 className="border-l-2 border-amber-500/30 pl-3 text-xl font-semibold tracking-tight text-slate-200">Recent Jobs</h2>
        
        {loading ? (
          <div className="flex justify-center p-12">
            <Loader2 className="h-8 w-8 animate-spin text-amber-500" />
          </div>
        ) : error ? (
          <div className="rounded-lg border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-400">
            {error}
          </div>
        ) : jobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-900/50 p-12 text-center">
            <div className="mb-4 rounded-full bg-slate-800 p-3 ring-1 ring-white/10">
              <Mic className="h-6 w-6 text-slate-400" />
            </div>
            <h3 className="text-lg font-medium text-slate-200">No transcriptions yet</h3>
            <p className="mt-1 text-sm text-slate-500">Upload an audio file above to get started.</p>
          </div>
        ) : (
          <div className="grid gap-4">
            {jobs.map((job) => (
              <JobCard key={job.id} job={job} onJobUpdate={refetch} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
