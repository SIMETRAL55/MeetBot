import { useState, useEffect } from "react";
import { Job } from "@/types";
import { api } from "@/lib/api";

export function useJobPolling(intervalMs: number = 3000) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = async () => {
    try {
      const data = await api.getJobs();
      setJobs(data);
      setError(null);
    } catch (err: unknown) {
      setError((err as Error).message || "Failed to fetch jobs");
    } finally {
      if (loading) setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    
    // Auto-refresh interval
    const interval = setInterval(() => {
      // Only poll if there are active jobs or we're currently empty
      // Fast check to see if we have pending/running jobs
      setJobs((currentJobs) => {
        const hasActiveJobs = currentJobs.some((j) => 
          ["pending", "transcribing", "diarizing", "aligning", "indexing", "reindexing"].includes(j.status)
        );
        
        if (hasActiveJobs || currentJobs.length === 0) {
          fetchJobs();
        }
        return currentJobs;
      });
    }, intervalMs);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs]);

  return { jobs, loading, error, refetch: fetchJobs };
}
