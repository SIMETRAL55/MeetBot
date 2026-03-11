"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { JobStatus } from "@/types";
import { getWsBaseUrl } from "@/lib/api";

export interface ProgressEvent {
  job_id: string;
  stage: JobStatus;
  stage_progress: number;
  overall_progress: number;
  logs: string[];
  status: string;
  message: string;
}

/**
 * WebSocket hook for real-time job progress streaming.
 *
 * FIX: Added `reconnectKey` counter so the hook can be forced to reconnect
 * after a reindex or restart changes the job status back to an active state.
 * Also uses configurable WS base URL instead of hardcoded localhost.
 *
 * Pass `enabled=false` to prevent the hook from opening a WebSocket at all
 * (e.g. when the job is already in a terminal state).  The caller can use the
 * initial REST-fetched job data directly in that case.
 */
export function useJobProgressWS(jobId: string, enabled: boolean = true) {
  const [data, setData] = useState<ProgressEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [reconnectKey, setReconnectKey] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  // Expose a reconnect method so callers can force a new WS connection
  // (e.g. after reindex changes job status from completed → reindexing).
  const reconnect = useCallback(() => {
    // Close existing connection if any
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close(1000);
    }
    setData(null);
    setError(null);
    setReconnectKey((k) => k + 1);
  }, []);

  useEffect(() => {
    // Skip WS connection entirely when disabled (e.g. job already completed).
    // The caller should use a REST fetch to populate UI state instead.
    if (!jobId || !enabled) return;

    // FIX: Use configurable WS base URL instead of hardcoded localhost
    const wsUrl = `${getWsBaseUrl()}/ws/jobs/${jobId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.ping) return; // ignore heartbeat
        if (payload.error) {
           setError(payload.error);
           return;
        }
        setData(payload);
      } catch (err) {
        console.error("Failed to parse WS message", err);
      }
    };

    ws.onerror = (event) => {
      // When the server closes immediately with code 4005 (job already completed)
      // the browser fires onerror before onclose with an empty event object.
      // Use warn-level so it does not clutter the console as a red error.
      console.warn("WS connection error (may be expected for completed/cancelled jobs)", event);
    };

    ws.onclose = (event) => {
      setConnected(false);
      // 4004 = Not found, 4005 = Already completed
      if (event.code === 4004) {
         setError("Job not found");
      }
      // If code is 1000/1001/4005 it's a normal closure
      wsRef.current = null;
    };

    return () => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close(1000);
      }
    };
  }, [jobId, reconnectKey, enabled]);

  return { data, error, connected, reconnect };
}
