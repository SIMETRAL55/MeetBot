import { Job, QueryResponse } from "../types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080/api";

/**
 * Derive the WebSocket base URL from the API base URL so that WS connections
 * work regardless of whether the backend is on localhost or a remote host.
 * FIX: Previously WS URLs were hardcoded to ws://localhost:8080 which broke
 * when the frontend was served from a different host.
 */
export function getWsBaseUrl(): string {
  try {
    const httpUrl = API_BASE_URL.replace(/\/api\/?$/, "");
    const url = new URL(httpUrl);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.origin;
  } catch {
    return "ws://localhost:8080";
  }
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function fetcher<T>(endpoint: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        "Accept": "application/json",
        ...options?.headers,
      },
    });
  } catch (networkErr) {
    // Network-level failure (CORS, server unreachable, etc.)
    throw new ApiError(0, `Network error: ${(networkErr as Error).message}`);
  }

  if (!response.ok) {
    if (response.status === 401 && typeof window !== 'undefined' && window.location.pathname !== '/login') {
      localStorage.removeItem("meetbot_user");
      window.location.href = '/login';
      return {} as T;
    }

    // FIX: Read the body as text exactly once, then try to parse as JSON.
    // Previously response.json() consumed the stream, making the subsequent
    // response.text() call throw "body stream already read".
    let message = "An error occurred";
    try {
      const rawText = await response.text();
      try {
        const errorData = JSON.parse(rawText);
        message = errorData.detail || errorData.message || rawText || message;
      } catch {
        message = rawText || message;
      }
    } catch {
      // body unreadable — keep default message
    }
    throw new ApiError(response.status, message);
  }

  // Handle empty responses
  const text = await response.text();
  if (!text) return {} as T;
  
  return JSON.parse(text) as T;
}

export const api = {
  // --- Dashboard / Job List ---
  
  getJobs: () => fetcher<Job[]>("/jobs"),
  
  uploadJob: async (file: File, options?: { language?: string, minSpeakers?: number, maxSpeakers?: number }) => {
    const formData = new FormData();
    formData.append("file", file);
    if (options?.language) formData.append("language", options.language);
    if (options?.minSpeakers) formData.append("min_speakers", options.minSpeakers.toString());
    if (options?.maxSpeakers) formData.append("max_speakers", options.maxSpeakers.toString());

    return fetcher<{ job_id: string; status: string; message: string }>("/jobs/upload", {
      method: "POST",
      // Do not set Content-Type header with FormData, browser will set it with boundary
      body: formData,
    });
  },

  deleteJob: (jobId: string) => fetcher<{ status: string }>(`/jobs/${jobId}`, {
    method: "DELETE"
  }),

  // --- Job Detail / Progress ---

  getJobStatus: async (jobId: string): Promise<Job> => {
    const raw = await fetcher<Record<string, unknown>>(`/jobs/${jobId}/status`);
    // Normalize: the status endpoint historically returned "job_id" instead of
    // "id". Both are now returned by the backend, but guard here as a
    // defensive fallback so job.id is never undefined in the UI.
    if (!raw.id && raw.job_id) raw.id = raw.job_id;
    return raw as unknown as Job;
  },
  
  cancelJob: (jobId: string) => fetcher<{ job_id: string; status: string; message: string }>(`/jobs/${jobId}/cancel`, {
    method: "POST"
  }),

  restartJob: (jobId: string) => fetcher<{ job_id: string; status: string; message: string }>(`/jobs/${jobId}/restart`, {
    method: "POST"
  }),

  reindexJob: (jobId: string) => fetcher<{ job_id: string; status: string; message: string }>(`/jobs/${jobId}/reindex`, {
    method: "POST"
  }),

  /** Return the URL that the browser should use to stream the audio file. */
  getAudioUrl: (jobId: string): string => `${API_BASE_URL}/jobs/${jobId}/audio`,

  // --- Transcript ---

  getTranscript: (jobId: string, type: "aligned" | "transcription" | "diarization" = "aligned") => {
    // Instead of downloading as a file, we want to fetch and parse the JSON for the editor.
    return fetcher<{ segments: import("../types").Segment[] } | import("../types").Segment[]>(
      `/jobs/${jobId}/download?type=${type}`
    );
  },

  updateSegment: (jobId: string, segmentId: string, data: { text?: string; speaker?: string }) => {
    return fetcher<{ status: string }>(`/jobs/${jobId}/segments/${segmentId}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
  },

  // --- RAG Chat ---

  queryJob: (jobId: string, query: string, llmMode: "local" | "hf" = "local") => {
    return fetcher<QueryResponse>(`/jobs/${jobId}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ q: query, llm_mode: llmMode }),
    });
  },
  
  getChatHistory: (jobId: string) => fetcher<import("../types").ChatMessage[]>(`/jobs/${jobId}/chat/history`),

  // --- Auth ---

  login: (username: string, password: string) => {
    return fetcher<{ user_id: string; username: string; display_name: string; is_admin: boolean }>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
  },

  register: (username: string, password: string, displayName?: string) => {
    return fetcher<{ user_id: string; username: string; display_name: string; is_admin: boolean }>("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password, display_name: displayName }),
    });
  },
};
