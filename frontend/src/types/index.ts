export type JobStatus = 
  | "pending" 
  | "transcribing" 
  | "diarizing" 
  | "aligning" 
  | "indexing" 
  | "reindexing"
  | "completed" 
  | "cancelled"
  | "failed";

export interface Job {
  id: string;
  original_filename: string;
  status: JobStatus;
  progress: number;
  stage_progress?: number;
  progress_message?: string;
  error_message?: string;
  db_dir?: string;
  result_json_path?: string;
  duration_seconds?: number | null;
  file_size?: number | null;
  created_at?: string;
  pageindex_status?: string | null;  // "pending" | "building" | "ready" | "failed" | null
}

export interface Segment {
  start: number;
  end: number;
  speaker: string;
  text: string;
}

export interface ChatSource {
  segment_index: number;
  start: number;
  end: number;
  speaker: string;
  text: string;
  distance: number;
  relevance: string;
  node_title?: string;  // PageIndex section title
  node_id?: string;     // PageIndex node ID
}

export interface ChatMessage {
  id?: string;
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  llm_backend?: string;
  /** FIX: Backend returns status field for streaming/completed/stopped/interrupted */
  status?: string;
  created_at?: string;
}

export interface QueryResponse {
  job_id: string;
  question: string;
  answer: string;
  sources: ChatSource[];
  llm_backend: string;
}
