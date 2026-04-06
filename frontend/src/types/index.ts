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
  retrieval_method?: "vector" | "pageindex";
  /** FIX: Backend returns status field for streaming/completed/stopped/interrupted */
  status?: string;
  created_at?: string;
  retrieval_level_note?: string;  // e.g. "segment" — set from retrieval_level_note WS event
  isStarred?: boolean;
}

/* ── PageIndex tree types ─────────────────────────────────────────────────── */

export interface PageIndexNode {
  node_id: string;
  level: number;
  title: string;
  summary: string | Record<string, unknown>;
  start_segment?: number;
  end_segment?: number;
  speakers?: string[];
  participants?: string[];   // ROOT level
  n_segments?: number;       // ROOT level
  children: PageIndexNode[];
}

export type PageIndexTree = PageIndexNode;

export interface QueryResponse {
  job_id: string;
  question: string;
  answer: string;
  sources: ChatSource[];
  llm_backend: string;
}

export interface AppSetting {
  key: string;
  value: string;
  description: string;
  type: "string" | "boolean" | "integer" | "password";
  sensitive: boolean;
  group: string;
  restart_required: boolean;
}

export interface UserProfile {
  user_id: string;
  username: string;
  display_name: string;
  is_admin: boolean;
  created_at: string | null;
  last_login: string | null;
}
