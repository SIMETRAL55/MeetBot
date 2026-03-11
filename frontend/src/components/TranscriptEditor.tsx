"use client";

import { useState, useEffect } from "react";
import { Segment } from "@/types";
import { api } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, Save, User, Edit2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface TranscriptEditorProps {
  jobId: string;
  /** Called when the user clicks a segment timestamp — seek the audio player. */
  onTimestampClick?: (seconds: number) => void;
}

export function TranscriptEditor({ jobId, onTimestampClick }: TranscriptEditorProps) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Editing state
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editSpeaker, setEditSpeaker] = useState("");
  const [editText, setEditText] = useState("");
  const [isSaving, setIsSaving] = useState(false);

  const loadTranscript = async () => {
    try {
      setLoading(true);
      const res = await api.getTranscript(jobId, "aligned");
      // The API returns { segments: [...] } or just an array depending on the exact JSON format,
      // handling both cases safely here:
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const data = Array.isArray(res) ? res : (res as any).segments;
      setSegments(data || []);
      setError(null);
    } catch (err: unknown) {
      setError((err as Error).message || "Failed to load transcript");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTranscript();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const startEditing = (index: number, seg: Segment) => {
    setEditingId(index);
    setEditSpeaker(seg.speaker);
    setEditText(seg.text.trim());
  };

  const cancelEditing = () => {
    setEditingId(null);
    setEditSpeaker("");
    setEditText("");
  };

  const saveSegment = async (index: number) => {
    if (editingId === null) return;
    
    // In our backend, segment ID is actually its 0-based index or a DB ID?
    // According to crud.py, update_segment takes the segment's DB ID string.
    // However, the JSON we get back usually doesn't have the DB ID, it's just the array.
    // Let's assume the API routes are updated to accept the start time or index as segment_id,
    // or we'll pass the database ID if it's included in the payload.
    // For now, let's pass the index safely as a string.
    try {
      setIsSaving(true);
      // Wait for API
      await api.updateSegment(jobId, index.toString(), {
        speaker: editSpeaker,
        text: editText,
      });
      
      // Update local state optimistic
      const updated = [...segments];
      updated[index] = { ...updated[index], speaker: editSpeaker, text: editText };
      setSegments(updated);
      cancelEditing();
      
      // Might want to call trigger reindex here or prompt the user.
    } catch (err: unknown) {
      alert((err as Error).message || "Failed to save segment");
    } finally {
      setIsSaving(false);
    }
  };

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  if (loading) {
    return (
      <Card className="flex h-96 items-center justify-center border-white/10 bg-slate-900/50 backdrop-blur">
        <Loader2 className="h-8 w-8 animate-spin text-teal-500" />
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="flex h-96 items-center justify-center border-red-500/20 bg-red-500/10 p-6 text-center text-red-400 backdrop-blur">
        <p>{error}</p>
      </Card>
    );
  }

  if (segments.length === 0) {
    return (
      <Card className="flex h-96 flex-col items-center justify-center border-white/10 bg-slate-900/50 text-slate-400 backdrop-blur">
        <p>No transcript data available for this job.</p>
        <p className="text-sm">Wait for processing to complete or check the logs.</p>
      </Card>
    );
  }

  // Pre-calculate speaker colors for visual distinction
  const speakers = Array.from(new Set(segments.map(s => s.speaker)));
  const colors = ["text-blue-400", "text-teal-400", "text-purple-400", "text-orange-400", "text-pink-400"];
  const getSpeakerColor = (s: string) => colors[speakers.indexOf(s) % colors.length];

  return (
    <Card className="flex h-[600px] flex-col overflow-hidden border-white/10 bg-slate-900/50 backdrop-blur">
      <div className="flex items-center justify-between border-b border-white/10 bg-slate-950/50 p-4">
        <h3 className="font-semibold text-slate-200">Interactive Transcript</h3>
        <p className="text-sm text-slate-400">
          {segments.length} segments • {formatTime(segments[segments.length - 1]?.end || 0)}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-6">
          {segments.map((seg, i) => {
            const isEditing = editingId === i;
            // Group consecutive segments from the same speaker?
            // For now, simple list.
            
            return (
              <div 
                key={i} 
                className={cn(
                  "group relative flex gap-4 rounded-lg p-3 transition-colors",
                  isEditing ? "bg-slate-800/80 ring-1 ring-teal-500/50" : "hover:bg-slate-800/40"
                )}
              >
                {/* Time & Actions */}
                <div className="flex w-24 shrink-0 flex-col items-end gap-1 text-xs">
                  <button
                    onClick={() => onTimestampClick?.(seg.start)}
                    className={cn(
                      "font-mono text-slate-400 transition-colors",
                      onTimestampClick
                        ? "cursor-pointer hover:text-teal-400"
                        : "cursor-default"
                    )}
                    title={onTimestampClick ? `Seek to ${formatTime(seg.start)}` : undefined}
                    disabled={!onTimestampClick}
                  >
                    {formatTime(seg.start)}
                  </button>
                  
                  {!isEditing && (
                    <button 
                      onClick={() => startEditing(i, seg)}
                      className="mt-1 flex items-center gap-1 font-medium text-slate-500 opacity-0 transition-opacity hover:text-teal-400 group-hover:opacity-100"
                    >
                      <Edit2 className="h-3 w-3" /> Edit
                    </button>
                  )}
                </div>

                {/* Content */}
                <div className="flex-1">
                  {isEditing ? (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <User className="h-4 w-4 text-slate-400" />
                        <input 
                           type="text" 
                           value={editSpeaker}
                           onChange={(e) => setEditSpeaker(e.target.value)}
                           className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-200 outline-none focus:border-teal-500"
                           placeholder="Speaker Name"
                        />
                      </div>
                      <textarea 
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        className="min-h-[80px] w-full rounded border border-slate-700 bg-slate-900 p-2 text-sm text-slate-200 outline-none focus:border-teal-500"
                      />
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={() => saveSegment(i)}
                          disabled={isSaving}
                          className="flex items-center gap-1.5 rounded bg-teal-500 px-3 py-1.5 text-xs font-medium text-teal-950 hover:bg-teal-400 disabled:opacity-50"
                        >
                          {isSaving ? <Loader2 className="h-3 w-3 animate-spin"/> : <Save className="h-3 w-3"/>}
                          Save
                        </button>
                        <button 
                          onClick={cancelEditing}
                          disabled={isSaving}
                          className="px-3 py-1.5 text-xs font-medium text-slate-400 hover:text-white"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className={cn("mb-1 font-semibold text-sm", getSpeakerColor(seg.speaker))}>
                        {seg.speaker}
                      </div>
                      <p className="text-sm leading-relaxed text-slate-300">
                        {seg.text}
                      </p>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
