"use client";

import { useDropzone } from "react-dropzone";
import { Upload, X, FileAudio, Settings2 } from "lucide-react";
import { useState, useCallback } from "react";
import { api } from "@/lib/api";
import { Card } from "@/components/ui/card";

export function UploadDropzone({ onUploadSuccess }: { onUploadSuccess: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Options state
  const [language, setLanguage] = useState("");
  const [minSpeakers, setMinSpeakers] = useState<number | "">("");
  const [maxSpeakers, setMaxSpeakers] = useState<number | "">("");

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "audio/*": [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"],
      "video/*": [".mp4", ".mkv", ".webm"]
    },
    maxFiles: 1,
  });

  const handleUpload = async () => {
    if (!file) return;

    try {
      setIsUploading(true);
      setError(null);

      const options: Record<string, string | number> = {};
      if (language) options.language = language;
      if (minSpeakers !== "") options.minSpeakers = minSpeakers;
      if (maxSpeakers !== "") options.maxSpeakers = maxSpeakers;

      await api.uploadJob(file, options);
      setFile(null);
      // Reset options
      setLanguage("");
      setMinSpeakers("");
      setMaxSpeakers("");
      onUploadSuccess();
    } catch (err: unknown) {
      setError((err as Error).message || "Failed to upload file");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Card className="border-white/6 bg-white/5 p-6 backdrop-blur">
      <div className="mb-4 flex items-center gap-2 text-lg font-semibold tracking-tight text-slate-100">
        <Upload className="h-5 w-5 text-amber-400" />
        New Transcription
      </div>

      {!file ? (
        <div
          {...getRootProps()}
          className={`flex h-40 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors ${
            isDragActive
              ? "border-amber-500 bg-amber-500/10"
              : "border-slate-700 bg-slate-900/50 hover:bg-slate-800/80"
          }`}
        >
          <input {...getInputProps()} />
          <Upload className={`mb-3 h-8 w-8 ${isDragActive ? "text-amber-400" : "text-slate-400"}`} />
          <p className="text-sm font-medium text-slate-200">
            {isDragActive ? "Drop audio file here..." : "Drag & drop an audio file"}
          </p>
          <p className="mt-1 text-xs text-slate-500">
            Supports MP3, WAV, M4A, FLAC, MP4
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between rounded-lg border border-white/6 bg-slate-900/50 p-3">
            <div className="flex items-center gap-3 overflow-hidden">
              <FileAudio className="h-8 w-8 shrink-0 text-amber-400/70" />
              <div className="truncate">
                <p className="truncate text-sm font-medium text-slate-200">{file.name}</p>
                <p className="text-xs text-slate-500">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
              </div>
            </div>
            <button
              onClick={() => setFile(null)}
              disabled={isUploading}
              className="ml-4 shrink-0 rounded-full p-1 text-slate-400 hover:bg-slate-800 hover:text-white disabled:opacity-50"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="rounded-lg border border-white/6 bg-slate-950/50 p-4">
            <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-slate-400">
              <Settings2 className="h-3 w-3" />
              Options
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <label className="mb-1.5 block text-xs font-medium text-slate-300">Language</label>
                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-200 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20"
                >
                  <option value="">Auto-detect</option>
                  <option value="en">English</option>
                  <option value="ja">Japanese</option>
                  <option value="zh">Chinese</option>
                  <option value="ko">Korean</option>
                </select>
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium text-slate-300">Min Speakers</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={minSpeakers}
                  onChange={(e) => setMinSpeakers(e.target.value ? parseInt(e.target.value) : "")}
                  placeholder="Auto"
                  className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-200 placeholder:text-slate-600 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20"
                />
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium text-slate-300">Max Speakers</label>
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={maxSpeakers}
                  onChange={(e) => setMaxSpeakers(e.target.value ? parseInt(e.target.value) : "")}
                  placeholder="Auto"
                  className="w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-200 placeholder:text-slate-600 outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20"
                />
              </div>
            </div>
          </div>

          {error && <p className="text-sm font-medium text-red-500">{error}</p>}

          <button
            onClick={handleUpload}
            disabled={isUploading}
            className="w-full rounded-md bg-amber-500 py-2.5 text-sm font-medium text-amber-950 shadow-[0_0_15px_rgba(245,158,11,0.4)] transition-all hover:bg-amber-400 hover:shadow-[0_0_25px_rgba(245,158,11,0.5)] disabled:pointer-events-none disabled:opacity-50"
          >
            {isUploading ? "Uploading..." : "Start Processing"}
          </button>
        </div>
      )}
    </Card>
  );
}
