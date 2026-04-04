"use client";

import { useRef, useState, useEffect, useCallback, forwardRef, useImperativeHandle } from "react";
import { Play, Pause, Volume2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTranslations } from "next-intl";

export interface AudioPlayerHandle {
  seekTo: (seconds: number) => void;
}

interface AudioPlayerProps {
  src: string;
  className?: string;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/**
 * HTML5 audio player with programmatic seek support.
 *
 * Use the forwarded ref to call `seekTo(seconds)` from a parent component
 * (e.g. clicking a transcript timestamp).
 *
 * @example
 * const playerRef = useRef<AudioPlayerHandle>(null);
 * playerRef.current?.seekTo(42.5);
 * <AudioPlayer ref={playerRef} src={api.getAudioUrl(jobId)} />
 */
export const AudioPlayer = forwardRef<AudioPlayerHandle, AudioPlayerProps>(
  function AudioPlayer({ src, className }, ref) {
    const t = useTranslations("AudioPlayer");
    const audioRef = useRef<HTMLAudioElement>(null);
    const progressRef = useRef<HTMLDivElement>(null);
    const [playing, setPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [volume, setVolume] = useState(1);
    const [error, setError] = useState<string | null>(null);

    // Expose seekTo via ref
    useImperativeHandle(ref, () => ({
      seekTo(seconds: number) {
        const audio = audioRef.current;
        if (!audio) return;
        audio.currentTime = Math.max(0, Math.min(seconds, audio.duration || seconds));
        audio.play().catch(() => null);
        setPlaying(true);
      },
    }));

    const togglePlay = useCallback(() => {
      const audio = audioRef.current;
      if (!audio) return;
      if (playing) {
        audio.pause();
      } else {
        audio.play().catch(() => null);
      }
    }, [playing]);

    const handleProgressClick = useCallback(
      (e: React.MouseEvent<HTMLDivElement>) => {
        const bar = progressRef.current;
        const audio = audioRef.current;
        if (!bar || !audio || !duration) return;
        const rect = bar.getBoundingClientRect();
        const pct = (e.clientX - rect.left) / rect.width;
        audio.currentTime = pct * duration;
      },
      [duration]
    );

    useEffect(() => {
      const audio = audioRef.current;
      if (!audio) return;

      const onPlay = () => setPlaying(true);
      const onPause = () => setPlaying(false);
      const onTimeUpdate = () => setCurrentTime(audio.currentTime);
      const onLoadedMetadata = () => setDuration(audio.duration);
      const onError = () => setError("Failed to load audio file");

      audio.addEventListener("play", onPlay);
      audio.addEventListener("pause", onPause);
      audio.addEventListener("timeupdate", onTimeUpdate);
      audio.addEventListener("loadedmetadata", onLoadedMetadata);
      audio.addEventListener("error", onError);

      return () => {
        audio.removeEventListener("play", onPlay);
        audio.removeEventListener("pause", onPause);
        audio.removeEventListener("timeupdate", onTimeUpdate);
        audio.removeEventListener("loadedmetadata", onLoadedMetadata);
        audio.removeEventListener("error", onError);
      };
    }, []);

    const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

    return (
      <div
        className={cn(
          "flex flex-col gap-2 rounded-xl border border-white/10 bg-slate-900/80 p-4 backdrop-blur",
          className
        )}
      >
        <div className="flex items-center gap-2 text-xs font-medium text-slate-400">
          <Volume2 className="h-3.5 w-3.5 text-indigo-400" />
          <span>{t("playback")}</span>
        </div>

        {/* Hidden native audio element */}
        {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
        <audio ref={audioRef} src={src} preload="metadata" />

        {error ? (
          <p className="text-xs text-red-400">{t("loadError")}</p>
        ) : (
          <>
            {/* Progress bar */}
            <div
              ref={progressRef}
              className="relative h-2 w-full cursor-pointer rounded-full bg-slate-700"
              onClick={handleProgressClick}
            >
              <div
                className="h-full rounded-full bg-indigo-500 transition-all"
                style={{ width: `${progressPct}%` }}
              />
              {/* Thumb */}
              <div
                className="absolute top-1/2 h-3.5 w-3.5 -translate-y-1/2 rounded-full bg-indigo-400 shadow-md ring-2 ring-indigo-500/30 transition-all"
                style={{ left: `calc(${progressPct}% - 7px)` }}
              />
            </div>

            {/* Controls row */}
            <div className="flex items-center justify-between">
              <button
                onClick={togglePlay}
                className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-500/20 text-indigo-400 transition-colors hover:bg-indigo-500/40"
                aria-label={playing ? t("pause") : t("play")}
              >
                {playing ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4 translate-x-px" />
                )}
              </button>

              <span className="font-mono text-xs text-slate-400">
                {formatTime(currentTime)}
                <span className="mx-1 text-slate-600">/</span>
                {formatTime(duration)}
              </span>

              {/* Volume */}
              <div className="flex items-center gap-1.5">
                <Volume2 className="h-3 w-3 text-slate-500" />
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={volume}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setVolume(v);
                    if (audioRef.current) audioRef.current.volume = v;
                  }}
                  className="h-1 w-16 cursor-pointer accent-indigo-500"
                  aria-label={t("volume")}
                />
              </div>
            </div>
          </>
        )}
      </div>
    );
  }
);
