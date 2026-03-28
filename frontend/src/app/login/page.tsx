"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Loader2, LogIn, UserPlus, Mic } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password) return;

    setLoading(true);
    setError(null);

    try {
      if (mode === "login") {
        const user = await api.login(username.trim(), password);
        localStorage.setItem("meetbot_user", JSON.stringify(user));
      } else {
        const user = await api.register(username.trim(), password, displayName.trim() || undefined);
        localStorage.setItem("meetbot_user", JSON.stringify(user));
      }
      router.push("/");
    } catch (err: unknown) {
      setError((err as Error).message || "Authentication failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-[calc(100vh-8rem)] items-center justify-center px-4">
      <div className="w-full max-w-md space-y-8">
        {/* Brand */}
        <div className="text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-amber-500/10 ring-1 ring-amber-500/25 shadow-lg shadow-amber-500/10">
            <Mic className="h-8 w-8 text-amber-400" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white">
            Welcome to MeetBot
          </h1>
          <p className="mt-2 text-sm text-slate-400">
            Audio Transcription & RAG Q&A — Local & Private
          </p>
        </div>

        {/* Card */}
        <div className="rounded-2xl border border-white/10 bg-slate-900/60 p-8 shadow-xl backdrop-blur">
          {/* Tab Switcher */}
          <div className="mb-6 flex rounded-lg bg-slate-950 p-1">
            <button
              onClick={() => { setMode("login"); setError(null); }}
              className={`flex-1 rounded-md py-2 text-sm font-semibold transition-colors ${
                mode === "login"
                  ? "bg-slate-800 text-amber-400 shadow"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              Sign In
            </button>
            <button
              onClick={() => { setMode("register"); setError(null); }}
              className={`flex-1 rounded-md py-2 text-sm font-semibold transition-colors ${
                mode === "register"
                  ? "bg-slate-800 text-amber-400 shadow"
                  : "text-slate-500 hover:text-slate-300"
              }`}
            >
              Create Account
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="username" className="mb-1.5 block text-sm font-medium text-slate-300">
                Username
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                autoComplete="username"
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-500 outline-none transition focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20"
              />
            </div>

            {mode === "register" && (
              <div>
                <label htmlFor="displayName" className="mb-1.5 block text-sm font-medium text-slate-300">
                  Display Name <span className="text-slate-500">(optional)</span>
                </label>
                <input
                  id="displayName"
                  type="text"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="How should we call you?"
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-500 outline-none transition focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20"
                />
              </div>
            )}

            <div>
              <label htmlFor="password" className="mb-1.5 block text-sm font-medium text-slate-300">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-500 outline-none transition focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/20"
              />
            </div>

            {error && (
              <div className="rounded-md border border-red-500/20 bg-red-500/10 px-4 py-2.5 text-sm text-red-400">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !username.trim() || !password}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-amber-500 py-2.5 text-sm font-semibold text-amber-950 transition-colors hover:bg-amber-400 disabled:opacity-50 shadow-[0_0_20px_rgba(245,158,11,0.25)]"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : mode === "login" ? (
                <><LogIn className="h-4 w-4" /> Sign In</>
              ) : (
                <><UserPlus className="h-4 w-4" /> Create Account</>
              )}
            </button>
          </form>
        </div>

        <p className="text-center text-xs text-slate-500">
          🔒 All processing runs on your server. Your data never leaves your machine.
        </p>
      </div>
    </div>
  );
}
