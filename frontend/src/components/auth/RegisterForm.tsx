"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api, ApiError } from "@/lib/api";

function RegisterFormInner() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const initialStep = searchParams.get("step") === "2" ? 2 : 1;
  const initialEmail = searchParams.get("email") || "";

  const [step, setStep] = useState(initialStep);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState(initialEmail);
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [otp, setOtp] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(step === 2 ? 60 : 0);

  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setTimeout(() => setResendCooldown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [resendCooldown]);

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (!email) { setError("Email is required"); return; }
    if (!username) { setError("Username is required"); return; }
    if (!password) { setError("Password is required"); return; }
    if (password !== confirm) { setError("Passwords do not match"); return; }
    if (password.length < 8) { setError("Password must be at least 8 characters"); return; }

    setLoading(true);
    try {
      await api.register(username, email, password);
      setStep(2);
      setResendCooldown(60);
    } catch (err) {
      if (err instanceof ApiError) {
        const detail = err.message;
        if (detail.includes("username") && detail.includes("taken")) {
          setError("Username is already taken.");
        } else if (detail.includes("email") && detail.includes("taken")) {
          setError("An account with this email already exists.");
        } else {
          setError(detail || "Registration failed. Please try again.");
        }
      } else {
        setError("Registration failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleVerifyOtp(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = await api.verifyRegisterOtp(email, otp);
      localStorage.setItem("meetbot_user", JSON.stringify(result));
      router.push("/");
    } catch (err) {
      if (err instanceof ApiError && err.status === 400) {
        setError("Invalid or expired code. Please try again.");
      } else {
        setError("Verification failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    if (resendCooldown > 0) return;
    setError(null);
    try {
      await api.resendOtp(email, "register");
      setResendCooldown(60);
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) {
        setError("Please wait before requesting another code.");
      } else {
        setError("Failed to resend code. Please try again.");
      }
    }
  }

  async function handleGoogleLogin() {
    setError(null);
    setLoading(true);
    try {
      const { signInWithGoogle, getFirebaseAuth } = await import("@/lib/firebase");
      const idToken = await signInWithGoogle();
      if (!idToken) {
        setError("Google sign-in was cancelled.");
        return;
      }

      const auth = await getFirebaseAuth();
      if (auth.currentUser && !auth.currentUser.emailVerified) {
        setError("Please verify your Google account email first.");
        return;
      }

      const result = await api.firebaseLogin(idToken);
      localStorage.setItem("meetbot_user", JSON.stringify(result));
      router.push("/");
    } catch {
      setError("Google sign-in failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  if (step === 2) {
    return (
      <div className="w-full max-w-sm space-y-6">
        <div className="space-y-1">
          <h1 className="font-display text-2xl font-semibold text-white">Verify your email</h1>
          <p className="text-sm text-zinc-400">
            We sent a 6-digit code to{" "}
            <span className="text-zinc-200 font-medium">{email}</span>.
          </p>
        </div>

        <form onSubmit={handleVerifyOtp} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
              Verification code
            </label>
            <input
              type="text"
              value={otp}
              onChange={(e) => setOtp(e.target.value.replace(/\D/g, "").slice(0, 6))}
              required
              inputMode="numeric"
              maxLength={6}
              autoComplete="one-time-code"
              className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 tracking-widest text-center text-lg"
              placeholder="000000"
            />
          </div>

          {error && (
            <p className="text-sm text-red-400 rounded-md border border-red-800 bg-red-950/40 px-3 py-2">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading || otp.length < 6}
            className="w-full rounded-md bg-amber-500 px-4 py-2 text-sm font-semibold text-black hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "Verifying…" : "Verify"}
          </button>
        </form>

        <p className="text-center text-xs text-zinc-500">
          Didn&apos;t receive it?{" "}
          <button
            onClick={handleResend}
            disabled={resendCooldown > 0}
            className="text-amber-500 hover:text-amber-400 disabled:text-zinc-600 transition-colors"
          >
            {resendCooldown > 0 ? `Resend in ${resendCooldown}s` : "Resend code"}
          </button>
        </p>

        <p className="text-center text-xs text-zinc-500">
          <button
            onClick={() => { setStep(1); setOtp(""); setError(null); }}
            className="text-amber-500 hover:text-amber-400 transition-colors"
          >
            ← Back
          </button>
        </p>
      </div>
    );
  }

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="space-y-1">
        <h1 className="font-display text-2xl font-semibold text-white">Create account</h1>
        <p className="text-sm text-zinc-400">Sign up to get started with MeetBot</p>
      </div>

      <form onSubmit={handleRegister} className="space-y-4">
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
            Username
          </label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            autoComplete="username"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500"
            placeholder="yourname"
          />
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
            Email
          </label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500"
            placeholder="you@example.com"
          />
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
            Password
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="new-password"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500"
            placeholder="••••••••"
          />
        </div>

        <div className="space-y-1.5">
          <label className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
            Confirm password
          </label>
          <input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            required
            autoComplete="new-password"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500"
            placeholder="••••••••"
          />
        </div>

        {error && (
          <p className="text-sm text-red-400 rounded-md border border-red-800 bg-red-950/40 px-3 py-2">
            {error}
          </p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-md bg-amber-500 px-4 py-2 text-sm font-semibold text-black hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Creating account…" : "Create account"}
        </button>
      </form>

      <div className="relative">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-zinc-700" />
        </div>
        <div className="relative flex justify-center text-xs text-zinc-500 uppercase tracking-wide">
          <span className="bg-[#07090f] px-2">or</span>
        </div>
      </div>

      <button
        onClick={handleGoogleLogin}
        disabled={loading}
        className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-200 hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
      >
        <svg className="h-4 w-4" viewBox="0 0 24 24">
          <path
            d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
            fill="#4285F4"
          />
          <path
            d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
            fill="#34A853"
          />
          <path
            d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
            fill="#FBBC05"
          />
          <path
            d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
            fill="#EA4335"
          />
        </svg>
        Continue with Google
      </button>

      <p className="text-center text-xs text-zinc-500">
        Already have an account?{" "}
        <a href="/login" className="text-amber-500 hover:text-amber-400 transition-colors">
          Sign in
        </a>
      </p>
    </div>
  );
}

export default function RegisterForm() {
  return (
    <Suspense>
      <RegisterFormInner />
    </Suspense>
  );
}
