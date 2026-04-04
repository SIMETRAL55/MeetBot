"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError } from "@/lib/api";

export default function LoginForm() {
  const router = useRouter();
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Email verification pending state (Google OAuth only)
  const [verificationPending, setVerificationPending] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);

  // OTP-unverified state (local accounts)
  const [showUnverified, setShowUnverified] = useState(false);
  const [unverifiedEmail, setUnverifiedEmail] = useState("");

  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setTimeout(() => setResendCooldown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [resendCooldown]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = await api.login(identifier, password);
      localStorage.setItem("meetbot_user", JSON.stringify(result));
      router.push("/");
    } catch (err) {
      if (err instanceof ApiError && err.status === 423) {
        setError("Account temporarily locked. Too many failed attempts.");
      } else if (err instanceof ApiError && err.status === 403 && err.message === "email_not_verified") {
        const email = (err.data?.email as string) || "";
        setUnverifiedEmail(email);
        setShowUnverified(true);
      } else {
        setError("Invalid username/email or password.");
      }
    } finally {
      setLoading(false);
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

      // Check email verification status from the Firebase session
      const auth = await getFirebaseAuth();
      if (auth.currentUser && !auth.currentUser.emailVerified) {
        setVerificationPending(true);
        return;
      }

      const result = await api.firebaseLogin(idToken);
      localStorage.setItem("meetbot_user", JSON.stringify(result));
      router.push("/");
    } catch (err) {
      if (err instanceof ApiError && err.status === 403 && err.message === "EMAIL_NOT_VERIFIED") {
        setVerificationPending(true);
      } else {
        setError("Google sign-in failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleVerified() {
    setError(null);
    setLoading(true);
    try {
      const { reloadAndCheckVerified } = await import("@/lib/firebase");
      const freshToken = await reloadAndCheckVerified();
      if (!freshToken) {
        setError("Email not verified yet. Please check your inbox and click the link.");
        return;
      }
      const result = await api.firebaseLogin(freshToken);
      localStorage.setItem("meetbot_user", JSON.stringify(result));
      router.push("/");
    } catch {
      setError("Login failed after verification. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    if (resendCooldown > 0) return;
    setError(null);
    try {
      const { sendVerificationEmail } = await import("@/lib/firebase");
      await sendVerificationEmail();
      setResendCooldown(60);
    } catch {
      setError("Failed to resend verification email.");
    }
  }

  async function handleResendRegisterOtp() {
    if (!unverifiedEmail) return;
    setError(null);
    try {
      await api.resendOtp(unverifiedEmail, "register");
      router.push(`/register?step=2&email=${encodeURIComponent(unverifiedEmail)}`);
    } catch {
      setError("Failed to resend verification code. Please try again.");
    }
  }

  if (showUnverified) {
    return (
      <div className="w-full max-w-sm space-y-6">
        <div className="space-y-1">
          <h1 className="font-display text-2xl font-semibold text-white">Verify your email</h1>
          <p className="text-sm text-zinc-400">Please verify your email before signing in.</p>
        </div>

        <div className="rounded-md border border-amber-800/50 bg-amber-950/30 px-4 py-3 text-sm text-zinc-300">
          A verification code was sent to <span className="text-white font-medium">{unverifiedEmail}</span>.
        </div>

        {error && (
          <p className="text-sm text-red-400 rounded-md border border-red-800 bg-red-950/40 px-3 py-2">
            {error}
          </p>
        )}

        <button
          onClick={handleResendRegisterOtp}
          className="w-full rounded-md bg-amber-500 px-4 py-2 text-sm font-semibold text-black hover:bg-amber-400 transition-colors"
        >
          Resend verification code
        </button>

        <p className="text-center text-xs text-zinc-500">
          <button
            onClick={() => setShowUnverified(false)}
            className="text-amber-500 hover:text-amber-400 transition-colors"
          >
            ← Back to sign in
          </button>
        </p>
      </div>
    );
  }

  if (verificationPending) {
    return (
      <div className="w-full max-w-sm space-y-6">
        <div className="space-y-1">
          <h1 className="font-display text-2xl font-semibold text-white">Verify your email</h1>
          <p className="text-sm text-zinc-400">Your email address needs to be verified before you can sign in.</p>
        </div>

        <div className="rounded-md border border-amber-800/50 bg-amber-950/30 px-4 py-3 text-sm text-zinc-300">
          Check your inbox for a verification link, click it, then return here.
        </div>

        {error && (
          <p className="text-sm text-red-400 rounded-md border border-red-800 bg-red-950/40 px-3 py-2">
            {error}
          </p>
        )}

        <button
          onClick={handleVerified}
          disabled={loading}
          className="w-full rounded-md bg-amber-500 px-4 py-2 text-sm font-semibold text-black hover:bg-amber-400 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Checking…" : "I've verified my email"}
        </button>

        <button
          onClick={handleResend}
          disabled={loading || resendCooldown > 0}
          className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm font-medium text-zinc-400 hover:bg-zinc-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {resendCooldown > 0 ? `Resend in ${resendCooldown}s` : "Resend verification email"}
        </button>

        <p className="text-center text-xs text-zinc-500">
          Wrong account?{" "}
          <button
            onClick={() => setVerificationPending(false)}
            className="text-amber-500 hover:text-amber-400 transition-colors"
          >
            Go back
          </button>
        </p>
      </div>
    );
  }

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="space-y-1">
        <h1 className="font-display text-2xl font-semibold text-white">Sign in</h1>
        <p className="text-sm text-zinc-400">Enter your username or email to continue</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
            Username or Email
          </label>
          <input
            type="text"
            value={identifier}
            onChange={(e) => setIdentifier(e.target.value)}
            required
            autoComplete="username"
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
            autoComplete="current-password"
            className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-white placeholder-zinc-500 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500"
            placeholder="••••••••"
          />
        </div>

        <div className="flex items-center justify-end">
          <a
            href="/forgot-password"
            className="text-xs text-amber-500 hover:text-amber-400 transition-colors"
          >
            Forgot password?
          </a>
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
          {loading ? "Signing in…" : "Sign in"}
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
        Don&apos;t have an account?{" "}
        <a href="/register" className="text-amber-500 hover:text-amber-400 transition-colors">
          Register
        </a>
      </p>
    </div>
  );
}
