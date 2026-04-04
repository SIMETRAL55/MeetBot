"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError } from "@/lib/api";

export default function ForgotPasswordForm() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [resetToken, setResetToken] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);

  useEffect(() => {
    if (resendCooldown <= 0) return;
    const timer = setTimeout(() => setResendCooldown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [resendCooldown]);

  async function handleSendOtp(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await api.forgotPassword(email);
      setStep(2);
      setResendCooldown(60);
    } catch {
      // Enum-safe: always show success to prevent email enumeration
      setStep(2);
      setResendCooldown(60);
    } finally {
      setLoading(false);
    }
  }

  async function handleVerifyOtp(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = await api.verifyResetOtp(email, otp);
      setResetToken(result.reset_token);
      setStep(3);
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

  async function handleResetPassword(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (newPassword.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    if (newPassword !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      await api.resetPassword(resetToken, newPassword);
      router.push("/login?message=password_updated");
    } catch (err) {
      if (err instanceof ApiError && err.status === 400) {
        setError("Reset link expired. Please start over.");
      } else {
        setError("Password reset failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    if (resendCooldown > 0) return;
    setError(null);
    try {
      await api.resendOtp(email, "reset");
      setResendCooldown(60);
    } catch (err) {
      if (err instanceof ApiError && err.status === 429) {
        setError("Please wait before requesting another code.");
      } else {
        setError("Failed to resend code. Please try again.");
      }
    }
  }

  if (step === 2) {
    return (
      <div className="w-full max-w-sm space-y-6">
        <div className="space-y-1">
          <h1 className="font-display text-2xl font-semibold text-white">Check your email</h1>
          <p className="text-sm text-zinc-400">
            We sent a 6-digit code to{" "}
            <span className="text-zinc-200 font-medium">{email}</span>.
          </p>
        </div>

        <form onSubmit={handleVerifyOtp} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
              Reset code
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
            {loading ? "Verifying…" : "Verify code"}
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

  if (step === 3) {
    return (
      <div className="w-full max-w-sm space-y-6">
        <div className="space-y-1">
          <h1 className="font-display text-2xl font-semibold text-white">New password</h1>
          <p className="text-sm text-zinc-400">Choose a new password for your account.</p>
        </div>

        <form onSubmit={handleResetPassword} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-zinc-300 uppercase tracking-wide">
              New password
            </label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
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
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
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
            {loading ? "Resetting…" : "Reset password"}
          </button>
        </form>
      </div>
    );
  }

  // Step 1: Email input
  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="space-y-1">
        <h1 className="font-display text-2xl font-semibold text-white">Forgot password</h1>
        <p className="text-sm text-zinc-400">
          Enter your email and we&apos;ll send you a reset code.
        </p>
      </div>

      <form onSubmit={handleSendOtp} className="space-y-4">
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
          {loading ? "Sending…" : "Send reset code"}
        </button>
      </form>

      <p className="text-center text-xs text-zinc-500">
        <a href="/login" className="text-amber-500 hover:text-amber-400 transition-colors">
          Back to sign in
        </a>
      </p>
    </div>
  );
}
