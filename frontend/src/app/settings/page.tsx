"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { api } from "../../lib/api";
import { BackButton } from "../../components/BackButton";
import { AppSetting, UserProfile } from "../../types";

const GROUP_ORDER = [
  "Authentication",
  "Models",
  "Backends",
  "RAG & Retrieval",
  "Indexing Performance",
  "Web Server",
  "PageIndex",
];

type SaveState = "idle" | "saving" | "saved" | "error";

interface SettingRowProps {
  setting: AppSetting;
  onSave: (key: string, value: string) => Promise<void>;
}

function SettingRow({ setting, onSave }: SettingRowProps) {
  const t = useTranslations("Settings");
  const [localValue, setLocalValue] = useState(setting.value);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [showPassword, setShowPassword] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync if setting prop changes (after parent refreshes)
  useEffect(() => {
    if (setting.type !== "password" || !setting.sensitive) {
      setLocalValue(setting.value);
    }
  }, [setting.value, setting.type, setting.sensitive]);

  const handleChange = useCallback(
    (newValue: string) => {
      setLocalValue(newValue);
      if (setting.type === "boolean") {
        // Immediate save for toggles
        setSaveState("saving");
        onSave(setting.key, newValue)
          .then(() => {
            setSaveState("saved");
            setTimeout(() => setSaveState("idle"), 2000);
          })
          .catch(() => setSaveState("error"));
        return;
      }
      // Debounce for text inputs
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        setSaveState("saving");
        onSave(setting.key, newValue)
          .then(() => {
            setSaveState("saved");
            setTimeout(() => setSaveState("idle"), 2000);
          })
          .catch(() => setSaveState("error"));
      }, 600);
    },
    [setting.key, setting.type, onSave]
  );

  const inputCls =
    "w-full border border-white/6 bg-[#080b14] text-slate-200 text-sm rounded-lg px-3 py-2 " +
    "focus:outline-none focus:border-indigo-500/40 focus:ring-1 focus:ring-indigo-500/20 transition-colors";

  return (
    <div className="py-3 border-b border-white/4 last:border-0">
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
              {setting.key}
            </span>
            {setting.restart_required && (
              <span className="text-[9px] text-indigo-400/70 ring-1 ring-indigo-500/20 rounded px-1 py-0.5">
                ⚠ {t("restartRequired")}
              </span>
            )}
            {saveState === "saving" && (
              <span className="text-[10px] text-slate-500 animate-pulse">{t("saving")}</span>
            )}
            {saveState === "saved" && (
              <span className="text-[10px] text-green-400">{t("saved")}</span>
            )}
            {saveState === "error" && (
              <span className="text-[10px] text-red-400">{t("error")}</span>
            )}
          </div>

          {setting.type === "boolean" ? (
            <div className="flex items-center gap-3">
              <button
                onClick={() => handleChange(localValue === "true" ? "false" : "true")}
                className={
                  "relative inline-flex h-5 w-9 items-center rounded-full transition-colors " +
                  (localValue === "true"
                    ? "bg-indigo-500/70"
                    : "bg-white/10")
                }
              >
                <span
                  className={
                    "inline-block h-3.5 w-3.5 rounded-full bg-white shadow transition-transform " +
                    (localValue === "true" ? "translate-x-4.5" : "translate-x-0.5")
                  }
                />
              </button>
              <span className="text-sm text-slate-400">
                {localValue === "true" ? t("on") : t("off")}
              </span>
            </div>
          ) : setting.type === "password" ? (
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={localValue}
                onChange={(e) => handleChange(e.target.value)}
                placeholder={t("passwordPlaceholder")}
                className={inputCls + " pr-10"}
              />
              <button
                type="button"
                onClick={() => setShowPassword((v) => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                tabIndex={-1}
              >
                {showPassword ? (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                )}
              </button>
            </div>
          ) : setting.type === "integer" ? (
            <input
              type="number"
              value={localValue}
              onChange={(e) => handleChange(e.target.value)}
              className={inputCls}
            />
          ) : (
            <input
              type="text"
              value={localValue}
              onChange={(e) => handleChange(e.target.value)}
              className={inputCls}
            />
          )}

          <p className="text-[11px] text-slate-600 mt-1">{setting.description}</p>
        </div>
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="font-mono text-[10px] uppercase tracking-widest text-slate-500 mb-3 pb-2 border-b border-white/6">
      {children}
    </h2>
  );
}

export default function SettingsPage() {
  const t = useTranslations("Settings");
  const router = useRouter();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [settings, setSettings] = useState<AppSetting[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const userJson = localStorage.getItem("meetbot_user");
    if (!userJson) {
      router.replace("/login");
      return;
    }
    api
      .getMe()
      .then((p) => {
        setProfile(p);
        if (p.is_admin) {
          return api.getSettings().then(setSettings);
        }
      })
      .catch((err) => setError(err.message || "Failed to load profile"))
      .finally(() => setLoading(false));
  }, [router]);

  const handleSave = useCallback(async (key: string, value: string) => {
    await api.updateSetting(key, value);
    const updated = await api.getSettings();
    setSettings(updated);
  }, []);

  const groupedSettings = GROUP_ORDER.reduce<Record<string, AppSetting[]>>(
    (acc, group) => {
      acc[group] = settings.filter((s) => s.group === group);
      return acc;
    },
    {}
  );

  const formatDate = (iso: string | null) => {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleString();
    } catch {
      return iso;
    }
  };

  return (
    <div className="mx-auto max-w-4xl">
      <BackButton />
      <h1 className="font-display text-3xl font-bold text-white mb-8">
        {t("title")}
      </h1>

        {loading && (
          <p className="text-slate-500 text-sm animate-pulse">{t("loading")}</p>
        )}

        {error && (
          <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-4 mb-6 text-red-400 text-sm">
            {error}
          </div>
        )}

        {profile && (
          <section className="rounded-xl border border-white/6 bg-[#0d1120] p-6 mb-8">
            <SectionLabel>{t("profile")}</SectionLabel>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500 block mb-0.5">
                  {t("username")}
                </span>
                <span className="text-slate-200">{profile.username}</span>
              </div>
              <div>
                <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500 block mb-0.5">
                  {t("displayName")}
                </span>
                <span className="text-slate-200">{profile.display_name}</span>
              </div>
              <div>
                <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500 block mb-0.5">
                  {t("role")}
                </span>
                {profile.is_admin ? (
                  <span className="text-[9px] bg-indigo-500/10 text-indigo-400 ring-1 ring-indigo-500/20 rounded px-2 py-0.5">
                    {t("admin")}
                  </span>
                ) : (
                  <span className="text-slate-400">{t("user")}</span>
                )}
              </div>
              <div>
                <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500 block mb-0.5">
                  {t("memberSince")}
                </span>
                <span className="text-slate-400">{formatDate(profile.created_at)}</span>
              </div>
              <div>
                <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500 block mb-0.5">
                  {t("lastLogin")}
                </span>
                <span className="text-slate-400">{formatDate(profile.last_login)}</span>
              </div>
            </div>
          </section>
        )}

        {profile && !profile.is_admin && !loading && (
          <p className="text-slate-600 text-sm">
            {t("adminOnly")}
          </p>
        )}

        {profile?.is_admin && settings.length > 0 &&
          GROUP_ORDER.map((group) => {
            const items = groupedSettings[group];
            if (!items || items.length === 0) return null;
            return (
              <section
                key={group}
                className="rounded-xl border border-white/6 bg-[#0d1120] p-6 mb-6"
              >
                <SectionLabel>{group}</SectionLabel>
                {items.map((s) => (
                  <SettingRow key={s.key} setting={s} onSave={handleSave} />
                ))}
              </section>
            );
          })}
    </div>
  );
}
