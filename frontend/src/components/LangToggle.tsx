"use client";

import { useEffect, useState } from "react";
import { useTranslations } from "next-intl";

export function LangToggle() {
  const t = useTranslations("LangToggle");
  const [locale, setLocale] = useState<"en" | "ja">("en");

  useEffect(() => {
    const stored = localStorage.getItem("meetbot_locale");
    if (stored === "ja" || stored === "en") setLocale(stored);
  }, []);

  const toggle = () => {
    const next = locale === "en" ? "ja" : "en";
    // Set cookie so the server can read it on next render
    document.cookie = `meetbot_locale=${next}; path=/; max-age=31536000; SameSite=Lax`;
    localStorage.setItem("meetbot_locale", next);
    window.location.reload();
  };

  return (
    <button
      onClick={toggle}
      className="flex h-8 items-center justify-center rounded-md px-2 text-xs font-medium text-slate-500 transition-colors hover:bg-white/5 hover:text-slate-300"
      title={locale === "en" ? "Switch to Japanese" : "Switch to English"}
    >
      {locale === "en" ? t("toJa") : t("toEn")}
    </button>
  );
}
