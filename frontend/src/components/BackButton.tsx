"use client";

import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";

export function BackButton({ href }: { href?: string }) {
  const t = useTranslations("BackButton");
  const router = useRouter();
  return (
    <button
      onClick={() => (href ? router.push(href) : router.back())}
      className="mb-4 flex items-center gap-1.5 text-sm text-slate-500 transition-colors hover:text-slate-300"
    >
      <ArrowLeft className="h-4 w-4" />
      {t("label")}
    </button>
  );
}
