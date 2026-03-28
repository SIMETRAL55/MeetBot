"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { LogOut, User as UserIcon, ChevronRight, Settings } from "lucide-react";

/* ── Breadcrumb helper ──────────────────────────────────────────────────────
   Parses pathname into human-readable crumbs.
   /             → [Dashboard]
   /job/[id]     → [Dashboard, Job]
   /chat/[id]    → [Dashboard, Chat]
   /settings     → [Dashboard, Settings]
   ──────────────────────────────────────────────────────────────────────── */
function useBreadcrumbs() {
  const pathname = usePathname();
  const segments = pathname.split("/").filter(Boolean);

  const crumbs: { label: string; href: string }[] = [
    { label: "Dashboard", href: "/" },
  ];

  if (segments[0] === "job" && segments[1]) {
    crumbs.push({ label: "Job", href: `/job/${segments[1]}` });
  } else if (segments[0] === "chat" && segments[1]) {
    crumbs.push({ label: "Chat", href: `/chat/${segments[1]}` });
  } else if (segments[0] === "settings") {
    crumbs.push({ label: "Settings", href: "/settings" });
  }

  return crumbs;
}

export function Header() {
  const router = useRouter();
  const pathname = usePathname();
  const crumbs = useBreadcrumbs();
  const [user, setUser] = useState<{ username: string; display_name: string } | null>(null);

  useEffect(() => {
    const savedUser = localStorage.getItem("meetbot_user");
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch {
        localStorage.removeItem("meetbot_user");
      }
    }
  }, [pathname]);

  const handleSignOut = () => {
    localStorage.removeItem("meetbot_user");
    setUser(null);
    router.push("/login");
  };

  if (pathname === "/login") return null;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/5 bg-[#07090f]/80 backdrop-blur-xl">
      <div className="container mx-auto flex h-14 items-center gap-4 px-4 sm:px-6">

        {/* Wordmark */}
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2 transition-opacity hover:opacity-80"
        >
          <span
            className="font-mono-editorial text-sm font-semibold tracking-[0.15em] text-amber-400 uppercase"
          >
            MEETBOT
          </span>
          <span
            className="hidden h-4 w-px bg-white/10 sm:block"
            aria-hidden
          />
        </Link>

        {/* Breadcrumb */}
        <nav
          aria-label="breadcrumb"
          className="flex flex-1 items-center gap-1 overflow-hidden"
        >
          {crumbs.map((crumb, i) => {
            const isLast = i === crumbs.length - 1;
            return (
              <span key={crumb.href} className="flex items-center gap-1 min-w-0">
                {i > 0 && (
                  <ChevronRight
                    className="h-3.5 w-3.5 shrink-0 text-slate-700"
                    aria-hidden
                  />
                )}
                {isLast ? (
                  <span
                    className="font-mono-editorial truncate text-xs font-medium text-slate-300"
                  >
                    {crumb.label}
                  </span>
                ) : (
                  <Link
                    href={crumb.href}
                    className="font-mono-editorial truncate text-xs font-medium text-slate-600 transition-colors hover:text-slate-400"
                  >
                    {crumb.label}
                  </Link>
                )}
              </span>
            );
          })}
        </nav>

        {/* Right: Settings + User */}
        <div className="flex shrink-0 items-center gap-2">
          <Link
            href="/settings"
            title="Settings"
            className="flex h-8 w-8 items-center justify-center rounded-md text-slate-600 transition-colors hover:bg-white/5 hover:text-slate-300"
          >
            <Settings className="h-4 w-4" />
          </Link>

          {user ? (
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1.5 rounded-md border border-white/5 bg-white/3 px-2.5 py-1.5">
                <UserIcon className="h-3 w-3 text-amber-500/70" />
                <span
                  className="font-mono-editorial text-xs text-slate-400"
                >
                  {user.display_name || user.username}
                </span>
              </div>
              <button
                onClick={handleSignOut}
                className="flex h-8 w-8 items-center justify-center rounded-md text-slate-600 transition-colors hover:bg-white/5 hover:text-red-400"
                title="Sign out"
              >
                <LogOut className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="rounded-md border border-amber-500/30 bg-amber-500/8 px-3 py-1.5 text-xs font-medium text-amber-400 transition-colors hover:bg-amber-500/15"
            >
              Sign In
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
