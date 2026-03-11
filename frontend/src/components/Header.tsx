"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, usePathname } from "next/navigation";
import { Mic, LogOut, User as UserIcon } from "lucide-react";

export function Header() {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<{ username: string; display_name: string } | null>(null);

  useEffect(() => {
    const savedUser = localStorage.getItem("meetbot_user");
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (e) {
        localStorage.removeItem("meetbot_user");
      }
    }
  }, [pathname]); // Refresh on navigation

  const handleSignOut = () => {
    localStorage.removeItem("meetbot_user");
    setUser(null);
    router.push("/login");
  };

  if (pathname === "/login") return null;

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/10 bg-slate-950/60 p-4 backdrop-blur supports-[backdrop-filter]:bg-slate-950/40">
      <div className="container mx-auto flex h-14 items-center gap-4">
        <Link href="/" className="flex items-center gap-2 font-bold tracking-tight text-white hover:text-teal-400 transition-colors">
          <span className="text-xl text-teal-500"><Mic className="h-6 w-6" /></span>
          <span className="hidden sm:inline">MeetBot</span>
        </Link>
        <nav className="flex flex-1 items-center space-x-6 text-sm font-medium ml-6">
          <Link href="/" className={`${pathname === "/" ? "text-teal-400" : "text-slate-300 hover:text-white transition-colors"}`}>Dashboard</Link>
        </nav>
        
        <div className="flex items-center gap-4">
          {user ? (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 rounded-full bg-slate-800/80 px-3 py-1.5 border border-white/5">
                <UserIcon className="h-3.5 w-3.5 text-teal-400" />
                <span className="text-xs font-medium text-slate-200">{user.display_name || user.username}</span>
              </div>
              <button
                onClick={handleSignOut}
                className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs font-medium text-slate-300 backdrop-blur transition-colors hover:border-red-500/50 hover:text-red-400"
              >
                <LogOut className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Sign Out</span>
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="flex items-center gap-2 rounded-md border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs font-medium text-slate-300 backdrop-blur transition-colors hover:border-teal-500/50 hover:text-white"
            >
              Sign In
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
