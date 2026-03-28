import type { Metadata } from "next";
import { IBM_Plex_Mono, Fraunces } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/Header";

const ibmPlexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

const fraunces = Fraunces({
  subsets: ["latin"],
  weight: "variable",
  variable: "--font-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MeetBot",
  description: "Advanced Audio Transcription & RAG Q&A",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`dark ${ibmPlexMono.variable} ${fraunces.variable}`}
      suppressHydrationWarning
    >
      <body
        className="min-h-screen bg-[#07090f] text-[#e8ecf4] antialiased"
        suppressHydrationWarning
      >
        {/* Editorial background — subtle deep gradient */}
        <div className="pointer-events-none fixed inset-0 -z-10">
          <div className="absolute top-0 right-0 h-[600px] w-[600px] rounded-full bg-amber-500/3 blur-[180px]" />
          <div className="absolute bottom-0 left-0 h-[500px] w-[500px] rounded-full bg-slate-700/10 blur-[150px]" />
          <div className="absolute top-[40%] left-[30%] h-[300px] w-[300px] rounded-full bg-indigo-900/8 blur-[120px]" />
        </div>

        <div className="relative flex min-h-screen flex-col">
          <Header />
          <main className="flex-1">
            <div className="container mx-auto px-4 py-8 sm:px-6">
              {children}
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
