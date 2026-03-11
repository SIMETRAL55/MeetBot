import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/Header";

const inter = Inter({ subsets: ["latin"] });

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
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${inter.className} min-h-screen bg-slate-950 text-slate-50 antialiased selection:bg-teal-500/30 selection:text-teal-200`}
        suppressHydrationWarning
      >
        <div className="relative flex min-h-screen flex-col">
          {/* Futuristic background gradients */}
          <div className="pointer-events-none fixed inset-0 flex justify-center opacity-30">
            <div className="absolute top-[-20%] right-[-10%] h-[500px] w-[500px] rounded-full bg-violet-600/20 blur-[120px]" />
            <div className="absolute bottom-[-20%] left-[-10%] h-[500px] w-[500px] rounded-full bg-teal-500/20 blur-[120px]" />
          </div>

          <Header />

          <main className="flex-1">
            <div className="container mx-auto py-8">
              {children}
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
