"use client";

import { useState, useRef, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Send, User, Bot, Loader2, ArrowLeft, Settings2, FileText, ChevronDown, ChevronRight, SlidersHorizontal } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { useChatWS } from "@/lib/hooks/useChatWS";
import { Card } from "@/components/ui/card";
import { ChatSource, Job } from "@/types";
import { api } from "@/lib/api";

function SourceAccordion({ sources }: { sources: ChatSource[] }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 rounded-md border border-slate-700 bg-slate-800/50">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-slate-300 hover:text-white"
      >
        <span className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5 text-teal-400" />
          Retrieved Context ({sources.length} items)
        </span>
        {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </button>
      
      {isOpen && (
        <div className="border-t border-slate-700 px-3 py-2">
          {sources.map((src, i) => (
            <div key={i} className="mb-3 last:mb-0">
              <div className="mb-1 flex items-center gap-2 text-[10px] font-semibold tracking-wider text-slate-400">
                <span className="rounded bg-slate-900 px-1 py-0.5" title="Distance (lower is better)">Dist: {src.distance?.toFixed(3) ?? "N/A"}</span>
                <span className="text-violet-400">{src.speaker}</span>
              </div>
              <p className="border-l-2 border-teal-500/50 pl-2 text-xs leading-relaxed text-slate-300">
                &quot;{src.text}&quot;
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

type RetrievalLevel = "chunk" | "segment" | "document";

export default function ChatPage() {
  const { id } = useParams() as { id: string };
  const router = useRouter();
  const [input, setInput] = useState("");
  const [llmMode, setLlmMode] = useState<"local" | "hf">("local");
  const [retrievalLevel, setRetrievalLevel] = useState<RetrievalLevel>("chunk");
  const [segmentCount, setSegmentCount] = useState(5);
  const [showSettings, setShowSettings] = useState(false);
  const [jobName, setJobName] = useState<string>("");
  
  const { messages, loading, fetchingHistory, error, sendMessage, loadHistory, abortQuery } = useChatWS(id);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Close any in-flight WebSocket when the user navigates away from this page.
  // Without this the streaming thread on the backend keeps running (and the LLM
  // keeps generating tokens) for 1-2 minutes until the natural end of the response.
  useEffect(() => {
    return () => {
      abortQuery();
    };
  }, [abortQuery]);

  useEffect(() => {
    const user = localStorage.getItem("meetbot_user");
    if (!user) {
      router.push("/login");
      return;
    }

    // Guard: useParams() may return undefined/null during Next.js hydration.
    // Without this check, API calls fire as /api/jobs/undefined/...
    if (!id) return;

    // Load job metadata to display the original filename in the header
    api.getJobStatus(id).then((job: Job) => {
      setJobName(job.original_filename);
    }).catch((err: any) => {
      console.error("Failed to load job status:", err);
    });

    // Load chat history — also ensures ChatSession exists on the backend
    // (GET /api/jobs/{id}/chat/history calls get_or_create_chat_session)
    // before any WebSocket connection is opened, preventing FOREIGN KEY errors.
    loadHistory();
  }, [id, router, loadHistory]);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;
    sendMessage(input, llmMode, retrievalLevel, retrievalLevel === "segment" ? segmentCount : undefined);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="mx-auto flex max-w-5xl flex-col h-[calc(100vh-8rem)]">
      {/* Header — needs a stacking context (relative + z-10) so the absolute
          retrieval-method dropdown overlays the Card below, not beneath it.
          backdrop-blur on the Card creates its own stacking context; without
          z-index here the dropdown would render behind it regardless of z-40. */}
      <div className="relative z-10 mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
         <div className="flex-1">
           <button onClick={() => router.push(`/job/${id}`)} className="mb-2 flex items-center text-sm font-medium text-slate-400 hover:text-white transition-colors">
              <ArrowLeft className="mr-1 h-4 w-4" /> Back to Job Details
           </button>
           <h1 className="text-2xl font-bold tracking-tight text-white drop-shadow-sm">MeetBot RAG Assistant</h1>
           <p className="mt-1 flex items-center text-sm text-slate-400">
             {/* FIX: Always show filename, never show raw hash ID to user */}
             {jobName ? `Conversation: ${jobName}` : "Loading conversation..."}
             <span className="ml-3 inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium bg-teal-500/10 text-teal-400">
               <span className="h-1.5 w-1.5 rounded-full bg-teal-500" />
               {loading ? "Streaming…" : fetchingHistory ? "Loading history…" : "Ready"}
             </span>
           </p>
         </div>

         {/* Settings Panel */}
         <div className="flex flex-col gap-3">
           {/* LLM selector */}
           <div className="flex items-center gap-3 rounded-lg border border-slate-700 bg-slate-900/50 p-2 text-sm shadow-sm backdrop-blur">
              <Settings2 className="h-4 w-4 text-slate-400" />
              <span className="font-medium text-slate-300">Model:</span>
              <div className="flex rounded bg-slate-950 p-1">
                <button 
                  onClick={() => setLlmMode("local")}
                  className={`flex-1 rounded-sm px-3 py-1 text-xs font-semibold transition-colors ${llmMode === "local" ? "bg-slate-800 text-teal-400 shadow" : "text-slate-500 hover:text-slate-300"}`}
                >
                  Local GGUF
                </button>
                <button 
                  onClick={() => setLlmMode("hf")}
                  className={`flex-1 rounded-sm px-3 py-1 text-xs font-semibold transition-colors ${llmMode === "hf" ? "bg-slate-800 text-violet-400 shadow" : "text-slate-500 hover:text-slate-300"}`}
                >
                  HuggingFace
                </button>
              </div>
           </div>

           {/* Retrieval level selector — expanded panel uses absolute positioning so
               it overlays content below instead of pushing the chat card down */}
           <div className="relative rounded-lg border border-slate-700 bg-slate-900/50 shadow-sm backdrop-blur">
             <button
               onClick={() => setShowSettings(!showSettings)}
               className="flex w-full items-center gap-2 p-2 text-sm"
             >
               <SlidersHorizontal className="h-4 w-4 text-slate-400" />
               <span className="font-medium text-slate-300">Retrieval:</span>
               <span className="ml-auto text-xs font-semibold text-teal-400 capitalize">{retrievalLevel}</span>
               {showSettings ? <ChevronDown className="h-3 w-3 text-slate-400" /> : <ChevronRight className="h-3 w-3 text-slate-400" />}
             </button>

             {showSettings && (
               <div className="absolute right-0 top-full z-40 mt-1 w-64 rounded-lg border border-slate-700 bg-slate-900 p-3 shadow-xl backdrop-blur space-y-3">
                 <div className="flex rounded bg-slate-950 p-1 gap-0.5">
                   {(["chunk", "segment", "document"] as RetrievalLevel[]).map((level) => (
                     <button
                       key={level}
                       onClick={() => setRetrievalLevel(level)}
                       className={`flex-1 rounded-sm px-3 py-1.5 text-xs font-semibold transition-colors capitalize ${
                         retrievalLevel === level
                           ? "bg-slate-800 text-teal-400 shadow"
                           : "text-slate-500 hover:text-slate-300"
                       }`}
                     >
                       {level}
                     </button>
                   ))}
                 </div>
                 <p className="text-[11px] text-slate-500 leading-relaxed">
                   {retrievalLevel === "chunk" && "Retrieves small text chunks for precise, focused answers."}
                   {retrievalLevel === "segment" && "Retrieves full transcript segments (speaker turns) for broader context."}
                   {retrievalLevel === "document" && "Retrieves full document context — useful for summaries."}
                 </p>
                 {retrievalLevel === "segment" && (
                   <div className="flex items-center gap-3">
                     <label className="text-xs font-medium text-slate-400 whitespace-nowrap">Segment Count:</label>
                     <input
                       type="number"
                       min={1}
                       max={50}
                       value={segmentCount}
                       onChange={(e) => setSegmentCount(Math.max(1, Math.min(50, parseInt(e.target.value) || 5)))}
                       className="w-16 rounded border border-slate-700 bg-slate-950 px-2 py-1 text-center text-xs text-slate-200 outline-none focus:border-teal-500"
                     />
                   </div>
                 )}
               </div>
             )}
           </div>
         </div>
      </div>

      <Card className="flex flex-1 flex-col overflow-hidden border-white/10 bg-slate-900/50 shadow-xl backdrop-blur">
        
        {/* Messages Layout */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center text-slate-400">
               <Bot className="mb-4 h-12 w-12 text-slate-600" />
               <p className="text-lg font-medium text-slate-300">How can I help you today?</p>
               <p className="mt-2 max-w-md text-sm">Ask me to summarize the meeting, list action items, or extract key decisions.</p>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} className={`flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                
                {msg.role === "assistant" && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-teal-500/20 text-teal-400 ring-1 ring-teal-500/50">
                    <Bot className="h-5 w-5" />
                  </div>
                )}
                
                <div className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-5 py-3.5 ${
                  msg.role === "user" 
                    ? "bg-violet-600 text-white shadow-md shadow-violet-900/20 rounded-tr-none" 
                    : "bg-slate-800 text-slate-200 shadow-md shadow-black/20 rounded-tl-none border border-slate-700"
                }`}>
                  {msg.role === "assistant" ? (
                    <div className="prose prose-sm prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-slate-900 prose-pre:border prose-pre:border-slate-800">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content || (loading && i === messages.length - 1 ? "..." : "")}
                      </ReactMarkdown>
                      {msg.sources && <SourceAccordion sources={msg.sources} />}
                    </div>
                  ) : (
                    <div className="whitespace-pre-wrap text-[15px] leading-relaxed">
                      {msg.content}
                    </div>
                  )}
                </div>

                {msg.role === "user" && (
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-700 text-slate-300 ring-1 ring-white/10">
                    <User className="h-5 w-5" />
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Input Bar */}
        <div className="border-t border-slate-800 bg-slate-950/50 p-4">
          {error && (
            <div className="mb-3 rounded text-sm text-red-400">
              Error: {error}
            </div>
          )}
          
          <form onSubmit={handleSubmit} className="relative mx-auto flex max-w-4xl items-end gap-2">
            <div className="relative flex-1">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={loading ? "Waiting for response..." : "Ask a question about your transcript... (Shift+Enter for new line)"}
                className="block max-h-[150px] w-full resize-none rounded-xl border border-slate-700 bg-slate-900 py-3.5 pl-4 pr-12 text-[15px] text-slate-200 placeholder:text-slate-500 focus:border-teal-500 focus:outline-none focus:ring-1 focus:ring-teal-500/50 disabled:opacity-50 min-h-[52px]"
                rows={1}
                disabled={loading}
              />
            </div>
            
            <button
              type="submit"
              disabled={!input.trim() || loading}
              className={`absolute bottom-2 right-2 flex h-9 w-9 items-center justify-center rounded-lg transition-all ${
                input.trim() && !loading
                  ? "bg-teal-500 text-teal-950 hover:bg-teal-400 hover:shadow-[0_0_15px_rgba(20,184,166,0.4)]" 
                  : "bg-slate-800 text-slate-500 disable opacity-50"
              }`}
            >
              <Send className="h-4 w-4 ml-0.5" />
            </button>

            {loading && (
              <button
                type="button"
                onClick={abortQuery}
                className="absolute -top-12 right-0 flex items-center gap-2 rounded-full bg-red-500/10 px-4 py-1.5 text-xs font-semibold text-red-400 ring-1 ring-red-500/20 hover:bg-red-500/20 transition-all"
              >
                <Loader2 className="h-3 w-3 animate-spin" />
                Stop Generation
              </button>
            )}
          </form>
          <div className="mt-2 text-center text-xs text-slate-500">
            MeetBot can make mistakes. Consider verifying important information.
          </div>
        </div>
      </Card>
      
    </div>
  );
}
