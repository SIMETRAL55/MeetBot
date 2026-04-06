"use client";

import { useState, useRef } from "react";
import { 
  FileUp, 
  FileText, 
  X, 
  ChevronRight, 
  ChevronLeft, 
  CheckCircle2, 
  FileCode,
  AlertCircle
} from "lucide-react";
import { useTranslations } from "next-intl";
import * as mammoth from "mammoth";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";

// Set up pdfjs worker
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.mjs`;

interface ContextPanelProps {
  onContextChange: (data: { fileText: string; fileName: string; pastedText: string; enabled: boolean }) => void;
}

export function ContextPanel({ onContextChange }: ContextPanelProps) {
  const t = useTranslations("ChatPage");
  const [isOpen, setIsOpen] = useState(true);
  const [fileText, setFileText] = useState("");
  const [fileName, setFileName] = useState("");
  const [pastedText, setPastedText] = useState("");
  const [enabled, setEnabled] = useState(false);
  const [parsing, setParsing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const notifyChange = (updates: Partial<{ fileText: string; fileName: string; pastedText: string; enabled: boolean }>) => {
    onContextChange({
      fileText: updates.fileText ?? fileText,
      fileName: updates.fileName ?? fileName,
      pastedText: updates.pastedText ?? pastedText,
      enabled: updates.enabled ?? enabled
    });
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setParsing(true);
    setError(null);
    setFileName(file.name);

    try {
      let text = "";
      if (file.name.endsWith(".pdf")) {
        const arrayBuffer = await file.arrayBuffer();
        const pdf = await pdfjs.getDocument({ data: arrayBuffer }).promise;
        let fullText = "";
        for (let i = 1; i <= pdf.numPages; i++) {
          const page = await pdf.getPage(i);
          const content = await page.getTextContent();
          const pageText = content.items
            .map((item) => ("str" in item ? item.str : ""))
            .join(" ");
          fullText += pageText + "\n";
        }
        text = fullText;
      } else if (file.name.endsWith(".docx")) {
        const arrayBuffer = await file.arrayBuffer();
        const result = await mammoth.extractRawText({ arrayBuffer });
        text = result.value;
      } else if (file.name.endsWith(".txt")) {
        text = await file.text();
      } else {
        throw new Error("Unsupported file type. Please use PDF, .docx, or .txt");
      }

      setFileText(text);
      notifyChange({ fileText: text, fileName: file.name });
    } catch (err) {
      console.error("File parsing error:", err);
      const errorMessage = err instanceof Error ? err.message : "Failed to parse file";
      setError(errorMessage);
      setFileName("");
      setFileText("");
    } finally {
      setParsing(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleRemoveFile = () => {
    setFileText("");
    setFileName("");
    notifyChange({ fileText: "", fileName: "" });
  };

  const handlePastedTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const text = e.target.value;
    setPastedText(text);
    notifyChange({ pastedText: text });
  };

  const toggleEnabled = () => {
    const newState = !enabled;
    setEnabled(newState);
    notifyChange({ enabled: newState });
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed right-0 top-1/2 -translate-y-1/2 z-10 hidden lg:flex h-12 w-6 items-center justify-center rounded-l-md border border-white/5 bg-[#080b14] text-slate-500 hover:text-slate-300 transition-colors"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>
    );
  }

  return (
    <aside className="hidden w-80 shrink-0 flex-col border-l border-white/5 bg-[#080b14] lg:flex">
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-3">
        <div className="flex items-center gap-2">
          <FileCode className="h-4 w-4 text-indigo-400" />
          <h2 className="font-display text-sm font-semibold text-slate-200">{t("contextPanel")}</h2>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="text-slate-500 hover:text-slate-300 transition-colors"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Toggle */}
        <div className="flex items-center justify-between p-3 rounded-lg border border-white/5 bg-white/2">
          <span className="text-xs font-medium text-slate-300">{t("includeInPrompt")}</span>
          <button
            onClick={toggleEnabled}
            className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none ${
              enabled ? "bg-indigo-600" : "bg-slate-700"
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                enabled ? "translate-x-4" : "translate-x-0"
              }`}
            />
          </button>
        </div>

        {/* File Upload */}
        <div className="space-y-2">
          <label className="font-mono-editorial text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-700">
            {t("uploadFile")}
          </label>
          
          {fileName ? (
            <div className="flex items-center gap-3 p-3 rounded-lg border border-indigo-500/20 bg-indigo-500/5 ring-1 ring-indigo-500/10">
              <FileText className="h-8 w-8 text-indigo-400/60 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium text-slate-200 truncate">{fileName}</p>
                <p className="text-[10px] text-slate-500">{fileText.length} {t("chars")}</p>
              </div>
              <button
                onClick={handleRemoveFile}
                className="p-1 rounded-full hover:bg-white/5 text-slate-500 hover:text-slate-300 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ) : (
            <div
              onClick={() => fileInputRef.current?.click()}
              className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
                parsing ? "border-indigo-500/40 bg-indigo-500/5" : "border-white/5 bg-white/2 hover:border-white/10 hover:bg-white/3"
              }`}
            >
              {parsing ? (
                <div className="flex flex-col items-center gap-2">
                  <div className="h-5 w-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                  <span className="text-[10px] text-indigo-400 font-medium tracking-wide">PARSING...</span>
                </div>
              ) : (
                <>
                  <FileUp className="h-6 w-6 text-slate-600 mb-2" />
                  <span className="text-xs text-slate-500 text-center">PDF, DOCX, TXT</span>
                </>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={handleFileChange}
                className="hidden"
              />
            </div>
          )}
          {error && (
            <div className="flex items-center gap-1.5 px-1 py-1 text-[10px] text-red-400">
              <AlertCircle className="h-3 w-3" />
              {error}
            </div>
          )}
        </div>

        {/* Paste Text */}
        <div className="space-y-2">
          <label className="font-mono-editorial text-[9px] font-semibold uppercase tracking-[0.15em] text-slate-700">
            {t("pasteText")}
          </label>
          <textarea
            value={pastedText}
            onChange={handlePastedTextChange}
            placeholder="Type or paste context here..."
            className="w-full h-40 bg-white/2 border border-white/5 rounded-lg p-3 text-xs text-slate-300 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500/40 focus:ring-1 focus:ring-indigo-500/20 resize-none transition-all"
          />
          {pastedText && (
            <div className="flex justify-end">
              <span className="text-[10px] text-slate-700">{pastedText.length} {t("chars")}</span>
            </div>
          )}
        </div>

        {/* Status Indicator */}
        {enabled && (fileText || pastedText) && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-md bg-green-500/5 border border-green-500/20">
            <CheckCircle2 className="h-3 w-3 text-green-500" />
            <span className="text-[10px] font-medium text-green-500/80 uppercase tracking-wider">Active in next prompt</span>
          </div>
        )}
      </div>
    </aside>
  );
}
