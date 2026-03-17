"use client";

import { useState, useRef, useCallback } from "react";
import { ChatMessage } from "@/types";
import { api, getWsBaseUrl } from "@/lib/api";

/**
 * Chat hook that opens a NEW WebSocket connection per question.
 *
 * The backend's /ws/chat/{job_id} endpoint is one-shot:
 *   1. Client connects.
 *   2. Client sends one JSON message:
 *      { question, llm_mode, retrieval_level, segment_count, retrieval_method }.
 *   3. Server streams back: sources → token* → done.
 *   4. Server closes the connection.
 *
 * This hook therefore does NOT keep a persistent connection. Instead, each
 * call to `sendMessage` opens a fresh WebSocket, streams the response into
 * state, and then cleans up.
 */
export function useChatWS(jobId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [fetchingHistory, setFetchingHistory] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(true); // always "ready" since we connect on demand
  const wsRef = useRef<WebSocket | null>(null);

  const loadHistory = useCallback(async () => {
    if (!jobId) return;
    setFetchingHistory(true);
    setError(null);
    try {
      const history = await api.getChatHistory(jobId);
      setMessages(history);
    } catch (err: any) {
      console.error("Failed to load chat history:", err);
      setError(err.message || "Failed to load chat history");
    } finally {
      setFetchingHistory(false);
    }
  }, [jobId]);

  const abortQuery = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      setLoading(false);
    }
  }, []);

  const sendMessage = useCallback(
    (
      question: string,
      llmMode: "local" | "hf" = "local",
      retrievalLevel: "chunk" | "segment" | "document" = "chunk",
      segmentCount?: number,
      retrievalMethod: "vector" | "pageindex" = "vector",
    ) => {
      if (!question.trim() || !jobId) return;

      setMessages((prev) => [...prev, { role: "user", content: question }]);
      setError(null);
      setLoading(true);

      // Add an empty assistant message we'll fill with streaming tokens
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      const wsUrl = `${getWsBaseUrl()}/ws/chat/${jobId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        const payload: Record<string, unknown> = {
          question,
          llm_mode: llmMode,
          retrieval_level: retrievalLevel,
          retrieval_method: retrievalMethod,
        };
        if (retrievalLevel === "segment" && segmentCount != null) {
          payload.segment_count = segmentCount;
        }
        ws.send(JSON.stringify(payload));
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);

          switch (payload.type) {
            case "sources":
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last?.role === "assistant") {
                  copy[copy.length - 1] = { ...last, sources: payload.data };
                }
                return copy;
              });
              break;

            case "token":
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last?.role === "assistant") {
                  copy[copy.length - 1] = {
                    ...last,
                    content: last.content + (payload.data ?? ""),
                  };
                }
                return copy;
              });
              break;

            case "done":
              // Replace initial sources with answer-similarity-filtered final list
              if (payload.filtered_sources && Array.isArray(payload.filtered_sources)) {
                setMessages((prev) => {
                  const copy = [...prev];
                  const last = copy[copy.length - 1];
                  if (last?.role === "assistant") {
                    copy[copy.length - 1] = {
                      ...last,
                      sources: payload.filtered_sources,
                    };
                  }
                  return copy;
                });
              }
              setLoading(false);
              break;

            case "error":
              setError(payload.data || "An error occurred");
              setLoading(false);
              break;
          }
        } catch (err) {
          console.error("Failed to parse chat msg:", err);
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection error");
        setLoading(false);
      };

      ws.onclose = () => {
        wsRef.current = null;
        setLoading(false);
      };
    },
    [jobId]
  );

  return {
    messages,
    loading,
    fetchingHistory,
    error,
    connected,
    sendMessage,
    loadHistory,
    abortQuery,
  };
}
