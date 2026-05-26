'use client';

import React from 'react';
import { FolderGit, Plus, MessageSquare, Trash2, ShieldCheck, X, Key, Eye, EyeOff } from 'lucide-react';

export interface Session {
  id: string;
  repoUrl: string;
  repoName: string;
  repoDescription: string;
  summary: string;
  filesFetched: string[];
  repoLanguage: string;
  repoStars: number;
  architecture: string;
  chatHistory?: { role: 'user' | 'assistant'; content: string }[];
}

interface SessionSidebarProps {
  sessions: Session[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSessionClick: () => void;
  onDeleteSession: (id: string) => void;
  onCloseClick?: () => void;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
}

export default function SessionSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSessionClick,
  onDeleteSession,
  onCloseClick,
  apiKey,
  onApiKeyChange,
}: SessionSidebarProps) {
  const [showKey, setShowKey] = React.useState(false);
  return (
    <aside className="w-full h-full flex flex-col bg-zinc-950/80 border-r border-white/5 backdrop-blur-xl">
      {/* Brand Header */}
      <div className="p-6 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-brand-indigo to-brand-violet flex items-center justify-center shadow-lg shadow-brand-indigo/25">
            <FolderGit className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-zinc-100 text-sm tracking-wide">OnboardFlow</h1>
            <span className="text-[10px] text-zinc-500 font-mono flex items-center gap-1">
              <ShieldCheck className="w-3 h-3 text-emerald-500" /> LangGraph Agent
            </span>
          </div>
        </div>

        {onCloseClick && (
          <button
            onClick={onCloseClick}
            className="md:hidden p-1.5 hover:bg-white/5 text-zinc-500 hover:text-zinc-200 rounded-lg transition-all duration-200"
            aria-label="Close Sidebar"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Action Button */}
      <div className="p-4">
        <button
          onClick={onNewSessionClick}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-gradient-to-r from-brand-indigo to-brand-violet hover:from-brand-indigo/90 hover:to-brand-violet/90 text-white font-medium text-xs shadow-lg shadow-brand-indigo/20 hover:shadow-brand-indigo/35 transition-all duration-300 transform active:scale-95"
        >
          <Plus className="w-4 h-4" />
          Analyze New Repo
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
        <div className="px-3 py-2 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
          Active Sessions
        </div>

        {sessions.length === 0 ? (
          <div className="p-6 text-center text-xs text-zinc-600 font-medium italic border border-dashed border-white/5 rounded-xl mx-2">
            No active sessions. Please ingest a repository to get started.
          </div>
        ) : (
          sessions.map((session) => {
            const isActive = session.id === activeSessionId;
            return (
              <div
                key={session.id}
                className={`group relative flex items-center justify-between w-full px-3 py-3 rounded-xl transition-all duration-300 cursor-pointer ${isActive
                    ? 'bg-white/5 text-zinc-100 shadow-inner border-l-2 border-brand-indigo'
                    : 'text-zinc-400 hover:bg-white/[0.02] hover:text-zinc-200'
                  }`}
                onClick={() => onSelectSession(session.id)}
              >
                <div className="flex items-center gap-3 overflow-hidden pr-6">
                  <MessageSquare className={`w-4.5 h-4.5 shrink-0 ${isActive ? 'text-brand-indigo' : 'text-zinc-500'}`} />
                  <div className="text-left overflow-hidden">
                    <div className="text-xs font-semibold truncate leading-tight">
                      {session.id}
                    </div>
                    <div className="text-[10px] text-zinc-500 truncate font-mono mt-0.5">
                      {session.repoName || 'Parsing...'}
                    </div>
                  </div>
                </div>

                {/* Delete button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                  className="absolute right-3 opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 hover:text-rose-400 rounded-md text-zinc-500 transition-all duration-200"
                  title="Delete Session"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* API Key Settings in Footer */}
      <div className="p-4 border-t border-white/5 bg-zinc-950/40 flex flex-col gap-2">
        <div className="flex items-center gap-1.5 text-[9px] font-semibold text-zinc-500 uppercase tracking-wider font-mono">
          <Key className="w-3.5 h-3.5 text-brand-indigo" />
          <span>Model API Key</span>
        </div>
        <div className="relative flex items-center">
          <input
            type={showKey ? "text" : "password"}
            placeholder="sk-proj-..."
            value={apiKey}
            onChange={(e) => onApiKeyChange(e.target.value)}
            className="w-full bg-zinc-900 border border-white/5 focus:border-brand-indigo outline-none text-[10px] pl-3 pr-8 py-2 rounded-lg text-zinc-300 placeholder-zinc-700 font-mono transition-all duration-300"
          />
          <button
            type="button"
            onClick={() => setShowKey(!showKey)}
            className="absolute right-2 text-zinc-500 hover:text-zinc-300 transition-colors p-1"
          >
            {showKey ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
          </button>
        </div>
        <div className="text-[8px] text-zinc-600 font-mono text-center mt-1">
          Powered by OpenAI gpt-4o-mini
        </div>
      </div>
    </aside>
  );
}
