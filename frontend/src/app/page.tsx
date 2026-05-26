'use client';

import React, { useState, useEffect } from 'react';
import SessionSidebar, { Session } from '../components/SessionSidebar';
import IngestPanel from '../components/IngestPanel';
import RepoSummary from '../components/RepoSummary';
import ChatWorkspace from '../components/ChatWorkspace';
import { api } from '../utils/api';
import { AlertCircle, RefreshCw, Menu, X, BookOpen, MessageSquare, Plus } from 'lucide-react';

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [showIngest, setShowIngest] = useState(true);
  const [restoring, setRestoring] = useState(false);
  const [isClient, setIsClient] = useState(false);
  const [apiKey, setApiKey] = useState('');
  
  // Responsive states
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [mobileTab, setMobileTab] = useState<'docs' | 'chat'>('docs');

  // Mark client side mount
  useEffect(() => {
    setIsClient(true);
    
    // Load sessions from localStorage
    const saved = localStorage.getItem('oss_sessions');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setSessions(parsed);
        
        const lastActive = localStorage.getItem('oss_active_session_id');
        if (lastActive && parsed.some((s: Session) => s.id === lastActive)) {
          setActiveSessionId(lastActive);
          setShowIngest(false);
          verifyAndRestoreSession(lastActive, parsed);
        }
      } catch (e) {
        console.error('Failed to parse saved sessions', e);
      }
    }

    // Load API Key
    const savedKey = localStorage.getItem('model_api_key') || '';
    setApiKey(savedKey);
  }, []);

  const handleApiKeyChange = (val: string) => {
    setApiKey(val);
    if (typeof window !== 'undefined') {
      localStorage.setItem('model_api_key', val);
    }
  };

  // Save sessions to localStorage
  const saveSessions = (updated: Session[]) => {
    setSessions(updated);
    if (isClient) {
      localStorage.setItem('oss_sessions', JSON.stringify(updated));
    }
  };

  // Verify if session exists in backend; if not, automatically re-ingest in background
  const verifyAndRestoreSession = async (id: string, currentSessionsList: Session[]) => {
    const sessionObj = currentSessionsList.find((s) => s.id === id);
    if (!sessionObj) return;

    try {
      const response = await api.getSession(id);
      if (!response.exists) {
        setRestoring(true);
        const ingestRes = await api.ingest({
          repo_url: sessionObj.repoUrl,
          session_id: id,
        });
        const updatedSessions = currentSessionsList.map((s) => {
          if (s.id === id) {
            return {
              ...s,
              repoLanguage: ingestRes.repo_language,
              repoStars: ingestRes.repo_stars,
              architecture: ingestRes.architecture,
            };
          }
          return s;
        });
        saveSessions(updatedSessions);
      } else {
        // Sync local storage to keep state fresh with backend ground truth
        const updatedSessions = currentSessionsList.map((s) => {
          if (s.id === id) {
            return {
              ...s,
              repoLanguage: response.repo_language || s.repoLanguage,
              repoStars: response.repo_stars || s.repoStars,
              architecture: response.architecture || s.architecture,
              chatHistory: response.chat_history?.map((msg) => ({
                role: msg.role as 'user' | 'assistant',
                content: msg.content,
              })) || s.chatHistory || [],
            };
          }
          return s;
        });
        saveSessions(updatedSessions);
      }
    } catch (err) {
      console.error('Error verifying/restoring session in backend', err);
    } finally {
      setRestoring(false);
    }
  };

  const handleSelectSession = (id: string) => {
    setActiveSessionId(id);
    setShowIngest(false);
    setIsSidebarOpen(false); // Close drawer on mobile
    if (isClient) {
      localStorage.setItem('oss_active_session_id', id);
    }
    verifyAndRestoreSession(id, sessions);
  };

  const handleNewSessionClick = () => {
    setShowIngest(true);
    setActiveSessionId(null);
    setIsSidebarOpen(false); // Close drawer on mobile
    if (isClient) {
      localStorage.removeItem('oss_active_session_id');
    }
  };

  const handleDeleteSession = (id: string) => {
    const updated = sessions.filter((s) => s.id !== id);
    saveSessions(updated);

    if (activeSessionId === id) {
      setActiveSessionId(null);
      setShowIngest(true);
      if (isClient) {
        localStorage.removeItem('oss_active_session_id');
      }
    }
  };

  const handleIngestSuccess = (newSession: Session) => {
    const index = sessions.findIndex((s) => s.id === newSession.id);
    let updated;
    if (index !== -1) {
      updated = [...sessions];
      updated[index] = newSession;
    } else {
      updated = [newSession, ...sessions];
    }
    saveSessions(updated);
    setActiveSessionId(newSession.id);
    setShowIngest(false);
    if (isClient) {
      localStorage.setItem('oss_active_session_id', newSession.id);
    }
  };

  const handleChatHistoryChange = (sessionId: string, history: any[]) => {
    const updated = sessions.map((s) => {
      if (s.id === sessionId) {
        return { ...s, chatHistory: history };
      }
      return s;
    });
    saveSessions(updated);
  };

  const activeSession = sessions.find((s) => s.id === activeSessionId);

  // Avoid SSR mismatch by waiting for client mount
  if (!isClient) {
    return (
      <div className="flex h-screen items-center justify-center bg-background text-foreground">
        <div className="w-8 h-8 rounded-full border-4 border-zinc-800 border-t-brand-indigo animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground relative">
      
      {/* 1. Mobile Sidebar Drawer Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-40 md:hidden backdrop-blur-sm transition-opacity duration-300"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* 2. Responsive Session Sidebar Container */}
      <div className={`fixed inset-y-0 left-0 w-80 z-50 md:relative md:translate-x-0 transform transition-transform duration-300 md:block shrink-0 h-full ${
        isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <SessionSidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={handleSelectSession}
          onNewSessionClick={handleNewSessionClick}
          onDeleteSession={handleDeleteSession}
          onCloseClick={() => setIsSidebarOpen(false)}
          apiKey={apiKey}
          onApiKeyChange={handleApiKeyChange}
        />
      </div>

      {/* 3. Main Viewport */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        
        {/* Mobile top navigation header */}
        <header className="flex md:hidden h-14 bg-zinc-950/80 border-b border-white/5 px-4 items-center justify-between shrink-0 z-30 backdrop-blur-md">
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 text-zinc-400 hover:text-zinc-100 hover:bg-white/5 rounded-xl transition-all duration-200"
            aria-label="Toggle Navigation Drawer"
          >
            {isSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          
          <div className="font-semibold text-xs tracking-wider uppercase text-zinc-300 font-mono">
            {activeSession ? activeSession.id : 'OnboardFlow'}
          </div>

          <button
            onClick={handleNewSessionClick}
            className="p-2 text-brand-indigo hover:text-white hover:bg-brand-indigo/15 rounded-xl transition-all duration-200"
            aria-label="Start New Ingestion"
          >
            <Plus className="w-5 h-5" />
          </button>
        </header>

        {/* Mobile View Toggle (Segmented tab selector) */}
        {activeSession && !showIngest && (
          <div className="flex lg:hidden bg-zinc-950/50 border-b border-white/5 p-1 px-4 gap-1 select-none z-20 shrink-0">
            <button
              onClick={() => setMobileTab('docs')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-semibold transition-all duration-300 ${
                mobileTab === 'docs'
                  ? 'bg-brand-indigo/15 text-brand-indigo border border-brand-indigo/20 shadow-inner'
                  : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              Overview
            </button>
            <button
              onClick={() => setMobileTab('chat')}
              className={`flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-xs font-semibold transition-all duration-300 ${
                mobileTab === 'chat'
                  ? 'bg-brand-indigo/15 text-brand-indigo border border-brand-indigo/20 shadow-inner'
                  : 'text-zinc-500 hover:text-zinc-300 border border-transparent'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              Conversational Chat
            </button>
          </div>
        )}

        {/* Restoring Session indicator */}
        {restoring && (
          <div className="bg-brand-indigo/10 border-b border-brand-indigo/20 px-6 py-2 flex items-center justify-between text-xs text-brand-indigo animate-pulse z-10">
            <div className="flex items-center gap-2">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span className="text-[11px]">Restoring agent context in backend...</span>
            </div>
          </div>
        )}

        {/* Dynamic content rendering */}
        <div className="flex-1 flex flex-col h-full overflow-hidden min-h-0">
          {showIngest || !activeSession ? (
            <IngestPanel
              onIngestSuccess={handleIngestSuccess}
              existingSessionIds={sessions.map((s) => s.id)}
              apiKey={apiKey}
            />
          ) : (
            <div className="flex-1 flex h-full overflow-hidden min-h-0">
              
              {/* Desktop layout: show side-by-side splits */}
              <div className="hidden lg:flex flex-1 h-full overflow-hidden">
                <RepoSummary session={activeSession} />
                <ChatWorkspace
                  sessionId={activeSession.id}
                  initialHistory={activeSession.chatHistory || []}
                  onChatHistoryChange={(history) => handleChatHistoryChange(activeSession.id, history)}
                />
              </div>

              {/* Mobile/Tablet layout: toggle viewports based on tab */}
              <div className="flex lg:hidden flex-1 h-full overflow-hidden">
                {mobileTab === 'docs' ? (
                  <RepoSummary session={activeSession} />
                ) : (
                  <ChatWorkspace
                    sessionId={activeSession.id}
                    initialHistory={activeSession.chatHistory || []}
                    onChatHistoryChange={(history) => handleChatHistoryChange(activeSession.id, history)}
                  />
                )}
              </div>

            </div>
          )}
        </div>
      </main>
    </div>
  );
}
