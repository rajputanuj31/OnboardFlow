'use client';

import React, { useState, useEffect } from 'react';
import { api } from '../utils/api';
import { Session } from './SessionSidebar';
import { GitBranch, Play, AlertCircle, RefreshCw } from 'lucide-react';

interface IngestPanelProps {
  onIngestSuccess: (session: Session) => void;
  existingSessionIds: string[];
  apiKey: string;
}

const STEPS = [
  'Connecting to GitHub Repository...',
  'Traversing directory structure and fetching config files...',
  'Extracting README and CONTRIBUTING guidelines...',
  'Invoking LangGraph Agent workflow...',
  'Generating repository summary & architecture notes in parallel...',
];

export default function IngestPanel({ onIngestSuccess, existingSessionIds, apiKey }: IngestPanelProps) {
  const [repoUrl, setRepoUrl] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [isFastForwarding, setIsFastForwarding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Auto-generate session ID from repository URL
  useEffect(() => {
    if (repoUrl) {
      try {
        const cleaned = repoUrl.replace(/\/$/, '');
        const parts = cleaned.split('/');
        if (parts.length >= 2) {
          const owner = parts[parts.length - 2];
          const name = parts[parts.length - 1];
          if (owner && name) {
            // Remove special characters
            const cleanId = `${owner}-${name}`.replace(/[^a-zA-Z0-9-_]/g, '').toLowerCase();
            setSessionId(cleanId);
            return;
          }
        }
      } catch {
        // Fallback to empty if URL is partially written/invalid
      }
    }
    setSessionId('');
  }, [repoUrl]);

  // Stepper effect during loading
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (loading && !isFastForwarding) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev < STEPS.length - 1) {
            return prev + 1;
          }
          return prev;
        });
      }, 7000);
    } else if (!loading) {
      setCurrentStep(0);
      setIsFastForwarding(false);
    }
    return () => clearInterval(timer);
  }, [loading, isFastForwarding]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!repoUrl.startsWith('https://github.com/')) {
      setError('Please enter a valid GitHub URL starting with https://github.com/');
      return;
    }

    if (!sessionId.trim()) {
      setError('Session ID cannot be empty');
      return;
    }

    if (existingSessionIds.includes(sessionId.trim())) {
      setError(`Session ID "${sessionId}" already exists. Overwrite? (Submit again to overwrite)`);
      // Allow overwriting if clicked again, but warn first
      if (error !== `Session ID "${sessionId}" already exists. Overwrite? (Submit again to overwrite)`) {
        return;
      }
    }

    setLoading(true);
    setCurrentStep(0);
    setIsFastForwarding(false);

    try {
      const response = await api.ingest({
        repo_url: repoUrl.trim(),
        session_id: sessionId.trim(),
      });

      const newSession: Session = {
        id: sessionId.trim(),
        repoUrl: repoUrl.trim(),
        repoName: response.repo_name,
        repoDescription: response.repo_description,
        summary: response.summary,
        filesFetched: response.files_fetched,
        repoLanguage: response.repo_language,
        repoStars: response.repo_stars,
        architecture: response.architecture,
      };

      // Fast-forward remaining steps for UX satisfaction
      setIsFastForwarding(true);
      
      let nextStep = currentStep;
      const ffInterval = setInterval(() => {
        nextStep += 1;
        setCurrentStep(nextStep);
        
        if (nextStep >= STEPS.length) {
          clearInterval(ffInterval);
          // Wait 800ms on the fully completed screen for satisfying feedback
          setTimeout(() => {
            setLoading(false);
            setIsFastForwarding(false);
            onIngestSuccess(newSession);
          }, 800);
        }
      }, 400);

    } catch (err: any) {
      setError(err.message || 'An error occurred during ingestion. Please check the backend log.');
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center p-8 bg-zinc-900/10 min-h-screen">
      <div className={`w-full max-w-xl glass-card rounded-2xl p-8 relative overflow-hidden transition-all duration-500 ${
        loading ? 'animate-pulse-glow' : 'glow-permanent'
      }`}>
        
        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-zinc-950/90 backdrop-blur-md flex flex-col items-center justify-center p-8 z-50">
            <div className="relative mb-6">
              <div className="w-16 h-16 rounded-full border-4 border-zinc-800 border-t-brand-indigo animate-spin"></div>
              <div className="absolute inset-2 bg-zinc-950 rounded-full flex items-center justify-center">
                <RefreshCw className="w-6 h-6 text-brand-indigo animate-pulse" />
              </div>
            </div>
            
            <h3 className="text-sm font-semibold text-zinc-200 mb-2">Analyzing Repository</h3>
            <p className="text-xs text-zinc-500 text-center max-w-sm mb-6 leading-relaxed">
              This can take 20-40 seconds depending on size. Fetch results are cached locally.
            </p>

            {/* Stepper display */}
            <div className="w-full max-w-sm space-y-3.5">
              {STEPS.map((step, idx) => {
                const isPassed = idx < currentStep;
                const isCurrent = idx === currentStep;
                return (
                  <div key={idx} className="flex items-center gap-3">
                    <div
                      className={`w-4 h-4 rounded-full flex items-center justify-center text-[9px] font-bold ${
                        isPassed
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : isCurrent
                          ? 'bg-brand-indigo/20 text-brand-indigo border border-brand-indigo animate-pulse'
                          : 'bg-zinc-900 text-zinc-600 border border-zinc-800'
                      }`}
                    >
                      {isPassed ? '✓' : idx + 1}
                    </div>
                    <span
                      className={`text-xs ${
                        isPassed ? 'text-zinc-500 line-through' : isCurrent ? 'text-brand-indigo font-medium' : 'text-zinc-600'
                      }`}
                    >
                      {step}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Normal Form Header */}
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-brand-indigo/10 flex items-center justify-center text-brand-indigo">
            <GitBranch className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-zinc-100">Ingest GitHub Repository</h2>
            <p className="text-xs text-zinc-500">Provide any public repo to extract architecture notes and start a chat.</p>
          </div>
        </div>

        {!apiKey && (
          <div className="mb-6 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-start gap-3 shadow-lg shadow-amber-500/5">
            <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="text-xs text-amber-300 leading-relaxed">
              <span className="font-semibold block mb-0.5 text-amber-200">Model API Key Required</span>
              Please configure your OpenAI API Key in the sidebar settings. Your key is stored locally in your browser's <code className="bg-amber-500/15 px-1 py-0.5 rounded text-[10px] font-mono text-amber-200">localStorage</code> and never shared.
            </div>
          </div>
        )}

        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <div className="text-xs text-rose-300 font-medium leading-relaxed">{error}</div>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Repo URL */}
          <div className="space-y-2">
            <label className="block text-xs font-semibold text-zinc-400">Repository URL</label>
            <input
              type="url"
              required
              placeholder="https://github.com/psf/requests"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              className="w-full bg-zinc-950 border border-white/5 focus:border-brand-indigo focus:ring-1 focus:ring-brand-indigo outline-none text-sm px-4 py-3 rounded-xl text-zinc-200 placeholder-zinc-700 transition-all duration-300"
            />
          </div>


          {/* Session ID */}
          <div className="space-y-2">
            <label className="block text-xs font-semibold text-zinc-400">Session ID</label>
            <input
              type="text"
              required
              placeholder="e.g. requests-v1"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              className="w-full bg-zinc-950 border border-white/5 focus:border-brand-indigo focus:ring-1 focus:ring-brand-indigo outline-none text-sm px-4 py-3 rounded-xl text-zinc-200 placeholder-zinc-700 font-mono transition-all duration-300"
            />
            <p className="text-[10px] text-zinc-600 font-mono">
              Identifier to query this repository session later without re-ingesting.
            </p>
          </div>

          {/* Submit */}
          <button
            type="submit"
            className="w-full flex items-center justify-center gap-2 py-3.5 px-4 rounded-xl bg-gradient-to-r from-brand-indigo via-brand-violet to-brand-fuchsia hover:brightness-110 text-white font-semibold text-sm shadow-xl shadow-brand-indigo/10 transition-all duration-300 transform active:scale-[0.98]"
          >
            <Play className="w-4 h-4 fill-current" />
            Analyze Repository
          </button>
        </form>

        {/* Quick Suggestion Box */}
        <div className="mt-8 border-t border-white/5 pt-5">
          <span className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider block mb-3">
            Quick Examples
          </span>
          <div className="flex flex-wrap gap-2">
            {[
              { label: 'requests (Python)', url: 'https://github.com/psf/requests' },
              { label: 'fastapi (Python)', url: 'https://github.com/fastapi/fastapi' },
            ].map((suggest) => (
              <button
                key={suggest.url}
                onClick={() => setRepoUrl(suggest.url)}
                className="text-[11px] bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-zinc-200 py-1.5 px-3 rounded-lg border border-white/5 transition-all duration-200"
                type="button"
              >
                {suggest.label}
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
