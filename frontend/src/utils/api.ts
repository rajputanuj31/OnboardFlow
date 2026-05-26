export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface IngestRequest {
  repo_url: string;
  session_id: string;
}

export interface IngestResponse {
  status: string;
  repo_name: string;
  repo_description: string;
  summary: string;
  files_fetched: string[];
  repo_language: string;
  repo_stars: number;
  architecture: string;
}

export interface QuestionRequest {
  session_id: string;
  question: string;
}

export interface QuestionResponse {
  answer: string;
  chat_history: { role: string; content: string }[];
}

export interface SessionResponse {
  exists: boolean;
  repo_name?: string;
  files_fetched?: string[];
  repo_language?: string;
  repo_stars?: number;
  architecture?: string;
  chat_history?: { role: string; content: string }[];
}

export interface HealthResponse {
  status: string;
  sessions: number;
}

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const localKey = typeof window !== 'undefined' ? localStorage.getItem('model_api_key') || '' : '';
  
  const defaultHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };
  if (localKey) {
    defaultHeaders['x-model-api-key'] = localKey;
  }

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options?.headers,
    },
  });

  if (!response.ok) {
    let errorMessage = `HTTP error! Status: ${response.status}`;
    try {
      const errorData = await response.json();
      if (errorData && errorData.detail) {
        if (typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          errorMessage = errorData.detail.map((d: any) => d.msg || JSON.stringify(d)).join(', ');
        }
      }
    } catch {
      // JSON parsing failed, fallback to default status text
      errorMessage = response.statusText || errorMessage;
    }
    throw new ApiError(errorMessage, response.status);
  }

  return response.json() as Promise<T>;
}

export const api = {
  async ingest(req: IngestRequest): Promise<IngestResponse> {
    return request<IngestResponse>('/ingest', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  async ask(req: QuestionRequest): Promise<QuestionResponse> {
    return request<QuestionResponse>('/ask', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  async getSession(sessionId: string): Promise<SessionResponse> {
    return request<SessionResponse>(`/session/${encodeURIComponent(sessionId)}`, {
      method: 'GET',
    });
  },

  async getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>('/health', {
      method: 'GET',
    });
  },
};
