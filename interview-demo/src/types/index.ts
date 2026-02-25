export interface ChatRequest {
  user_id: string;
  thread_id?: string;
  message: string;
  password_verified?: boolean;
}

export interface VerifyPasswordRequest {
  user_id: string;
  thread_id?: string;
  password: string;
  retry_count: number;
}

export interface VerifyPasswordResponse {
  success: boolean;
  message: string;
  retry_count: number;
  locked: boolean;
}

export interface RetrievedDoc {
  id: string;
  title: string;
  source: string;
  content: string;
  score?: number;
}

export interface IntentResult {
  intent: 'symptom' | 'process' | 'mixed' | 'non_medical';
  confidence: number;
  has_symptom: boolean;
  has_process: boolean;
  symptom_query?: string;
  process_query?: string;
  need_symptom_search: boolean;
  need_process_search: boolean;
}

export interface UsedDocs {
  medical: RetrievedDoc[];
  process: RetrievedDoc[];
}

export interface ChatResponse {
  user_id: string;
  thread_id: string;
  reply: string;
  intent_result?: IntentResult;
  used_docs: UsedDocs;
  need_password_input?: boolean;
  password_prompt?: string;
  password_retry_count?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  intent_result?: IntentResult;
  used_docs?: UsedDocs;
}

export interface ThreadInfo {
  thread_id: string;
  title: string;
  created_at: string;
  last_active_at: string;
  is_deleted: boolean;
}

export interface UserInfo {
  user_id: string;
  name?: string;
  created_at: string;
}

export interface UserState {
  user_id: string;
  name?: string;
}

export interface ThreadState {
  thread_id: string;
  title: string;
}
