import { useState, useCallback, useEffect, useRef } from 'react';
import { chat as chatApi, verifyPassword } from '../lib/api';
import type { Message, ChatResponse } from '../types';

export function useChat(userId: string | null, threadId: string | null) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(threadId);
  const [needPasswordInput, setNeedPasswordInput] = useState(false);
  const [passwordPrompt, setPasswordPrompt] = useState('');
  const [passwordRetryCount, setPasswordRetryCount] = useState(0);
  const [passwordVerified, setPasswordVerified] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setCurrentThreadId(threadId);
  }, [threadId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handlePasswordVerified = useCallback(async () => {
    if (!pendingMessage || !userId) return;

    setIsLoading(true);
    try {
      const response: ChatResponse = await chatApi({
        user_id: userId,
        thread_id: currentThreadId || undefined,
        message: pendingMessage,
        password_verified: true,
      });

      if (response.thread_id && response.thread_id !== currentThreadId) {
        setCurrentThreadId(response.thread_id);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.reply,
        timestamp: Date.now(),
        intent_result: response.intent_result,
        used_docs: response.used_docs,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setNeedPasswordInput(false);
      setPasswordPrompt('');
      setPasswordRetryCount(0);
      setPasswordVerified(false);
      setPendingMessage(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '请求失败，请检查后端服务';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [pendingMessage, userId, currentThreadId]);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || !userId) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: content.trim(),
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response: ChatResponse = await chatApi({
        user_id: userId,
        thread_id: currentThreadId || undefined,
        message: content.trim(),
        password_verified: passwordVerified,
      });

      // 检查是否需要密码验证
      if (response.need_password_input) {
        // 保存待发送的消息
        setPendingMessage(content.trim());
        setNeedPasswordInput(true);
        setPasswordPrompt(response.password_prompt || '请输入密码');
        setPasswordRetryCount(response.password_retry_count || 0);
        setIsLoading(false);
        return;
      }

      if (response.thread_id && response.thread_id !== currentThreadId) {
        setCurrentThreadId(response.thread_id);
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.reply,
        timestamp: Date.now(),
        intent_result: response.intent_result,
        used_docs: response.used_docs,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '请求失败，请检查后端服务';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [userId, currentThreadId, passwordVerified]);

  const handlePasswordSubmit = useCallback(async (password: string) => {
    if (!userId) return;

    setIsLoading(true);
    try {
      const response = await verifyPassword({
        user_id: userId,
        thread_id: currentThreadId || undefined,
        password,
        retry_count: passwordRetryCount,
      });

      if (response.success) {
        setPasswordVerified(true);
        // 验证成功后，发送待处理的消息
        if (pendingMessage) {
          await handlePasswordVerified();
        }
      } else {
        setError(response.message);
        if (response.locked) {
          setNeedPasswordInput(false);
          setPasswordPrompt('');
          setPasswordRetryCount(0);
          setPendingMessage(null);
        } else {
          setPasswordRetryCount(response.retry_count);
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '密码验证请求失败';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [userId, currentThreadId, passwordRetryCount, pendingMessage, handlePasswordVerified]);

  const cancelPasswordInput = useCallback(() => {
    setNeedPasswordInput(false);
    setPasswordPrompt('');
    setPasswordRetryCount(0);
    setPendingMessage(null);
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
    setNeedPasswordInput(false);
    setPasswordPrompt('');
    setPasswordRetryCount(0);
    setPasswordVerified(false);
    setPendingMessage(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    currentThreadId,
    needPasswordInput,
    passwordPrompt,
    passwordRetryCount,
    sendMessage,
    handlePasswordSubmit,
    cancelPasswordInput,
    clearMessages,
    messagesEndRef,
  };
}
