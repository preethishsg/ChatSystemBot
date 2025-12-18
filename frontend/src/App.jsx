import React, { useState, useEffect, useRef } from 'react';
import { Send, Database, Search, FileText, Loader2 } from 'lucide-react';

export default function RAGChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [apiUrl, setApiUrl] = useState('https://preethishsg-reg-system-backend.hf.space');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${apiUrl}/stats`);
      if (!response.ok) return;
      const data = await response.json();
      setStats(data);
    } catch {
      // ❌ silently ignore
    }
  };

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(`${apiUrl}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input })
      });

      if (!response.ok) return;

      const data = await response.json();

      // ❌ DO NOT SHOW MODEL / API ERRORS
      if (
        !data?.answer ||
        typeof data.answer !== 'string' ||
        data.answer.toLowerCase().includes('huggingface') ||
        data.answer.toLowerCase().includes('model error') ||
        data.answer.toLowerCase().includes('error')
      ) {
        return;
      }

      const assistantMessage = {
        type: 'assistant',
        content: data.answer,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch {
      // ❌ silently ignore
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto p-6 max-w-6xl">

        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            RAG System Chat
          </h1>
          <p className="text-slate-400">
            Powered by Custom Vector DB & LLM
          </p>
        </div>

        {stats && (
          <div className="grid grid-cols-3 gap-4 mb-6">
            <StatCard icon={<Database />} label="Documents" value={stats.total_documents} />
            <StatCard icon={<FileText />} label="Dimension" value={stats.dimension} />
            <StatCard icon={<Search />} label="Status" value="Active" />
          </div>
        )}

        <div className="bg-white/10 backdrop-blur-sm rounded-2xl border border-white/20 shadow-2xl overflow-hidden">
          <div className="h-[500px] overflow-y-auto p-6 space-y-4">

            {messages.length === 0 && (
              <EmptyState />
            )}

            {messages.map((message, index) => (
              <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[80%] rounded-2xl p-4 ${
                  message.type === 'user'
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500'
                    : 'bg-white/10 border border-white/20'
                }`}>
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  <p className="text-xs text-slate-400 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-white/10 border border-white/20 rounded-2xl p-4 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-slate-400">Retrieving and generating...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="border-t border-white/20 p-4 bg-black/20">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question..."
                className="flex-1 bg-white/10 border border-white/20 rounded-lg px-4 py-3 focus:outline-none"
                disabled={loading}
              />
              <button
                onClick={handleSubmit}
                disabled={loading || !input.trim()}
                className="bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg px-6 py-3 font-semibold"
              >
                {loading ? <Loader2 className="animate-spin" /> : <Send />}
              </button>
            </div>

            <div className="mt-3 flex items-center gap-2 text-xs text-slate-400">
              Backend:
              <input
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="bg-white/5 border border-white/10 rounded px-2 py-1 font-mono"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- Small Components ---------- */

const StatCard = ({ icon, label, value }) => (
  <div className="bg-white/10 rounded-lg p-4 border border-white/20">
    <div className="flex items-center gap-2 text-slate-400 text-sm">
      {icon}
      {label}
    </div>
    <p className="text-2xl font-bold">{value}</p>
  </div>
);

const EmptyState = () => (
  <div className="h-full flex items-center justify-center text-center">
    <Database className="w-16 h-16 mx-auto mb-4 text-purple-400 opacity-50" />
    <p className="text-slate-400 text-lg">Ask a question to begin</p>
  </div>
);
