import React, { useState, useEffect, useRef } from 'react';
import { Send, Database, Search, FileText, Loader2 } from 'lucide-react';

export default function RAGChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [apiUrl, setApiUrl] = 
useState('https://preethishsg-reg-system-backend.hf.space');
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
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Error fetching stats:', error);
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
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
          k: 3        
        })
      });

      const data = await response.json();

      const assistantMessage = {
        type: 'assistant',
        content: data.answer,
        retrievedDocs: data.retrieved_documents,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: `Error: ${error.message}. Make sure the backend is 
running at ${apiUrl}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
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
    <div className="min-h-screen bg-gradient-to-br from-slate-900 
via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto p-6 max-w-6xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r 
from-purple-400 to-pink-400 bg-clip-text text-transparent">
            RAG System Chat
          </h1>
          <p className="text-slate-400">
            Powered by BGE-micro embeddings, Custom Vector DB & Mistral AI
          </p>
        </div>

        {stats && (
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 
border border-white/20">
              <div className="flex items-center gap-2 mb-1">
                <Database className="w-4 h-4 text-purple-400" />
                <span className="text-sm text-slate-400">Documents</span>
              </div>
              <p className="text-2xl 
font-bold">{stats.total_documents}</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 
border border-white/20">
              <div className="flex items-center gap-2 mb-1">
                <FileText className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-slate-400">Dimension</span>
              </div>
              <p className="text-2xl font-bold">{stats.dimension}</p>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 
border border-white/20">
              <div className="flex items-center gap-2 mb-1">
                <Search className="w-4 h-4 text-green-400" />
                <span className="text-sm text-slate-400">Status</span>
              </div>
              <p className="text-lg font-semibold 
text-green-400">Active</p>
            </div>
          </div>
        )}

        <div className="bg-white/10 backdrop-blur-sm rounded-2xl border 
border-white/20 shadow-2xl overflow-hidden">
          <div className="h-[500px] overflow-y-auto p-6 space-y-4">
            {messages.length === 0 && (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <Database className="w-16 h-16 mx-auto mb-4 
text-purple-400 opacity-50" />
                  <p className="text-slate-400 text-lg">
                    Ask a question to retrieve and generate answers
                  </p>
                  <p className="text-slate-500 text-sm mt-2">
                    Try: "What is machine learning?" or "Explain neural 
networks"
                  </p>
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.type === 'user' ? 'justify-end' 
: 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl p-4 ${
                    message.type === 'user'
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500'
                      : message.type === 'error'
                      ? 'bg-red-500/20 border border-red-500/50'
                      : 'bg-white/10 border border-white/20'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>
                  
                  {message.retrievedDocs && message.retrievedDocs.length > 
0 && ( <div>
                   
                      <p className="text-xs text-slate-400 mb-2">
                        📚 Retrieved {message.retrievedDocs.length} 
documents
                      </p>
                      <div className="space-y-2">
                        {message.retrievedDocs.map((doc, i) => (
                          <div key={i} className="text-xs bg-black/20 
rounded p-2">
                            <div className="flex items-center 
justify-between mb-1">
                              <span className="font-mono 
text-purple-300">{doc.id}</span>
                              <span className="text-green-400">
                                {(doc.score * 100).toFixed(1)}% match
                              </span>
                            </div>
                            <p className="text-slate-300 line-clamp-2">
                              {doc.metadata.text}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <p className="text-xs text-slate-400 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-white/10 border border-white/20 
rounded-2xl p-4">
                  <div className="flex items-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-slate-400">Retrieving and 
generating...</span>
                  </div>
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
                className="flex-1 bg-white/10 border border-white/20 
rounded-lg px-4 py-3 focus:outline-none focus:border-purple-400 
focus:ring-2 focus:ring-purple-400/20 placeholder-slate-500"
                disabled={loading}
              />
              <button
                onClick={handleSubmit}
                disabled={loading || !input.trim()}
                className="bg-gradient-to-r from-purple-500 to-pink-500 
hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 
disabled:cursor-not-allowed rounded-lg px-6 py-3 font-semibold 
transition-all transform hover:scale-105 flex items-center gap-2"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
                Send
              </button>
            </div>
            
            <div className="mt-3 flex items-center gap-2 text-xs 
text-slate-400">
              <span>Backend:</span>
              <input
                type="text"
                value={apiUrl}
                onChange={(e) => setApiUrl(e.target.value)}
                className="bg-white/5 border border-white/10 rounded px-2 
py-1 text-slate-300 font-mono text-xs"
                placeholder="http://localhost:8000"
              />
            </div>
          </div>
        </div>

        <div className="mt-6 text-center text-slate-400 text-sm">
          <p>Custom Vector DB With FIASS | BGE-micro Embeddings | 
Mistral AI Generation</p>
        </div>
      </div>
    </div>
  );
}
