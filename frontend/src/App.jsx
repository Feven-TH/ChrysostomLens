import { useState, useEffect, useRef } from 'react';
import {
  Flame,
  BookOpen,
  FileText,
  Send,
  Sun,
  Moon,
  ChevronRight,
  Menu,
  X,
  Trash2
} from 'lucide-react';

function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('chrysostom-theme') || 'vespers';
  });
  
  const [status, setStatus] = useState({
    status: 'checking',
    message: 'Connecting to Golden-Mouthed server...'
  });
  
  // Single source of truth for chat history, loaded from localStorage
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('chrysostom-messages');
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error("Failed to parse saved messages", e);
      }
    }
    return [
      {
        role: 'assistant',
        content: "Welcome, pilgrim. I am your theological assistant for St. John Chrysostom's Homilies on the Gospel of Matthew. Ask me questions about the scriptures, the moral teachings of the Golden-Mouthed, or details within his homilies.",
        sources: [],
        loading: false
      }
    ];
  });
  
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeSources, setActiveSources] = useState(null);
  const [activeSourceIndex, setActiveSourceIndex] = useState(0);
  const [activeCitationKey, setActiveCitationKey] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const messagesEndRef = useRef(null);
  const sourceContentRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Sync theme to DOM
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('chrysostom-theme', theme);
  }, [theme]);

  // Sync chat history to localStorage (with loading states cleared)
  useEffect(() => {
    const cleanMessages = messages.map(msg => {
      if (msg.loading) {
        return { ...msg, loading: false, content: msg.content || 'Consultation interrupted.' };
      }
      return msg;
    });
    localStorage.setItem('chrysostom-messages', JSON.stringify(cleanMessages));
  }, [messages]);

// Fetch status on load
  const fetchStatus = async () => {
    try {
      setStatus(prev => ({ ...prev, status: 'checking', message: 'Connecting...' }));

      const backendUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/api/status`);
      
      const data = await response.json();
      setStatus(data);
    } catch (err) {
      console.error(err);
      setStatus({
        status: 'error',
        message: 'Could not connect to the server. Please make sure the backend is running.'
      });
    }
  };

  useEffect(() => {
    fetchStatus();
    // Cleanup request on unmount
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'daylight' ? 'vespers' : 'daylight');
  };

  const handleSuggestQuery = (query) => {
    if (loading) return;
    setInput(query);
    setSidebarOpen(false);
  };

  const handleClearChat = () => {
    if (window.confirm("Are you sure you want to clear your study history?")) {
      const defaultMsg = [
        {
          role: 'assistant',
          content: "Welcome, pilgrim. I am your theological assistant for St. John Chrysostom's Homilies on the Gospel of Matthew. Ask me questions about the scriptures, the moral teachings of the Golden-Mouthed, or details within his homilies.",
          sources: [],
          loading: false
        }
      ];
      setMessages(defaultMsg);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userQuery = input.trim();
    setInput('');
    setLoading(true);

    // Grab the last 4 messages (2 full chat turns) to serve as sliding history window context
    const historySlice = messages
      .filter(msg => !msg.loading && !msg.isError)
      .slice(-4)
      .map(msg => ({
        role: msg.role,
        content: msg.content
      }));

    // Add user message and a placeholder assistant message
    setMessages(prev => [
      ...prev,
      { role: 'user', content: userQuery },
      { role: 'assistant', content: '', sources: [], loading: true }
    ]);

    // Abort previous stream connection if active
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      // Switch the request to a POST request to send query and sliding window history payload
      const backendUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/api/query`,{
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userQuery,
          history: historySlice
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Process the SSE response stream chunk-by-chunk using a Reader loop
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Retain the last unfinished line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);
            try {
              const data = JSON.parse(dataStr);
              if (data.type === 'sources') {
                setMessages(prev => prev.map((msg, idx) => {
                  if (idx === prev.length - 1) {
                    return { ...msg, sources: data.sources };
                  }
                  return msg;
                }));
              } else if (data.type === 'content') {
                setMessages(prev => prev.map((msg, idx) => {
                  if (idx === prev.length - 1) {
                    return { ...msg, content: (msg.content || '') + data.delta };
                  }
                  return msg;
                }));
              } else if (data.type === 'error') {
                setMessages(prev => prev.map((msg, idx) => {
                  if (idx === prev.length - 1) {
                    return { ...msg, content: data.message, isError: true, loading: false };
                  }
                  return msg;
                }));
                setLoading(false);
              } else if (data.type === 'done') {
                setMessages(prev => prev.map((msg, idx) => {
                  if (idx === prev.length - 1) {
                    return { ...msg, loading: false };
                  }
                  return msg;
                }));
                setLoading(false);
              }
            } catch (err) {
              console.error("SSE JSON parse error", err);
            }
          }
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log("Fetch request aborted.");
        return;
      }
      console.error("Fetch/Stream error", err);
      setMessages(prev => prev.map((msg, idx) => {
        if (idx === prev.length - 1 && msg.loading) {
          return {
            ...msg,
            content: "A connection issue occurred while fetching St. John Chrysostom's Homilies. Please ensure the backend server is running.",
            isError: true,
            loading: false
          };
        }
        return msg;
      }));
      setLoading(false);
    }
  };

  const handleOpenInspector = (sources) => {
    setActiveSources(sources);
    setActiveSourceIndex(0);
    setActiveCitationKey(null);
  };

  const getCitationKey = (source) => {
    if (!source) return '';
    return `H${normalizeCitationHomily(source.homily)} §${source.paragraph_index}`;
  };

  const normalizeCitationHomily = (homily) => {
    const match = String(homily ?? '').match(/\d+/);
    return match ? match[0] : String(homily ?? '');
  };

  const renderMessageContent = (content) => {
    if (!content) return null;
    return content
      .replace(/\s*\[H\d+\s+§\d+\]/g, '')
      .replace(/[ \t]+\n/g, '\n')
      .trim();
  };

  // Helper to extract paragraphs from source content
  const parseSections = (content) => {
    if (!content) return { global: '', local: '', raw: '' };
    
    const globalMatch = content.match(/\[GLOBAL CONTEXT\]:([\s\S]*?)(?=\[LOCAL CONTEXT\]|\[RAW TEXT\]|$)/);
    const localMatch = content.match(/\[LOCAL CONTEXT\]:([\s\S]*?)(?=\[RAW TEXT\]|$)/);
    const rawMatch = content.match(/\[RAW TEXT\]:([\s\S]*?)$/);

    return {
      global: globalMatch ? globalMatch[1].trim() : '',
      local: localMatch ? localMatch[1].trim() : '',
      raw: rawMatch ? rawMatch[1].trim() : content
    };
  };

  return (
    <div className="parchment-texture app-container">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Sidebar Panel - Suggested Queries & Branding */}
      <div className={`gold-border sidebar-panel ${sidebarOpen ? 'open' : ''}`}>
        {/* Branding header */}
        <div style={{
          padding: '1.5rem',
          borderBottom: '1.5px solid var(--border-color)',
          textAlign: 'center',
          backgroundColor: 'var(--bg-tertiary)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', justifyContent: 'center' }}>
            <Flame className="candle-flame" size={24} style={{ color: 'var(--accent-gold)', filter: 'drop-shadow(0 0 4px var(--accent-gold))' }} />
            <span style={{ fontFamily: 'var(--font-title)', fontSize: '1.4rem', fontWeight: 'bold', letterSpacing: '0.05em', color: 'var(--accent-crimson)' }}>
              ChrysostomLens
            </span>
          </div>
          <p style={{ fontSize: '0.75rem', fontStyle: 'italic', color: 'var(--text-secondary)', marginTop: '4px' }}>
            &ldquo;The Golden-Mouthed Study Assistant&rdquo;
          </p>
        </div>

        {/* Suggested Queries Scroll */}
        <div style={{ padding: '1.2rem', flex: 1, display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-crimson)' }}>
            <BookOpen size={16} />
            <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-title)', fontWeight: 'bold' }}>Theological Prompts</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {[
              "What does St. John Chrysostom say about the direct illumination of the Spirit vs written scripture?",
              "How does he interpret the lineage of Christ in Homily 1?",
              "What is his teaching on humility when studying the Gospels?",
              "Why did God give us scriptures instead of writing on our hearts?",
              "How does Chrysostom describe the character of Matthew the publican?"
            ].map((query, index) => (
              <button
                key={index}
                onClick={() => handleSuggestQuery(query)}
                disabled={loading}
                style={{
                  textAlign: 'left',
                  backgroundColor: 'var(--bg-card)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '16px',
                  padding: '12px',
                  fontSize: '0.82rem',
                  lineHeight: 1.45,
                  color: 'var(--text-primary)',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '6px',
                  boxShadow: 'var(--shadow-sm)'
                }}
                className="hover-trigger"
                onMouseEnter={(e) => { if(!loading) e.currentTarget.style.borderColor = 'var(--border-gold)'; }}
                onMouseLeave={(e) => { if(!loading) e.currentTarget.style.borderColor = 'var(--border-color)'; }}
              >
                <ChevronRight size={12} style={{ marginTop: '2px', flexShrink: 0, color: 'var(--accent-gold)' }} />
                <span>{query}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Footer info */}
        <div style={{
          padding: '1rem',
          textAlign: 'center',
          fontSize: '0.65rem',
          color: 'var(--text-muted)',
          borderTop: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-tertiary)'
        }}>
          St. John Chrysostom, Homilies on Matthew
          <div style={{ marginTop: '4px', display: 'flex', justifyContent: 'center', gap: '8px' }}>
            <span>A.D. 390</span>
            <span>&bull;</span>
            <span>Study Dashboard</span>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-container">
        {/* Header toolbar */}
        <div className="header-toolbar">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {/* Mobile hamburger menu */}
            <button className="menu-toggle-btn" onClick={() => setSidebarOpen(true)}>
              <Menu size={22} />
            </button>
            <h2 style={{ fontSize: '1.1rem', color: 'var(--text-primary)' }}>Ask St. John Chrysostom</h2>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* Clear history button */}
            <button
              onClick={handleClearChat}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                color: 'var(--text-secondary)',
                fontSize: '0.8rem',
                fontWeight: 600
              }}
              title="Clear Conversation History"
            >
              <Trash2 size={16} style={{ color: 'var(--accent-crimson)' }} />
              <span>Reset</span>
            </button>

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                color: 'var(--text-secondary)',
                fontSize: '0.8rem',
                fontWeight: 600
              }}
              title="Toggle Vespers/Daylight Theme"
            >
              {theme === 'vespers' ? (
                <>
                  <Sun size={18} style={{ color: 'var(--accent-gold)' }} />
                  <span>Daylight</span>
                </>
              ) : (
                <>
                  <Moon size={18} style={{ color: 'var(--accent-crimson)' }} />
                  <span>Vespers</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Message Feed */}
        <div className="message-feed">
          {messages.map((msg, index) => {
            const isAssistant = msg.role === 'assistant';
            return (
              <div
                key={index}
                className="animate-fade-in chat-message-target"
                style={{
                  display: 'flex',
                  justifyContent: isAssistant ? 'flex-start' : 'flex-end',
                  width: '100%'
                }}
              >
                <div className={`chat-message-bubble ${isAssistant ? 'assistant' : 'user'}`}>
                  {/* Crimson margin accent bar for patristic text */}
                  {isAssistant && (
                    <div style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: '4px',
                      backgroundColor: 'var(--accent-crimson)',
                      borderRadius: '18px 0 0 18px'
                    }} />
                  )}

                  {/* Header / Identity */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                    <span>{isAssistant ? 'St. John Chrysostom (Assistant)' : 'Pilgrim'}</span>
                    {isAssistant && msg.loading && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Flame size={12} className="candle-flame" style={{ color: 'var(--accent-gold)' }} />
                        <span style={{ fontSize: '0.65rem', fontStyle: 'italic' }}>Reading the Homilies...</span>
                      </div>
                    )}
                  </div>

                  {/* Message Content */}
                  <div className={isAssistant ? 'patristic-response' : 'user-response'} style={{
                    color: msg.isError ? 'var(--accent-crimson)' : undefined
                  }}>
                    {isAssistant ? renderMessageContent(msg.content) : msg.content}
                    {isAssistant && msg.loading && !msg.content && (
                      <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>Formulating response...</span>
                    )}
                  </div>

                  {/* Sources button */}
                  {isAssistant && msg.sources && msg.sources.length > 0 && (
                    <div style={{
                      marginTop: '8px',
                      paddingTop: '8px',
                      borderTop: '1px solid var(--border-color)',
                      display: 'flex',
                      flexWrap: 'wrap',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      gap: '8px'
                    }}>
                      <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        Found {msg.sources.length} matching excerpts
                      </span>
                      <button
                        onClick={() => handleOpenInspector(msg.sources)}
                        style={{
                          background: 'none',
                          border: 'none',
                          color: 'var(--accent-crimson)',
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.textDecoration = 'underline'}
                        onMouseLeave={(e) => e.currentTarget.style.textDecoration = 'none'}
                      >
                        <FileText size={12} />
                        View original excerpts
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="input-bar-container">
          <form onSubmit={handleSubmit} style={{
            display: 'flex',
            gap: '8px',
            backgroundColor: 'var(--bg-primary)',
            border: '1px solid var(--border-gold)',
            borderRadius: 'var(--radius-md)',
            padding: '4px 10px',
            boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.05)',
            alignItems: 'center'
          }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={loading || status.status !== 'ready'}
              placeholder={status.status === 'ready' ? "Ask the Golden-Mouthed..." : "Connecting to server..."}
              style={{
                flex: 1,
                border: 'none',
                background: 'none',
                outline: 'none',
                padding: '10px',
                fontSize: '0.9rem',
                color: 'var(--text-primary)',
                fontFamily: 'var(--font-sans)',
                minWidth: 0
              }}
            />
            <button
              type="submit"
              disabled={!input.trim() || loading || status.status !== 'ready'}
              className="btn-gold"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 16px',
                margin: '2px 0',
                flexShrink: 0
              }}
            >
              <span>Consult</span>
              <Send size={14} />
            </button>
          </form>
          {/* <div className="input-footer">
            <span>Enter your question to search the Homilies on Matthew.</span>
            <span>Powered by AI and semantic search technology.</span>
          </div> */}
        </div>
      </div>

      {/* Source Passages Page */}
      {activeSources && (
        <div className="gold-border animate-fade-in inspector-sidebar">
          <div className="source-page-header">
            <div className="source-page-title">
              <BookOpen size={18} />
              <h3>Source Passages</h3>
            </div>
            <button
              onClick={() => setActiveSources(null)}
              className="source-page-close"
              title="Back to chat"
            >
              <span>Back to Chat</span>
              <X size={18} />
            </button>
          </div>

          {/* Excerpt Tabs */}
          <div className="source-tabs">
            {activeSources.map((src, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setActiveSourceIndex(idx);
                  setActiveCitationKey(null);
                }}
                className={`source-tab ${activeSourceIndex === idx ? 'active' : ''}`}
              >
                Excerpt #{idx + 1}
              </button>
            ))}
          </div>

          {/* Inspector Content */}
          <div ref={sourceContentRef} className="source-page-content">
            <div className="source-page-inner">
            {/* Citation Information */}
            <div className="source-meta-card">
              <div style={{ fontWeight: 'bold', color: 'var(--accent-crimson)', fontFamily: 'var(--font-title)', marginBottom: '4px' }}>
                {activeSources[activeSourceIndex]?.homily || 'HOMILY ?'}
              </div>
              <div style={{ color: 'var(--text-secondary)' }}>
                Paragraph Number: <span style={{ fontWeight: 600 }}>{activeSources[activeSourceIndex]?.paragraph_index}</span>
              </div>
            </div>

            {/* Excerpt context View */}
            {(() => {
              const { global, local, raw } = parseSections(activeSources[activeSourceIndex]?.content);
              return (
                <div
                  className={`source-evidence-card ${activeCitationKey === getCitationKey(activeSources[activeSourceIndex]) ? 'highlighted' : ''}`}
                >
                  <div className="source-identity">
                    <span>Homily {normalizeCitationHomily(activeSources[activeSourceIndex]?.homily)}</span>
                    <span>Paragraph {activeSources[activeSourceIndex]?.paragraph_index}</span>
                  </div>

                  <section className="source-section source-section-primary">
                    <h4>Original Passage</h4>
                    <p>{raw || 'Original passage not available.'}</p>
                  </section>

                  <section className="source-section">
                    <h4>AI Summary</h4>
                    <p>{local || 'Summary not available.'}</p>
                  </section>

                  <section className="source-section">
                    <h4>Context</h4>
                    <p>{global || 'Overview not available.'}</p>
                  </section>
                </div>
              );
            })()}
            </div>
          </div>

          {/* Sidebar Footer */}
          <div style={{
            padding: '1rem',
            borderTop: '1px solid var(--border-color)',
            backgroundColor: 'var(--bg-secondary)',
            fontSize: '0.7rem',
            textAlign: 'center',
            color: 'var(--text-muted)'
          }}>
            These passages are selected from the translation of the Homilies.
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
