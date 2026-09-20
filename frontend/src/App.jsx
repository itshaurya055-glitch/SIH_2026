/*
  ASTRO-DARK // MISSION: ASTRO-LIGHT TELEMETRY CONSOLE
  Aerospace Cyber-HUD // Outfit × Inter × JetBrains Mono
*/

import { useEffect, useState, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import './index.css';

/* ─── Constants ──────────────────────────────────────────────── */
const BACKEND = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';
const WS_URL = BACKEND.replace(/^http/, 'ws') + '/ws/telemetry';
const MAX_PTS = 60;

/* ─── Helpers ────────────────────────────────────────────────── */
const fmt = (v, d = 1) => (v != null ? Number(v).toFixed(d) : '--');
const clamp = (v, min, max) => Math.min(Math.max(v, min), max);

function modeColor(mode) {
  switch (mode) {
    case 'NOMINAL': return '#10b981';
    case 'COOL_DOWN': return '#f59e0b';
    case 'HIBERNATION': return '#ef4444';
    case 'HAZARD_BYPASS': return '#a78bfa';
    default: return '#94a3b8';
  }
}

function motorColor(temp) {
  if (temp >= 75) return '#ef4444';
  if (temp >= 55) return '#f59e0b';
  return '#38bdf8';
}

/* ─── Material Symbol Icon Component ────────────────────────── */
function Icon({ name, size = 18, className = '', style = {} }) {
  return (
    <span
      className={`material-symbols-outlined ${className}`}
      style={{
        fontSize: size,
        fontVariationSettings: "'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24",
        verticalAlign: 'middle',
        display: 'inline-block',
        ...style,
      }}
    >
      {name}
    </span>
  );
}

/* ─── Astro Dial Gauge (SVG 36x36 viewBox with drop-shadow) ── */
function AstroDialGauge({ value, label, unit = '%', color = '#38bdf8', glowClass = 'glow-cyan', subLabel }) {
  // r=56, circumference = 2 * PI * 56 = 351.858
  const circ = 351.858;
  const pct = clamp(value ?? 0, 0, 100);
  const offset = circ - (pct / 100) * circ;

  return (
    <div className="flex flex-col items-center relative group">
      <div className="relative flex items-center justify-center" style={{ width: '9rem', height: '9rem', minWidth: '9rem', minHeight: '9rem' }}>
        <svg className="transform -rotate-90" viewBox="0 0 144 144" style={{ width: '100%', height: '100%', display: 'block' }}>
          {/* Background Ticks Track */}
          <circle cx="72" cy="72" r="56" fill="transparent" stroke="#1e293b" strokeWidth="4" />
          {/* Dashed Secondary Ring */}
          <circle cx="72" cy="72" r="63" fill="transparent" stroke={`${color}33`} strokeDasharray="4 6" strokeWidth="1.5" />
          {/* Gauge Value Ring */}
          <circle
            cx="72" cy="72" r="56"
            fill="transparent"
            stroke={color}
            strokeWidth="6.5"
            strokeDasharray="351.858"
            strokeDashoffset={offset}
            strokeLinecap="round"
            className={`gauge-ring-transition ${glowClass}`}
          />
        </svg>
        {/* Readout */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
          <div className="font-headline font-bold text-3xl tracking-tight text-white flex items-baseline justify-center">
            <span>{fmt(value, 0)}</span>
            <span className="text-sm font-mono ml-0.5" style={{ color }}>{unit}</span>
          </div>
          <div className="flex items-center justify-center gap-1 mt-0.5">
            <span className="w-1 h-1 rounded-full" style={{ backgroundColor: color }} />
            <span className="font-mono text-[10px] tracking-[0.2em] font-semibold text-slate-300 uppercase">{label}</span>
          </div>
          {subLabel && <span className="text-[9px] font-mono text-slate-400 mt-0.5">{subLabel}</span>}
        </div>
      </div>
    </div>
  );
}

/* ─── Metric Tile (Cyan/Purple HUD Glass) ────────────────────── */
function MetricTile({ icon, label, value, accent = '#38bdf8', subText, bar }) {
  return (
    <div className="tech-card p-3.5 rounded-lg border-l-2" style={{ borderLeftColor: accent }}>
      <div className="flex items-center justify-between mb-2">
        <div className="w-7 h-7 rounded bg-slate-900/80 border border-slate-700/50 flex items-center justify-center" style={{ color: accent }}>
          <Icon name={icon} size={16} />
        </div>
        <span className="font-mono text-[9px] font-bold tracking-[0.18em] text-slate-400 uppercase">{label}</span>
      </div>
      <div className="flex items-baseline justify-between">
        <span className="font-headline font-bold text-2xl text-slate-100 tracking-tight" style={{ textShadow: `0 0 10px ${accent}66` }}>
          {value}
        </span>
        {subText && (
          <span className="font-mono text-[9px] font-semibold px-1.5 py-0.5 rounded border" style={{ color: accent, borderColor: `${accent}40`, backgroundColor: `${accent}15` }}>
            {subText}
          </span>
        )}
      </div>
      {bar != null && (
        <div className="w-full bg-slate-800/80 h-1.5 rounded-full mt-2.5 overflow-hidden border border-slate-700/40">
          <div className="h-full transition-all duration-700" style={{ width: `${clamp(bar, 0, 100)}%`, backgroundColor: accent, boxShadow: `0 0 8px ${accent}` }} />
        </div>
      )}
    </div>
  );
}

/* ─── Motor Temperature Bar ──────────────────────────────────── */
function MotorColumn({ label, temp }) {
  const pct = clamp(((temp - 25) / 75) * 100, 0, 100);
  const color = motorColor(temp);
  return (
    <div className="flex flex-col items-center gap-1.5">
      <span className="font-mono text-[9px] text-slate-400 font-bold">{label}</span>
      <div className="w-6 h-20 bg-slate-900/90 border border-slate-700/60 rounded flex flex-col justify-end overflow-hidden p-0.5">
        <div className="motor-bar w-full rounded-sm" style={{ height: `${pct}%`, backgroundColor: color, boxShadow: `0 0 8px ${color}88` }} />
      </div>
      <span className="font-mono text-[10px] font-bold" style={{ color }}>{fmt(temp, 1)}°C</span>
    </div>
  );
}

/* ─── Terminal Log Line ──────────────────────────────────────── */
function TermLine({ ts, text, highlight }) {
  return (
    <p className={`flex items-baseline gap-2 font-mono text-[11px] ${highlight ? 'text-rose-300 bg-rose-950/40 px-1.5 py-0.5 rounded border border-rose-500/30' : 'text-slate-300'}`}>
      <span className="text-cyan-500/60 text-[9px] flex-shrink-0">{ts}</span>
      <span className="text-cyan-400 font-bold flex-shrink-0">&gt;</span>
      <span className="flex-1">{text}</span>
    </p>
  );
}

/* ─── Radar Visualizer Widget ────────────────────────────────── */
function RadarWidget({ gridPos }) {
  return (
    <section className="tech-card tech-corner rounded-lg overflow-hidden p-4 relative">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800/80 pb-2">
        <div className="flex items-center gap-2">
          <Icon name="public" size={16} className="text-cyan-400" />
          <h3 className="font-mono text-[11px] font-bold tracking-[0.16em] text-slate-200">ORBITAL SECTOR 7 MAP</h3>
        </div>
        <span className="font-mono text-[9px] text-cyan-400/80 tracking-wider">
          ROVER: [{gridPos ? `${gridPos[0]}, ${gridPos[1]}` : '0, 0'}]
        </span>
      </div>
      {/* Interactive Tactical Radar Viewport */}
      <div className="relative h-44 w-full rounded border border-cyan-500/30 bg-[#040814] overflow-hidden flex items-center justify-center">
        {/* Fine Grid and Crosshairs */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(56,189,248,0.08)_1px,transparent_1px),linear-gradient(to_bottom,rgba(56,189,248,0.08)_1px,transparent_1px)] bg-[size:16px_16px]" />
        <div className="absolute w-full h-[1px] bg-cyan-500/20" />
        <div className="absolute h-full w-[1px] bg-cyan-500/20" />
        {/* Concentric Radar Rings */}
        <div className="absolute w-36 h-36 rounded-full border border-cyan-500/20" />
        <div className="absolute w-24 h-24 rounded-full border border-cyan-500/30 border-dashed" />
        <div className="absolute w-12 h-12 rounded-full border border-cyan-500/40" />
        {/* Rotating Radar Beam */}
        <div className="absolute w-44 h-44 pointer-events-none animate-radar">
          <div className="w-1/2 h-1/2 origin-bottom-right" style={{ background: 'conic-gradient(from 180deg at 100% 100%, rgba(56, 189, 248, 0.4) 0deg, transparent 55deg)' }} />
        </div>
        {/* Center Planet / Core Body */}
        <div className="relative z-10 w-9 h-9 rounded-full bg-cyan-950/90 border border-cyan-400 flex items-center justify-center text-cyan-300 shadow-[0_0_15px_rgba(56,189,248,0.6)]">
          <Icon name="public" size={18} />
        </div>
        {/* Orbital Track Node 1 */}
        <div className="absolute top-9 right-14 flex items-center gap-1 z-10">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-300 shadow-[0_0_8px_#38bdf8]" />
          </span>
          <span className="font-mono text-[8px] text-cyan-300/80 tracking-tighter">NODE_A</span>
        </div>
        {/* Orbital Track Node 2 (Target Lock) */}
        <div className="absolute bottom-8 left-12 flex items-center gap-1 z-10">
          <div className="w-3.5 h-3.5 border border-purple-400 border-dashed rounded-full flex items-center justify-center animate-spin" style={{ animationDuration: '6s' }}>
            <span className="w-1.5 h-1.5 bg-purple-400 rounded-full shadow-[0_0_6px_#c084fc]" />
          </div>
          <span className="font-mono text-[8px] text-purple-300/80 tracking-tighter">GATEWAY</span>
        </div>
        {/* Tech Reticle Marks */}
        <span className="absolute top-2 left-2 font-mono text-[8px] text-cyan-500/60">408.2 KM // 108.4°E</span>
        <span className="absolute bottom-2 right-2 font-mono text-[8px] text-emerald-400/80">RADAR: LOCK</span>
      </div>
    </section>
  );
}

/* ─── Custom Dark Recharts Tooltip ──────────────────────────── */
function DarkTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#070c18] border border-cyan-500/30 p-2 rounded shadow-lg font-mono text-[10px]">
      <div className="text-cyan-400 font-bold mb-1">TICK {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: <strong>{fmt(p.value)}</strong>
        </div>
      ))}
    </div>
  );
}

/* ─── Chat Message Component ─────────────────────────────────── */
function ChatMsg({ role, text, tools }) {
  const isAI = role === 'ai';
  return (
    <div className={`flex ${isAI ? 'justify-start' : 'justify-end'} mb-2.5`}>
      <div className={`max-w-[88%] p-2.5 font-mono text-[11px] leading-relaxed ${isAI ? 'chat-bubble-ai text-slate-200' : 'chat-bubble-user text-purple-200'}`}>
        {isAI && (
          <div className="font-mono text-[8px] font-bold text-cyan-400 tracking-wider mb-1 flex items-center gap-1">
            <Icon name="psychology" size={12} />
            AI-CORE // MISSION COPILOT
          </div>
        )}
        {tools?.length > 0 && (
          <div className="mb-1.5 flex flex-wrap gap-1">
            {tools.map((t, i) => (
              <span key={i} className="font-mono text-[8px] bg-cyan-950/60 border border-cyan-500/40 text-cyan-300 px-1.5 py-0.5 rounded">
                ⚙ {t}
              </span>
            ))}
          </div>
        )}
        {text}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   MAIN APP COMPONENT
═══════════════════════════════════════════════════════════════ */
export default function App() {
  /* ── State ── */
  const [activeTab, setActiveTab] = useState('telemetry');
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState([]);
  const [latest, setLatest] = useState(null);
  const [anomalyAlert, setAnomalyAlert] = useState(null);
  const [missionLog, setMissionLog] = useState([]);
  const [rnnState, setRnnState] = useState(null);
  const [termLines, setTermLines] = useState([]);
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', text: 'COPILOT ONLINE. Mission: Astro-Light system telemetry active. Query status, anomalies, or path navigation.' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [decisions, setDecisions] = useState([]);
  const [utcClock, setUtcClock] = useState('');

  const wsRef = useRef(null);
  const termRef = useRef(null);
  const chatEndRef = useRef(null);
  const anomalyTimer = useRef(null);
  const reconnectTimer = useRef(null);

  /* ── UTC Clock ── */
  useEffect(() => {
    const tickClock = () => {
      const n = new Date();
      setUtcClock(
        `${String(n.getUTCHours()).padStart(2, '0')}:${String(n.getUTCMinutes()).padStart(2, '0')}:${String(n.getUTCSeconds()).padStart(2, '0')}`
      );
    };
    tickClock();
    const id = setInterval(tickClock, 1000);
    return () => clearInterval(id);
  }, []);

  /* ── WebSocket Connection ── */
  const connect = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      pushTerm('WS_LINK: ASTRO-LIGHT TELEMETRY FEED LINKED', false);
    };

    ws.onmessage = (evt) => {
      const r = JSON.parse(evt.data);
      setLatest(r);

      setHistory(prev => {
        const next = [...prev, r];
        return next.length > MAX_PTS ? next.slice(next.length - MAX_PTS) : next;
      });

      // Terminal Feed
      const slip = fmt(r.wheel_slip_pct, 1);
      const batt = fmt(r.battery_pct, 1);
      const mode = r.mode || 'NOMINAL';
      pushTerm(`OBT_TICK:${String(r.tick).padStart(4, '0')} | BAT:${batt}% | SLIP:${slip}% | MODE:${mode} | SOIL:${r.soil_type || '--'}`, r.anomaly?.is_anomaly);

      // Anomaly Banner
      if (r.anomaly?.is_anomaly) {
        const feat = r.anomaly.top_contributing_features?.[0];
        setAnomalyAlert({
          score: r.anomaly.anomaly_score?.toFixed(3),
          feature: feat ? `${feat.feature} (z=${feat.z_score?.toFixed(2)})` : 'UNSPECIFIED',
          by: r.anomaly.detected_by,
        });
        clearTimeout(anomalyTimer.current);
        anomalyTimer.current = setTimeout(() => setAnomalyAlert(null), 8000);
      }
    };

    ws.onclose = () => {
      setConnected(false);
      pushTerm('WS_LINK: TELEMETRY DISCONNECTED — RETRYING...', true);
      reconnectTimer.current = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      clearTimeout(reconnectTimer.current);
      clearTimeout(anomalyTimer.current);
    };
  }, [connect]);

  /* ── Terminal feed helper ── */
  const pushTerm = (text, highlight = false) => {
    const ts = new Date().toISOString().substr(11, 8);
    setTermLines(prev => {
      const next = [...prev, { ts, text, highlight, id: Date.now() + Math.random() }];
      return next.length > 80 ? next.slice(next.length - 80) : next;
    });
  };

  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight;
  }, [termLines]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  /* ── Poll Mission Log, RNN, Decisions ── */
  useEffect(() => {
    const fetchExtras = async () => {
      try {
        const [logRes, rnnRes, decRes] = await Promise.all([
          fetch(`${BACKEND}/api/mission_log?limit=20`),
          fetch(`${BACKEND}/api/rnn/state`),
          fetch(`${BACKEND}/api/agent/decisions`),
        ]);
        const log = await logRes.json();
        const rnn = await rnnRes.json();
        const dec = await decRes.json();
        setMissionLog(log.entries || []);
        setRnnState(rnn);
        setDecisions(dec.decisions || []);
      } catch { /* backend connecting */ }
    };
    fetchExtras();
    const id = setInterval(fetchExtras, 5000);
    return () => clearInterval(id);
  }, []);

  /* ── Fault Injection & Battery Reset ── */
  const injectFault = async (fault_type, target = 'general', magnitude = 1.5) => {
    try {
      await fetch(`${BACKEND}/api/inject_fault`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fault_type, target, magnitude, duration_ticks: 25 }),
      });
      pushTerm(`FAULT INJECTED: ${fault_type.toUpperCase()} ON ${target.toUpperCase()}`, true);
    } catch { pushTerm('FAULT INJECT FAILED: BACKEND UNAVAILABLE', true); }
  };

  const resetBattery = async () => {
    try {
      await fetch(`${BACKEND}/api/reset_battery`, { method: 'POST' });
      pushTerm('SOLAR RECHARGE INITIATED — BATTERY NOMINAL', false);
    } catch { }
  };

  /* ── AI Copilot Chat ── */
  const sendChat = async () => {
    const msg = chatInput.trim();
    if (!msg || chatLoading) return;
    setChatInput('');
    setChatHistory(prev => [...prev, { role: 'user', text: msg }]);
    setChatLoading(true);
    try {
      const res = await fetch(`${BACKEND}/api/agent/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, history: [] }),
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setChatHistory(prev => [...prev, {
        role: 'ai',
        text: data.response || data.message || 'No response from copilot.',
        tools: data.tools_used || data.tools_called || [],
      }]);
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'ai', text: 'COMMS ERROR — BACKEND UNREACHABLE OR TIMED OUT' }]);
    } finally {
      setChatLoading(false);
    }
  };

  /* ── Derived Telemetry Values ── */
  const motorTemps = latest?.motor_temp || {};
  const fl = motorTemps.fl ?? motorTemps.front_left ?? 35;
  const fr = motorTemps.fr ?? motorTemps.front_right ?? 35;
  const rl = motorTemps.rl ?? motorTemps.rear_left ?? 35;
  const rr = motorTemps.rr ?? motorTemps.rear_right ?? 35;
  const battPct = latest?.battery_pct ?? 78;
  const signalPct = latest?.comms_signal ?? 82;
  const tiltDeg = latest?.tilt_deg ?? 0;
  const slipPct = latest?.wheel_slip_pct ?? 0;
  const roverMode = latest?.mode ?? 'NOMINAL';
  const modeC = modeColor(roverMode);

  /* ── Tabs ── */
  const tabs = [
    { id: 'telemetry', icon: 'query_stats', label: 'Telemetry' },
    { id: 'sensors', icon: 'sensors', label: 'Sensors' },
    { id: 'log', icon: 'summarize', label: 'Mission Log' },
    { id: 'copilot', icon: 'psychology', label: 'AI-Core' },
  ];

  return (
    <div className="font-body overflow-x-hidden select-none min-h-screen flex flex-col">
      <div className="crt-overlay" />
      <div className="hud-sweep" />

      {/* ── TOP AEROSPACE HEADER ──────────────────────────────── */}
      <header className="sticky top-0 w-full z-50 px-4 py-3 bg-[#060a16]/90 backdrop-blur-2xl border-b border-cyan-500/20 shadow-[0_4px_25px_rgba(0,0,0,0.8)]">
        <div className="app-container flex justify-between items-center flex-wrap gap-2">
          {/* Title & Mission ID */}
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-cyan-950/70 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shadow-[0_0_12px_rgba(56,189,248,0.3)]">
              <Icon name="rocket_launch" size={19} />
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                <h1 className="font-headline font-bold text-base tracking-[0.14em] uppercase text-slate-100 flex items-center gap-1.5">
                  MISSION: ASTRO-LIGHT
                </h1>
              </div>
              <div className="flex items-center gap-2 text-[9px] font-mono text-cyan-400/70 tracking-widest pl-3">
                <span>SYS_ID: AL-904</span>
                <span>•</span>
                <span>UTC: {utcClock}</span>
              </div>
            </div>
          </div>

          {/* Nominal Status & WebSocket Connection */}
          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-sm bg-emerald-950/60 border border-emerald-500/40 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              <span className="font-mono text-[10px] font-semibold tracking-widest text-emerald-400 uppercase">
                {roverMode}
              </span>
            </div>

            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-sm border ${connected ? 'bg-cyan-950/60 border-cyan-500/40 text-cyan-300' : 'bg-rose-950/60 border-rose-500/40 text-rose-300'}`}>
              <span className={`w-2 h-2 rounded-full ${connected ? 'bg-cyan-400 animate-pulse' : 'bg-rose-500'}`} />
              <span className="font-mono text-[10px] font-semibold tracking-wider">
                {connected ? 'LIVE TELEMETRY' : 'OFFLINE'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* ── ANOMALY ALERT BANNER ──────────────────────────────── */}
      {anomalyAlert && (
        <div className="app-container mt-3">
          <div className="anomaly-banner p-3 flex justify-between items-center shadow-lg">
            <div>
              <div className="font-mono text-xs font-bold text-rose-400 flex items-center gap-1.5 tracking-wider">
                <Icon name="warning" size={16} />
                ⚠ ANOMALY DETECTED — SCORE {anomalyAlert.score}
              </div>
              <div className="font-mono text-[10px] text-slate-200 mt-0.5">
                {anomalyAlert.feature} via {anomalyAlert.by}
              </div>
            </div>
            <button
              onClick={() => setAnomalyAlert(null)}
              className="font-mono text-[10px] text-slate-400 hover:text-slate-200 bg-slate-900/60 border border-slate-700 px-2 py-1 rounded"
            >
              DISMISS
            </button>
          </div>
        </div>
      )}

      {/* ── MAIN TACTICAL CONSOLE AREA ────────────────────────── */}
      <main className="pt-4 px-3.5 app-container flex-1 space-y-4">

        {/* ════════════════ TELEMETRY TAB ════════════════ */}
        {activeTab === 'telemetry' && (
          <div className="space-y-4">

            {/* Top Grid: Core Power & Signal Gauge Panel + Live Telemetry Feed */}
            <div className="grid-telemetry-top">

              {/* Primary Avionics Panel */}
              <section className="tech-card tech-corner tech-corner-full text-cyan-400 rounded-lg p-5">
                <div className="corner-tr" />
                <div className="corner-bl" />

                {/* Sub-bar Header */}
                <div className="flex items-center justify-between pb-3.5 mb-3 border-b border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <span className="inline-block w-1.5 h-3.5 bg-cyan-400 shadow-[0_0_8px_#38bdf8]" />
                    <h2 className="font-mono text-[11px] font-bold tracking-[0.18em] text-slate-300">CORE POWER &amp; SIGNAL</h2>
                  </div>
                  <div className="flex items-center gap-1.5 font-mono text-[10px] text-cyan-400 bg-cyan-950/50 px-2 py-0.5 rounded border border-cyan-500/30">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                    <span>AUTO-SYNC: ON</span>
                  </div>
                </div>

                {/* Dial Gauges Grid */}
                <div className="grid grid-cols-2 gap-4 py-2">
                  <AstroDialGauge
                    value={battPct}
                    label="BATTERY"
                    unit="%"
                    color="#38bdf8"
                    glowClass="glow-cyan"
                    subLabel={`${fmt(battPct * 0.284, 1)} VDC`}
                  />
                  <AstroDialGauge
                    value={signalPct}
                    label="SIGNAL"
                    unit="%"
                    color="#a78bfa"
                    glowClass="glow-purple"
                    subLabel={`-${fmt(30 + (100 - signalPct) * 0.46, 0)} dBm`}
                  />
                </div>

                {/* Sub Telemetry Counters */}
                <div className="mt-4 pt-3 border-t border-slate-800/80 grid grid-cols-2 gap-3">
                  <div className="flex items-center gap-2 bg-slate-900/70 px-3 py-2 rounded border border-cyan-500/15">
                    <div className="w-6 h-6 rounded bg-cyan-950/80 flex items-center justify-center text-cyan-400">
                      <Icon name="bolt" size={15} />
                    </div>
                    <div>
                      <span className="block text-[8px] font-mono uppercase text-slate-400 tracking-wider">SOLAR BUS</span>
                      <span className="font-mono text-xs font-semibold text-cyan-200">12.4 kW INPUT</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 bg-slate-900/70 px-3 py-2 rounded border border-purple-500/15">
                    <div className="w-6 h-6 rounded bg-purple-950/80 flex items-center justify-center text-purple-400">
                      <Icon name="cell_tower" size={15} />
                    </div>
                    <div>
                      <span className="block text-[8px] font-mono uppercase text-slate-400 tracking-wider">UPLINK LAG</span>
                      <span className="font-mono text-xs font-semibold text-purple-200">4.2ms LATENCY</span>
                    </div>
                  </div>
                </div>
              </section>

              {/* Live Mission Telemetry Feed + Fault Controls */}
              <div className="space-y-3.5 flex flex-col">
                <section className="tech-card rounded-lg overflow-hidden border border-cyan-500/20 shadow-lg flex-1 flex flex-col">
                  <div className="bg-gradient-to-r from-cyan-950/80 via-slate-900/90 to-slate-950 px-3.5 py-2 border-b border-cyan-500/20 flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <span className="w-2 h-2 rounded-sm bg-cyan-400 shadow-[0_0_6px_#38bdf8]" />
                      <span className="font-mono text-[10px] font-bold tracking-[0.16em] text-cyan-300">LIVE TELEMETRY FEED</span>
                    </div>
                    <div className="flex items-center gap-2 font-mono text-[9px] text-slate-400">
                      <span className="text-cyan-400/80">FREQ: 14.2 GHz</span>
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                    </div>
                  </div>

                  <div ref={termRef} className="p-3.5 h-44 overflow-y-auto telemetry-scroll bg-[#040813]/95 font-mono text-[11px] leading-relaxed border-l-2 border-l-cyan-500/60 space-y-1">
                    {termLines.length === 0 && (
                      <p className="text-slate-500 font-mono text-xs">&gt; Awaiting datastream...</p>
                    )}
                    {termLines.map(l => (
                      <TermLine key={l.id} ts={l.ts} text={l.text} highlight={l.highlight} />
                    ))}
                    <div className="mt-1 flex items-center gap-1 text-cyan-400">
                      <span className="text-cyan-500/50 text-[9px]">SYNC</span>
                      <span className="text-slate-400">&gt;</span>
                      <span className="inline-block w-2 h-3.5 bg-cyan-400 animate-pulse shadow-[0_0_6px_#38bdf8]" />
                    </div>
                  </div>
                </section>

                {/* Fault Controls */}
                <section className="tech-card rounded-lg p-3 border border-slate-800/80">
                  <div className="font-mono text-[10px] font-bold tracking-[0.14em] text-slate-400 uppercase mb-2 flex items-center gap-1.5">
                    <Icon name="warning" size={14} className="text-amber-400" />
                    TACTICAL FAULT INJECTION CONTROLS
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { label: 'MOTOR OVERHEAT', icon: 'thermostat', fn: () => injectFault('motor_temp_spike', 'front_left', 2.0) },
                      { label: 'BATTERY DRAIN', icon: 'battery_saver', fn: () => injectFault('battery_drain', 'general', 1.5) },
                      { label: 'COMMS DROPOUT', icon: 'signal_disconnected', fn: () => injectFault('comms_dropout', 'general', 2.0) },
                      { label: 'TILT SPIKE', icon: 'terrain', fn: () => injectFault('tilt_spike', 'general', 1.5) },
                      { label: 'SOLAR RECHARGE', icon: 'battery_charging_full', fn: resetBattery },
                    ].map(({ label, icon, fn }) => (
                      <button key={label} className="fault-btn" onClick={fn}>
                        <Icon name={icon} size={14} />
                        {label}
                      </button>
                    ))}
                  </div>
                </section>
              </div>

            </div>

            {/* Tactical Quad Sensor Matrix & Metrics */}
            <div className="grid-4col">
              <MetricTile
                icon="air"
                label="OXYGEN"
                value="21.4%"
                accent="#38bdf8"
                subText="+0.2% FLOW"
                bar={84}
              />
              <MetricTile
                icon="weight"
                label="GRAVITY"
                value="0.02 G"
                accent="#a78bfa"
                subText="LOW-G MODE"
                bar={12}
              />
              <MetricTile
                icon="radiology"
                label="RADIATION"
                value="SAFE"
                accent="#10b981"
                subText="SHIELD ACTIVE"
                bar={100}
              />
              <MetricTile
                icon="memory"
                label="HARDWARE"
                value="98.2%"
                accent="#7dd3fc"
                subText="EFFICIENCY"
                bar={98}
              />
            </div>

            {/* Radar Visualizer & Charts Grid */}
            <div className="grid-2col">
              <RadarWidget gridPos={latest?.grid_pos} />

              {/* Battery Trend Chart */}
              <section className="tech-card rounded-lg p-4">
                <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
                  <span className="font-mono text-[11px] font-bold text-slate-200 tracking-wider">BATTERY LEVEL TREND</span>
                  <span className="font-mono text-sm font-bold text-cyan-400">{fmt(battPct, 1)}%</span>
                </div>
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="1 4" stroke="rgba(56, 189, 248, 0.1)" />
                      <XAxis dataKey="tick" stroke="#64748b" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9 }} />
                      <YAxis domain={[0, 100]} stroke="#64748b" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9 }} />
                      <Tooltip content={<DarkTooltip />} />
                      <Line type="monotone" dataKey="battery_pct" name="Battery%" stroke="#38bdf8" strokeWidth={2} dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>
            </div>

            {/* Slip & Signal Chart */}
            <section className="tech-card rounded-lg p-4">
              <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
                <span className="font-mono text-[11px] font-bold text-slate-200 tracking-wider">WHEEL SLIP &amp; COMMS SIGNAL TREND</span>
                <span className="font-mono text-xs text-purple-300">SLIP: {fmt(slipPct, 1)}% | SIG: {fmt(signalPct, 0)}%</span>
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="1 4" stroke="rgba(56, 189, 248, 0.1)" />
                    <XAxis dataKey="tick" stroke="#64748b" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9 }} />
                    <YAxis domain={[0, 100]} stroke="#64748b" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: '#94a3b8' }} />
                    <Line type="monotone" dataKey="wheel_slip_pct" name="Slip%" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="stepAfter" dataKey="comms_signal" name="Signal%" stroke="#a78bfa" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>

          </div>
        )}

        {/* ════════════════ SENSORS TAB ════════════════ */}
        {activeTab === 'sensors' && (
          <div className="space-y-4">
            <div className="grid-2col">
              {/* Motor Temperatures */}
              <section className="tech-card rounded-lg p-4">
                <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
                  <h3 className="font-mono text-[11px] font-bold tracking-wider text-slate-200">MOTOR THERMAL DIAGNOSTICS (°C)</h3>
                  <span className="font-mono text-xs text-cyan-400">4 ACTUATORS</span>
                </div>
                <div className="flex justify-around items-flex-end py-3">
                  <MotorColumn label="FL" temp={fl} />
                  <MotorColumn label="FR" temp={fr} />
                  <MotorColumn label="RL" temp={rl} />
                  <MotorColumn label="RR" temp={rr} />
                </div>
              </section>

              {/* Chassis Tilt & RNN */}
              <div className="space-y-4">
                <section className="tech-card rounded-lg p-4">
                  <div className="font-mono text-[10px] font-bold tracking-wider text-slate-400 mb-1 uppercase">CHASSIS TILT ANGLE</div>
                  <div className="font-headline font-bold text-4xl" style={{ color: Math.abs(tiltDeg) > 25 ? '#ef4444' : '#38bdf8' }}>
                    {fmt(tiltDeg, 1)}°
                  </div>
                  <div className="font-mono text-xs text-slate-400 mt-2">
                    PITCH: <span className="text-slate-100">{fmt(latest?.chassis_pitch, 1)}°</span> | ROLL: <span className="text-slate-100">{fmt(latest?.chassis_roll, 1)}°</span>
                  </div>
                </section>

                <section className="tech-card rounded-lg p-4">
                  <div className="font-mono text-[10px] font-bold tracking-wider text-slate-400 mb-1 uppercase">RNN ΔC LEARNED CORRECTION</div>
                  {rnnState ? (
                    <div>
                      <div className="font-headline font-bold text-3xl text-cyan-400 glow-cyan">
                        +{fmt(rnnState.improvement_pct, 1)}%
                      </div>
                      <div className="font-mono text-xs text-slate-400 mt-1">
                        MAE: {fmt(rnnState.before_mae, 3)} → {fmt(rnnState.after_mae, 3)} | STEPS: {rnnState.steps}
                      </div>
                    </div>
                  ) : (
                    <div className="font-mono text-xs text-slate-500">Connecting to RNN telemetry...</div>
                  )}
                </section>
              </div>
            </div>

            {/* Motor Temp Trend */}
            <section className="tech-card rounded-lg p-4">
              <h3 className="font-mono text-[11px] font-bold tracking-wider text-slate-200 mb-3 border-b border-slate-800 pb-2">MOTOR TEMP TREND</h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={history}>
                    <CartesianGrid strokeDasharray="1 4" stroke="rgba(56, 189, 248, 0.1)" />
                    <XAxis dataKey="tick" stroke="#64748b" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9 }} />
                    <YAxis domain={[20, 90]} stroke="#64748b" tick={{ fontFamily: 'JetBrains Mono', fontSize: 9 }} />
                    <Tooltip content={<DarkTooltip />} />
                    <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: '#94a3b8' }} />
                    <Line type="monotone" dataKey="motor_temp.fl" name="FL" stroke="#38bdf8" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="motor_temp.fr" name="FR" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="motor_temp.rl" name="RL" stroke="#a78bfa" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="motor_temp.rr" name="RR" stroke="#ef4444" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        )}

        {/* ════════════════ MISSION LOG TAB ════════════════ */}
        {activeTab === 'log' && (
          <section className="tech-card rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <h2 className="font-mono text-[11px] font-bold tracking-wider text-slate-200">ANOMALY MISSION LOG</h2>
              <span className="font-mono text-xs text-cyan-400">{missionLog.length} LOGGED EVENTS</span>
            </div>
            {missionLog.length === 0 ? (
              <div className="font-mono text-xs text-slate-500 text-center py-8">
                NO ANOMALIES DETECTED — TELEMETRY STREAM NOMINAL
              </div>
            ) : (
              <div className="grid-2col">
                {missionLog.map((entry, i) => (
                  <div key={i} className="tech-card p-3 border-l-2 border-l-rose-500">
                    <div className="flex justify-between items-center mb-1">
                      <span className="font-mono text-xs font-bold text-rose-400">OBT TICK {entry.tick}</span>
                      <span className="font-mono text-[9px] bg-slate-900 border border-slate-700 px-1.5 py-0.5 text-slate-300">{entry.detected_by}</span>
                    </div>
                    <div className="font-mono text-xs text-slate-200">
                      SCORE: <strong>{entry.anomaly_score?.toFixed(3)}</strong> | MODE: <span className="text-emerald-400">{entry.mode}</span>
                    </div>
                    {entry.top_contributing_features?.slice(0, 2).map((f, j) => (
                      <div key={j} className="font-mono text-[10px] text-rose-300 mt-1">
                        ⚠ {f.feature} (z={f.z_score?.toFixed(2)})
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* ════════════════ AI COPILOT TAB ════════════════ */}
        {activeTab === 'copilot' && (
          <div className="grid-2col">
            {/* Chat Panel */}
            <section className="tech-card rounded-lg p-4 flex flex-col">
              <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
                <div className="flex items-center gap-2">
                  <Icon name="psychology" size={18} className="text-cyan-400" />
                  <h2 className="font-mono text-[11px] font-bold tracking-wider text-slate-200">MISSION COPILOT AI</h2>
                </div>
                <span className={`font-mono text-[10px] ${connected ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {connected ? '● ONLINE' : '● OFFLINE'}
                </span>
              </div>

              {/* Quick Commands */}
              <div className="flex flex-wrap gap-1.5 mb-3">
                {['What is current status?', 'Explain last anomaly', 'Is path safe?', 'Recommend action'].map(q => (
                  <button key={q} onClick={() => setChatInput(q)} className="fault-btn text-[9px] py-1 px-2">
                    {q}
                  </button>
                ))}
              </div>

              {/* Chat Feed */}
              <div className="telemetry-scroll h-80 bg-[#040813] border border-cyan-500/20 p-3 rounded flex flex-col overflow-y-auto mb-3">
                {chatHistory.map((msg, i) => (
                  <ChatMsg key={i} {...msg} />
                ))}
                {chatLoading && (
                  <div className="font-mono text-xs text-cyan-400 py-1">
                    COPILOT THINKING<span className="cursor-blink">...</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Input */}
              <div className="flex">
                <input
                  value={chatInput}
                  onChange={e => setChatInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && sendChat()}
                  placeholder="QUERY COPILOT AI CORE..."
                  className="tbsm-input flex-1"
                />
                <button onClick={sendChat} className="tbsm-send-btn">
                  SEND
                </button>
              </div>
            </section>

            {/* Decisions Log */}
            <section className="tech-card rounded-lg p-4">
              <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
                <h2 className="font-mono text-[11px] font-bold tracking-wider text-slate-200">AUTONOMOUS DECISIONS LOG</h2>
                <span className="font-mono text-xs text-purple-300">{decisions.length} ACTIONS</span>
              </div>
              {decisions.length === 0 ? (
                <div className="font-mono text-xs text-slate-500 text-center py-8">
                  NO AUTONOMOUS ACTIONS REQUIRED — NOMINAL
                </div>
              ) : (
                <div className="space-y-2.5 max-h-96 overflow-y-auto telemetry-scroll">
                  {decisions.slice(-10).reverse().map((d, i) => (
                    <div key={i} className="tech-card p-3 border-l-2 border-l-cyan-400">
                      <div className="flex justify-between items-center mb-1">
                        <span className="font-mono text-xs font-bold text-cyan-300 uppercase">{d.action}</span>
                        <span className="font-mono text-[9px] text-slate-400">TICK {d.tick}</span>
                      </div>
                      {d.reasoning && (
                        <div className="font-mono text-[10px] text-slate-300 italic border-l border-slate-700 pl-2">
                          {d.reasoning}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </section>
          </div>
        )}

      </main>

      {/* ── MILITARY-GRADE DOCKED BOTTOM NAVIGATION ──────────── */}
      <nav className="fixed bottom-0 w-full z-50 bg-[#060a16]/95 backdrop-blur-2xl border-t border-cyan-500/25 shadow-[0_-10px_25px_rgba(0,0,0,0.8)]">
        <div className="max-w-md mx-auto flex justify-around items-center px-3 py-2">
          {tabs.map(t => {
            const active = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`relative flex flex-col items-center justify-center py-1 px-3 transition-colors ${active ? 'text-cyan-400' : 'text-slate-400 hover:text-cyan-300'}`}
              >
                {active && (
                  <div className="absolute -top-2.5 w-8 h-[2px] bg-cyan-400 shadow-[0_0_10px_#38bdf8]" />
                )}
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${active ? 'bg-cyan-950/80 border border-cyan-500/40 shadow-[0_0_12px_rgba(56,189,248,0.3)]' : 'bg-slate-900/60 border border-slate-800'}`}>
                  <Icon name={t.icon} size={20} />
                </div>
                <span className={`font-mono text-[10px] tracking-wider mt-1 font-semibold ${active ? 'text-cyan-300' : 'text-slate-400'}`}>
                  {t.label}
                </span>
              </button>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
