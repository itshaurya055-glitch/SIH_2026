import { useEffect, useState, useRef } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { Activity, Battery, Thermometer, Wifi, AlertTriangle, Zap, ArrowDown } from 'lucide-react';
import './index.css';

const MAX_POINTS = 60;

function App() {
  const [data, setData] = useState([]);
  const [connected, setConnected] = useState(false);
  const [activeFaults, setActiveFaults] = useState(0);
  
  const ws = useRef(null);

  useEffect(() => {
    const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
    const wsUrl = backendUrl.replace(/^http/, 'ws') + '/ws/telemetry';
    
    const connect = () => {
      ws.current = new WebSocket(wsUrl);
      
      ws.current.onopen = () => setConnected(true);
      
      ws.current.onmessage = (event) => {
        const reading = JSON.parse(event.data);
        setActiveFaults(reading.active_fault_count || 0);
        
        setData(prev => {
          const newData = [...prev, reading];
          if (newData.length > MAX_POINTS) {
            return newData.slice(newData.length - MAX_POINTS);
          }
          return newData;
        });
      };

      ws.current.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000); // Reconnect loop
      };
    };
    
    connect();
    
    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  const injectFault = async (faultType, target = "general", magnitude = 1.0) => {
    try {
      const backendUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
      await fetch(`${backendUrl}/api/inject_fault`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fault_type: faultType,
          target,
          magnitude,
          duration_ticks: 20
        })
      });
    } catch (err) {
      console.error("Failed to inject fault:", err);
    }
  };

  const latest = data.length > 0 ? data[data.length - 1] : null;

  return (
    <div className="dashboard-container">
      <header>
        <h1>
          <Activity size={28} color="var(--accent-color)" />
          Mission Copilot Telemetry
        </h1>
        <div className="status-indicator">
          <div className={`status-dot ${connected ? 'connected' : ''}`}></div>
          {connected ? 'Live Stream Active' : 'Connecting to Backend...'}
        </div>
      </header>

      {activeFaults > 0 && (
        <div className="fault-banner">
          <AlertTriangle size={24} />
          <span>Warning: {activeFaults} active anomaly detected in telemetry stream.</span>
        </div>
      )}

      <div className="grid">
        {/* Controls Panel */}
        <div className="panel grid-wide">
          <div className="panel-header">
            <h2 className="panel-title">Fault Injection Demo Controls</h2>
          </div>
          <div className="controls-grid">
            <button className="btn btn-warning" onClick={() => injectFault('motor_temp_spike', 'front_left', 2.0)}>
              <Thermometer size={18} /> Spike Front Left Motor Temp
            </button>
            <button className="btn btn-danger" onClick={() => injectFault('battery_drain', 'general', 1.5)}>
              <Battery size={18} /> Simulate Battery Drain
            </button>
            <button className="btn btn-warning" onClick={() => injectFault('comms_dropout', 'general', 2.0)}>
              <Wifi size={18} /> Disrupt Communications
            </button>
            <button className="btn btn-accent" onClick={() => injectFault('tilt_spike', 'general', 1.5)}>
              <ArrowDown size={18} /> Simulate Terrain Drop
            </button>
          </div>
        </div>

        {/* Battery Chart */}
        <div className="panel">
          <div className="panel-header">
            <h2 className="panel-title"><Battery size={18} style={{display:'inline', marginRight:'8px'}}/> Battery Level</h2>
            <div className="panel-value">{latest ? latest.battery_pct.toFixed(1) : '--'}%</div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis dataKey="tick" stroke="var(--text-muted)" />
                <YAxis domain={[0, 100]} stroke="var(--text-muted)" />
                <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--border-color)' }} />
                <Line type="monotone" dataKey="battery_pct" stroke="var(--success-color)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Motors Chart */}
        <div className="panel">
          <div className="panel-header">
            <h2 className="panel-title"><Zap size={18} style={{display:'inline', marginRight:'8px'}}/> Motor Temperatures</h2>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis dataKey="tick" stroke="var(--text-muted)" />
                <YAxis domain={[20, 80]} stroke="var(--text-muted)" />
                <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--border-color)' }} />
                <Legend />
                <Line type="monotone" dataKey="motor_temp.front_left" name="FL" stroke="#ef4444" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="motor_temp.front_right" name="FR" stroke="#f97316" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="motor_temp.rear_left" name="RL" stroke="#eab308" strokeWidth={2} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="motor_temp.rear_right" name="RR" stroke="#3b82f6" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Comms Chart */}
        <div className="panel">
          <div className="panel-header">
            <h2 className="panel-title"><Wifi size={18} style={{display:'inline', marginRight:'8px'}}/> Comms Signal</h2>
            <div className="panel-value">{latest ? latest.comms_signal.toFixed(1) : '--'}%</div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis dataKey="tick" stroke="var(--text-muted)" />
                <YAxis domain={[0, 100]} stroke="var(--text-muted)" />
                <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--border-color)' }} />
                <Line type="stepAfter" dataKey="comms_signal" stroke="var(--accent-color)" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Tilt Chart */}
        <div className="panel">
          <div className="panel-header">
            <h2 className="panel-title"><ArrowDown size={18} style={{display:'inline', marginRight:'8px'}}/> Tilt (Pitch/Roll)</h2>
            <div className="panel-value">{latest ? latest.tilt_deg.toFixed(1) : '--'}°</div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                <XAxis dataKey="tick" stroke="var(--text-muted)" />
                <YAxis domain={[-45, 45]} stroke="var(--text-muted)" />
                <Tooltip contentStyle={{ backgroundColor: 'var(--panel-bg)', borderColor: 'var(--border-color)' }} />
                <Line type="monotone" dataKey="tilt_deg" stroke="#a855f7" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}

export default App;
