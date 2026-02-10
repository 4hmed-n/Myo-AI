import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart } from 'recharts';

function App() {
  // --- STATE ---
  const [formData, setFormData] = useState({
    age: 45, sex: 1, cp: 2, trestbps: 130, chol: 200,
    fbs: 0, restecg: 0, thalach: 150, exang: 0,
    oldpeak: 1.0, slope: 1, ca: 0, thal: 3
  });
  
  const [simulationData, setSimulationData] = useState([]);
  const [currentRisk, setCurrentRisk] = useState(0);
  const [yearsAhead, setYearsAhead] = useState(0); // The Time Travel Slider

  // --- LOGIC ---
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: parseFloat(e.target.value) });
  };

  // The "Chronos" Engine: Fetches 20-year projection
  const runSimulation = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setSimulationData(data);
      
      // Set initial risk (Year 0)
      if(data.length > 0) setCurrentRisk(data[0].probability);
    } catch (error) {
      console.error("Backend offline?", error);
    }
  };

  // Auto-run simulation when data changes (Live Update)
  useEffect(() => {
    const timer = setTimeout(() => { runSimulation(); }, 500); // Debounce
    return () => clearTimeout(timer);
  }, [formData]);

  // Update displayed risk when Time Travel slider moves
  useEffect(() => {
    if (simulationData[yearsAhead]) {
      setCurrentRisk(simulationData[yearsAhead].probability);
    }
  }, [yearsAhead, simulationData]);

  // --- STYLES (Dark Mode) ---
  const styles = {
    container: { background: "#1e1e1e", color: "#e0e0e0", minHeight: "100vh", padding: "20px", fontFamily: "Consolas, monospace" },
    header: { borderBottom: "1px solid #333", paddingBottom: "10px", marginBottom: "20px" },
    grid: { display: "grid", gridTemplateColumns: "350px 1fr", gap: "20px" },
    panel: { background: "#252526", padding: "20px", borderRadius: "8px", border: "1px solid #333" },
    label: { display: "flex", justifyContent: "space-between", marginBottom: "5px", fontSize: "12px", color: "#aaa" },
    input: { width: "100%", background: "#333", border: "none", height: "5px", borderRadius: "5px", outline: "none", cursor: "pointer" },
    valueBox: { color: "#4ec9b0", fontWeight: "bold" },
    riskGauge: { textAlign: "center", marginBottom: "30px" },
    riskValue: { fontSize: "48px", fontWeight: "bold", color: currentRisk > 0.5 ? "#f14c4c" : "#4ec9b0" },
    riskLabel: { fontSize: "14px", textTransform: "uppercase", letterSpacing: "2px", color: currentRisk > 0.5 ? "#f14c4c" : "#4ec9b0" }
  };

  // --- RENDER ---
  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h2>██ Myo-Sim Bio-Deck — Chronos Time-Travel Interface</h2>
      </div>

      <div style={styles.grid}>
        
        {/* LEFT COLUMN: CONTROLS */}
        <div style={styles.panel}>
          <h3 style={{color: "#4ec9b0", marginTop: 0}}>Patient Vitals</h3>
          
          <Control label="Age" name="age" val={formData.age} min={20} max={80} onChange={handleChange} />
          <Control label="Systolic BP" name="trestbps" val={formData.trestbps} min={90} max={200} onChange={handleChange} />
          <Control label="Cholesterol" name="chol" val={formData.chol} min={120} max={400} onChange={handleChange} />
          <Control label="Max Heart Rate" name="thalach" val={formData.thalach} min={60} max={220} onChange={handleChange} />
          <Control label="ST Depression" name="oldpeak" val={formData.oldpeak} min={0} max={5} step={0.1} onChange={handleChange} />
          
          <h3 style={{color: "#ce9178"}}>Categorical</h3>
          <div style={{display: 'flex', gap: '10px', marginBottom: '15px'}}>
            <button onClick={() => setFormData({...formData, sex: 1})} style={{...btnStyle, background: formData.sex===1?'#007acc':'#333'}}>Male</button>
            <button onClick={() => setFormData({...formData, sex: 0})} style={{...btnStyle, background: formData.sex===0?'#007acc':'#333'}}>Female</button>
          </div>
          
          <div style={{display: 'flex', gap: '10px'}}>
             <select name="cp" onChange={handleChange} style={inputStyle}>
                <option value="0">Typical Angina</option>
                <option value="1">Atypical Angina</option>
                <option value="2">Non-anginal Pain</option>
                <option value="3">Asymptomatic</option>
             </select>
          </div>

          <h3 style={{color: "#dcdcaa", marginTop: "30px"}}>⏳ Chronos Engine</h3>
          <div style={styles.label}>
            <span>Years Ahead: +{yearsAhead}</span>
            <span style={styles.valueBox}>{formData.age + yearsAhead} yrs old</span>
          </div>
          <input 
            type="range" min="0" max="20" value={yearsAhead} 
            onChange={(e) => setYearsAhead(parseInt(e.target.value))} 
            style={{...styles.input, height: "10px", background: "#dcdcaa"}} 
          />
        </div>

        {/* RIGHT COLUMN: VISUALS */}
        <div style={styles.panel}>
          
          {/* TOP: RISK GAUGE */}
          <div style={styles.riskGauge}>
             <div style={styles.riskLabel}>CVD Probability</div>
             <div style={styles.riskValue}>{(currentRisk * 100).toFixed(1)}%</div>
             <div style={{color: "#aaa"}}>Simulated Status: {currentRisk > 0.5 ? "HIGH RISK" : "LOW RISK"}</div>
          </div>

          {/* BOTTOM: CHRONOS CHART */}
          <div style={{height: "300px", marginTop: "40px"}}>
             <h4 style={{textAlign: 'center', color: '#aaa'}}>Chronos Engine: 20-Year Risk Projection</h4>
             <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={simulationData}>
                  <defs>
                    <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f14c4c" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#f14c4c" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="age" stroke="#888" />
                  <YAxis domain={[0, 1]} stroke="#888" />
                  <Tooltip contentStyle={{background: '#333', border: 'none'}} />
                  <ReferenceLine y={0.5} stroke="gray" strokeDasharray="3 3" />
                  
                  {/* The Prediction Line */}
                  <Area type="monotone" dataKey="probability" stroke="#f14c4c" fillOpacity={1} fill="url(#colorRisk)" />
                  
                  {/* The Time Travel Dot */}
                  {simulationData[yearsAhead] && (
                     <ReferenceLine x={simulationData[yearsAhead].age} stroke="#dcdcaa" />
                  )}
                </AreaChart>
             </ResponsiveContainer>
          </div>

        </div>
      </div>
    </div>
  );
}

// Helper Component for Sliders
const Control = ({ label, name, val, min, max, onChange, step=1 }) => (
  <div style={{marginBottom: "15px"}}>
    <div style={{display: "flex", justifyContent: "space-between", marginBottom: "5px", fontSize: "12px", color: "#aaa"}}>
      <span>{label}</span>
      <span style={{color: "#4ec9b0", fontWeight: "bold"}}>{val}</span>
    </div>
    <input type="range" name={name} value={val} min={min} max={max} step={step} onChange={onChange} 
           style={{width: "100%", cursor: "pointer"}} />
  </div>
);

const btnStyle = { flex:1, padding: '8px', border:'none', color:'white', cursor:'pointer', borderRadius:'4px' };
const inputStyle = { width:'100%', padding:'8px', background:'#333', border:'1px solid #444', color:'white', borderRadius:'4px' };

export default App;