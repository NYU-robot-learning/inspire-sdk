#!/usr/bin/env python3
"""
Web-based terminal interface for RH56 Robot Hand with Wrist
Run with: python wristhand_terminal_web.py
Then open http://localhost:5000 in your browser
"""

from flask import Flask, render_template_string, request, jsonify
import threading
import time
from wrist_hand_sdk import RH56RobotHandWithWrist

app = Flask(__name__)

# Global robot hand instance
hand = None
connected = False
lock = threading.Lock()

# Wrist limits
PITCH_MIN = -22.66
PITCH_MAX = 22.12
YAW_MIN = -25.50
YAW_MAX = 25.50

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RH56 Robot Hand with Wrist Controller</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .container { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-bottom: 20px; }
        h2 { color: #555; margin-bottom: 15px; font-size: 1.2em; border-bottom: 2px solid #4CAF50; padding-bottom: 5px; }
        .form-group { margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
        label { min-width: 120px; font-weight: 500; }
        input[type="text"], input[type="number"] { 
            padding: 8px; border: 1px solid #ddd; border-radius: 4px; 
            font-size: 14px; width: 150px;
        }
        button {
            padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;
            font-size: 14px; font-weight: 500; transition: background 0.2s;
        }
        .btn-primary { background: #4CAF50; color: white; }
        .btn-primary:hover { background: #45a049; }
        .btn-danger { background: #f44336; color: white; }
        .btn-danger:hover { background: #da190b; }
        .btn-secondary { background: #2196F3; color: white; }
        .btn-secondary:hover { background: #0b7dda; }
        .status { padding: 8px 15px; border-radius: 4px; font-weight: 500; display: inline-block; }
        .status-connected { background: #4CAF50; color: white; }
        .status-disconnected { background: #f44336; color: white; }
        .slider-group { margin: 15px 0; }
        .slider-container { display: flex; align-items: center; gap: 15px; margin: 10px 0; }
        .slider-label { min-width: 120px; }
        input[type="range"] { flex: 1; height: 6px; }
        .slider-value { min-width: 80px; text-align: right; font-weight: 500; }
        .button-group { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px; }
        #feedback { 
            background: #f9f9f9; border: 1px solid #ddd; border-radius: 4px; 
            padding: 15px; min-height: 150px; max-height: 300px; overflow-y: auto;
            font-family: 'Courier New', monospace; font-size: 12px; white-space: pre-wrap;
        }
        .finger-row { margin: 8px 0; }
    </style>
</head>
<body>
    <h1>🤖 RH56 Robot Hand with Wrist Controller</h1>
    
    <div class="container">
        <h2>Connection</h2>
        <div class="form-group">
            <label>Port:</label>
            <input type="text" id="port" value="/dev/ttyUSB0">
            <label>Baudrate:</label>
            <input type="text" id="baudrate" value="115200">
            <button class="btn-primary" onclick="toggleConnection()" id="connectBtn">Connect</button>
            <span class="status status-disconnected" id="status">Disconnected</span>
        </div>
    </div>
    
    <div class="container">
        <h2>Speed & Force Control</h2>
        <div class="slider-group">
            <div class="slider-container">
                <span class="slider-label">Speed:</span>
                <input type="range" id="speed" min="0" max="100" value="50" oninput="updateSpeed(this.value)">
                <span class="slider-value" id="speedValue">50%</span>
            </div>
            <div class="slider-container">
                <span class="slider-label">Force:</span>
                <input type="range" id="force" min="0" max="100" value="50" oninput="updateForce(this.value)">
                <span class="slider-value" id="forceValue">50%</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <h2>Wrist Control</h2>
        <div class="slider-group">
            <div class="slider-container">
                <span class="slider-label">Pitch:</span>
                <input type="range" id="pitch" min="{{ pitch_min }}" max="{{ pitch_max }}" step="0.1" value="0" 
                       oninput="updatePitch(this.value)">
                <span class="slider-value" id="pitchValue">0.0°</span>
            </div>
            <div class="slider-container">
                <span class="slider-label">Yaw:</span>
                <input type="range" id="yaw" min="{{ yaw_min }}" max="{{ yaw_max }}" step="0.1" value="0" 
                       oninput="updateYaw(this.value)">
                <span class="slider-value" id="yawValue">0.0°</span>
            </div>
            <div class="form-group">
                <label>Movement Time (ms):</label>
                <input type="number" id="movementTime" value="1000" min="100" max="5000" step="100">
                <button class="btn-secondary" onclick="centerWrist()">Center Wrist</button>
                <button class="btn-secondary" onclick="readWrist()">Read Wrist</button>
            </div>
        </div>
    </div>
    
    <div class="container">
        <h2>Finger Control (0=Open, 100=Closed)</h2>
        <div id="fingers">
            {% for i in range(6) %}
            <div class="finger-row slider-container">
                <span class="slider-label">Finger {{ i+1 }}:</span>
                <input type="range" id="finger{{ i }}" min="0" max="100" value="0" 
                       oninput="updateFinger({{ i }}, this.value)">
                <span class="slider-value" id="finger{{ i }}Value">0%</span>
            </div>
            {% endfor %}
        </div>
    </div>
    
    <div class="container">
        <h2>Quick Actions</h2>
        <div class="button-group">
            <button class="btn-secondary" onclick="openAll()">Open All</button>
            <button class="btn-secondary" onclick="closeAll()">Close All</button>
            <button class="btn-secondary" onclick="halfClose()">Half Close</button>
            <button class="btn-secondary" onclick="resetSliders()">Reset</button>
        </div>
    </div>
    
    <div class="container">
        <h2>Sensor Feedback</h2>
        <div class="button-group">
            <button class="btn-secondary" onclick="readAngles()">Read Angles</button>
            <button class="btn-secondary" onclick="readForces()">Read Forces</button>
            <button class="btn-secondary" onclick="readTemps()">Read Temps</button>
            <button class="btn-secondary" onclick="readWrist()">Read Wrist</button>
        </div>
        <div id="feedback"></div>
    </div>
    
    <script>
        let connected = false;
        
        function log(message) {
            const feedback = document.getElementById('feedback');
            const time = new Date().toLocaleTimeString();
            feedback.textContent += `[${time}] ${message}\\n`;
            feedback.scrollTop = feedback.scrollHeight;
        }
        
        async function apiCall(endpoint, data = {}) {
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                if (result.error) {
                    log(`Error: ${result.error}`);
                } else if (result.message) {
                    log(result.message);
                }
                return result;
            } catch (error) {
                log(`Error: ${error.message}`);
                return { error: error.message };
            }
        }
        
        async function toggleConnection() {
            if (connected) {
                const result = await apiCall('/disconnect');
                if (!result.error) {
                    connected = false;
                    document.getElementById('connectBtn').textContent = 'Connect';
                    document.getElementById('connectBtn').className = 'btn-primary';
                    document.getElementById('status').textContent = 'Disconnected';
                    document.getElementById('status').className = 'status status-disconnected';
                }
            } else {
                const port = document.getElementById('port').value;
                const baudrate = document.getElementById('baudrate').value;
                const result = await apiCall('/connect', { port, baudrate });
                if (!result.error) {
                    connected = true;
                    document.getElementById('connectBtn').textContent = 'Disconnect';
                    document.getElementById('connectBtn').className = 'btn-danger';
                    document.getElementById('status').textContent = 'Connected';
                    document.getElementById('status').className = 'status status-connected';
                    if (result.angles) {
                        result.angles.forEach((angle, i) => {
                            document.getElementById(`finger${i}`).value = angle;
                            document.getElementById(`finger${i}Value`).textContent = angle.toFixed(1) + '%';
                        });
                    }
                    if (result.wrist) {
                        document.getElementById('pitch').value = result.wrist.pitch;
                        document.getElementById('pitchValue').textContent = result.wrist.pitch.toFixed(1) + '°';
                        document.getElementById('yaw').value = result.wrist.yaw;
                        document.getElementById('yawValue').textContent = result.wrist.yaw.toFixed(1) + '°';
                    }
                }
            }
        }
        
        function updateSpeed(value) {
            document.getElementById('speedValue').textContent = value + '%';
            if (connected) apiCall('/set_speed', { speed: parseInt(value) });
        }
        
        function updateForce(value) {
            document.getElementById('forceValue').textContent = value + '%';
            if (connected) apiCall('/set_force', { force: parseInt(value) });
        }
        
        function updatePitch(value) {
            document.getElementById('pitchValue').textContent = parseFloat(value).toFixed(1) + '°';
            if (connected) {
                const time = parseInt(document.getElementById('movementTime').value);
                apiCall('/set_pitch', { pitch: parseFloat(value), time });
            }
        }
        
        function updateYaw(value) {
            document.getElementById('yawValue').textContent = parseFloat(value).toFixed(1) + '°';
            if (connected) {
                const time = parseInt(document.getElementById('movementTime').value);
                apiCall('/set_yaw', { yaw: parseFloat(value), time });
            }
        }
        
        function updateFinger(index, value) {
            document.getElementById(`finger${index}Value`).textContent = value + '%';
            if (connected) apiCall('/set_finger', { index: parseInt(index), angle: parseInt(value) });
        }
        
        function centerWrist() {
            if (!connected) { log('Please connect first'); return; }
            const time = parseInt(document.getElementById('movementTime').value);
            apiCall('/center_wrist', { time }).then(() => {
                document.getElementById('pitch').value = 0;
                document.getElementById('pitchValue').textContent = '0.0°';
                document.getElementById('yaw').value = 0;
                document.getElementById('yawValue').textContent = '0.0°';
            });
        }
        
        function openAll() {
            if (!connected) { log('Please connect first'); return; }
            for (let i = 0; i < 6; i++) {
                document.getElementById(`finger${i}`).value = 0;
                document.getElementById(`finger${i}Value`).textContent = '0%';
            }
            apiCall('/open_all');
        }
        
        function closeAll() {
            if (!connected) { log('Please connect first'); return; }
            for (let i = 0; i < 6; i++) {
                document.getElementById(`finger${i}`).value = 100;
                document.getElementById(`finger${i}Value`).textContent = '100%';
            }
            apiCall('/close_all');
        }
        
        function halfClose() {
            if (!connected) { log('Please connect first'); return; }
            for (let i = 0; i < 6; i++) {
                document.getElementById(`finger${i}`).value = 50;
                document.getElementById(`finger${i}Value`).textContent = '50%';
            }
            apiCall('/half_close');
        }
        
        function resetSliders() {
            for (let i = 0; i < 6; i++) {
                document.getElementById(`finger${i}`).value = 0;
                document.getElementById(`finger${i}Value`).textContent = '0%';
            }
            document.getElementById('speed').value = 50;
            document.getElementById('speedValue').textContent = '50%';
            document.getElementById('force').value = 50;
            document.getElementById('forceValue').textContent = '50%';
            document.getElementById('pitch').value = 0;
            document.getElementById('pitchValue').textContent = '0.0°';
            document.getElementById('yaw').value = 0;
            document.getElementById('yawValue').textContent = '0.0°';
            log('Sliders reset');
        }
        
        async function readAngles() {
            if (!connected) { log('Please connect first'); return; }
            const result = await apiCall('/read_angles');
        }
        
        async function readForces() {
            if (!connected) { log('Please connect first'); return; }
            const result = await apiCall('/read_forces');
        }
        
        async function readTemps() {
            if (!connected) { log('Please connect first'); return; }
            const result = await apiCall('/read_temps');
        }
        
        async function readWrist() {
            if (!connected) { log('Please connect first'); return; }
            const result = await apiCall('/read_wrist');
            if (result && result.wrist) {
                document.getElementById('pitch').value = result.wrist.pitch;
                document.getElementById('pitchValue').textContent = result.wrist.pitch.toFixed(1) + '°';
                document.getElementById('yaw').value = result.wrist.yaw;
                document.getElementById('yawValue').textContent = result.wrist.yaw.toFixed(1) + '°';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, 
                                 pitch_min=PITCH_MIN, pitch_max=PITCH_MAX,
                                 yaw_min=YAW_MIN, yaw_max=YAW_MAX)

@app.route('/connect', methods=['POST'])
def connect():
    global hand, connected
    data = request.json
    port = data.get('port', '/dev/ttyUSB0')
    baudrate = int(data.get('baudrate', 115200))
    
    with lock:
        try:
            hand = RH56RobotHandWithWrist(port=port, baudrate=baudrate, hand_id=1)
            if hand.connect():
                hand.set_speeds([50] * 6)
                hand.set_forces([50] * 6)
                
                angles = hand.get_angles()
                wrist_angles = hand.get_wrist_angles()
                
                connected = True
                result = {'message': 'Connected successfully!'}
                if angles:
                    result['angles'] = angles
                if wrist_angles:
                    result['wrist'] = {'pitch': wrist_angles[0], 'yaw': wrist_angles[1]}
                return jsonify(result)
            else:
                return jsonify({'error': 'Failed to connect'}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 400

@app.route('/disconnect', methods=['POST'])
def disconnect():
    global hand, connected
    with lock:
        if hand:
            hand.disconnect()
            hand = None
        connected = False
        return jsonify({'message': 'Disconnected'})

@app.route('/set_speed', methods=['POST'])
def set_speed():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    speed = request.json.get('speed', 50)
    with lock:
        hand.set_speeds([speed] * 6)
    return jsonify({'message': f'Speed set to {speed}%'})

@app.route('/set_force', methods=['POST'])
def set_force():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    force = request.json.get('force', 50)
    with lock:
        hand.set_forces([force] * 6)
    return jsonify({'message': f'Force set to {force}%'})

@app.route('/set_pitch', methods=['POST'])
def set_pitch():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    pitch = float(request.json.get('pitch', 0))
    time_ms = int(request.json.get('time', 1000))
    with lock:
        hand.set_wrist_pitch(pitch, movement_time_ms=time_ms)
    return jsonify({'message': f'Pitch set to {pitch:.1f}°'})

@app.route('/set_yaw', methods=['POST'])
def set_yaw():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    yaw = float(request.json.get('yaw', 0))
    time_ms = int(request.json.get('time', 1000))
    with lock:
        hand.set_wrist_yaw(yaw, movement_time_ms=time_ms)
    return jsonify({'message': f'Yaw set to {yaw:.1f}°'})

@app.route('/set_finger', methods=['POST'])
def set_finger():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    index = int(request.json.get('index', 0))
    angle = int(request.json.get('angle', 0))
    angles = [-1] * 6
    angles[index] = angle
    with lock:
        hand.set_angles(angles)
    return jsonify({'message': f'Finger {index+1} set to {angle}%'})

@app.route('/center_wrist', methods=['POST'])
def center_wrist():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    time_ms = int(request.json.get('time', 1000))
    with lock:
        hand.center_wrist(movement_time_ms=time_ms)
    return jsonify({'message': 'Wrist centered'})

@app.route('/open_all', methods=['POST'])
def open_all():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    with lock:
        hand.set_angles([0] * 6)
    return jsonify({'message': 'Opening all fingers...'})

@app.route('/close_all', methods=['POST'])
def close_all():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    with lock:
        hand.set_angles([100] * 6)
    return jsonify({'message': 'Closing all fingers...'})

@app.route('/half_close', methods=['POST'])
def half_close():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    with lock:
        hand.set_angles([50] * 6)
    return jsonify({'message': 'Half closing...'})

@app.route('/read_angles', methods=['POST'])
def read_angles():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    with lock:
        angles = hand.get_angles()
    if angles:
        text = "Angles: " + ", ".join([f"F{i+1}:{a:.1f}%" for i, a in enumerate(angles)])
        return jsonify({'message': text})
    return jsonify({'error': 'Failed to read angles'}), 400

@app.route('/read_forces', methods=['POST'])
def read_forces():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    with lock:
        forces = hand.get_forces()
    if forces:
        text = "Forces: " + ", ".join([f"F{i+1}:{f:.1f}%" for i, f in enumerate(forces)])
        return jsonify({'message': text})
    return jsonify({'error': 'Failed to read forces'}), 400

@app.route('/read_temps', methods=['POST'])
def read_temps():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    with lock:
        temps = hand.get_temperatures()
    if temps:
        text = "Temps: " + ", ".join([f"F{i+1}:{t}°C" for i, t in enumerate(temps)])
        return jsonify({'message': text})
    return jsonify({'error': 'Failed to read temps'}), 400

@app.route('/read_wrist', methods=['POST'])
def read_wrist():
    global hand, connected
    if not connected or not hand:
        return jsonify({'error': 'Not connected'}), 400
    with lock:
        wrist_state = hand.get_wrist_state()
    if wrist_state:
        text = (f"Wrist - Pitch: {wrist_state.pitch_angle:.2f}°, "
               f"Yaw: {wrist_state.yaw_angle:.2f}°, "
               f"Current1: {wrist_state.current1}, Current2: {wrist_state.current2}, "
               f"Temp1: {wrist_state.temperature1}°C, Temp2: {wrist_state.temperature2}°C, "
               f"Error1: {wrist_state.error_code1}, Error2: {wrist_state.error_code2}")
        return jsonify({
            'message': text,
            'wrist': {
                'pitch': wrist_state.pitch_angle,
                'yaw': wrist_state.yaw_angle
            }
        })
    # Fallback to just angles
    with lock:
        angles = hand.get_wrist_angles()
    if angles:
        text = f"Wrist Angles - Pitch: {angles[0]:.2f}°, Yaw: {angles[1]:.2f}°"
        return jsonify({
            'message': text,
            'wrist': {'pitch': angles[0], 'yaw': angles[1]}
        })
    return jsonify({'error': 'Failed to read wrist state'}), 400

if __name__ == '__main__':
    print("Starting web server...")
    print("Open http://localhost:5000 in your browser")
    print("Press Ctrl+C to stop")
    app.run(host='0.0.0.0', port=5000, debug=False)
