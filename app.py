# -*- coding: utf-8 -*-
"""
=== FILE: app.py ===
NeuroType Systems - Flask Web Interface for the 6x6 Hybrid Speller (EOG + EEG).
"""

import os
import time
import threading
from flask import Flask, jsonify, request, render_template_string

# Import các mô-đun BCI, Eye Tracker, Word Predictor, Profile Manager
try:
    import bci_processor_svm
    from bci_processor_svm import latest_prediction, start_websocket
except ImportError:
    bci_processor_svm = None
    latest_prediction = {
        "label": "NEUTRAL",
        "confidence": 1.0,
        "blink_detected": False,
        "attention_high": False,
        "eog_raw": 0,
        "eeg_raw": 0,
        "system_mode": "HYBRID",
        "processed": False
    }
    def start_websocket(ip):
         print(f"[Warning] Thư viện bci_processor_svm không tồn tại. Đang mô phỏng kết nối đến ESP32 tại {ip}.")

# === EYE TRACKER REMOVED ===
eye_tracker = None
print("[EyeTracker] Module EyeTracker OpenCV đã được loại bỏ theo yêu cầu.")

try:
    from word_predictor import WordPredictor
    predictor = WordPredictor()
except ImportError:
    class DummyPredictor:
        def predict(self, prefix, top_n=5):
            return ["có", "không", "và", "những", "bạn"]
        def learn_word(self, word): pass
    predictor = DummyPredictor()

try:
    from profile_manager import profile_mgr
except ImportError:
    profile_mgr = None

app = Flask(__name__)

# Cấu hình Bàn phím 6x6
GRID = [
    ['A','B','C','D','E','F'],
    ['G','H','I','J','K','L'],
    ['M','N','O','P','Q','R'],
    ['S','T','U','V','W','X'],
    ['Y','Z','0','1','2','3'],
    ['4','5','6','7','8','9']
]

# Trạng thái văn bản hiện tại của applet
app_state = {
    "text": "",
    "cursor_row": 0,
    "cursor_col": 0,
    "scan_mode": "ROW", # ROW -> COL -> CHOSEN
    "system_mode": "HYBRID",
    "manual_override": False
}

# Khởi động Socket kết nối tĩnh đến ESP32 trong môi trường thật (IP cấu hình mặc định)
try:
    processor_inst = start_websocket("192.168.1.198")
except Exception as e:
    print(f"[App] Lỗi khởi chạy WebSocket: {e}")
    processor_inst = None

# Hàm chuyển đổi Telex tiếng Việt đơn giản (Client side hoặc Server side, ở đây hỗ trợ cả hai)
def telex_convert(s):
    """
    Quy tắc Telex cơ bản:
    aw->ă, aa->â, ee->ê, oo->ô, ow->ơ, uw->ư, dd->đ
    Dấu sắc (s), huyền (f), hỏi (r), ngã (x), nặng (j)
    Ở phiên bản thực tế, chúng ta chuyển đổi trực tiếp trên client để phản hồi tức thì và chính xác nhất.
    """
    # Xử lý Telex phía máy chủ nếu cần
    # (Để đạt trải nghiệm gõ tối đa, cơ chế Telex hoàn hảo được bổ sung trực tiếp trong Javascript đính kèm)
    return s

@app.route("/")
def index():
    inline_html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NEUROTYPE - BCI Speller Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Space+Grotesk:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0a0a0a;
                --panel-bg: #141414;
                --cyan-glow: rgba(0, 255, 235, 0.4);
                --cyan-active: #00ffeb;
                --neon-green: #0fbc14;
                --cell-hover: #222;
                --accent-red: #ff3333;
            }
            body {
                background-color: var(--bg);
                color: #e0e0e0;
                font-family: 'Space Grotesk', sans-serif;
                margin: 0;
                padding: 16px;
                display: flex;
                flex-direction: column;
                min-height: 100vh;
                box-sizing: border-box;
            }
            header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #222;
                padding-bottom: 12px;
                margin-bottom: 16px;
            }
            .title-area h1 {
                margin: 0;
                font-size: 24px;
                letter-spacing: 2px;
                color: var(--cyan-active);
                text-shadow: 0 0 10px var(--cyan-glow);
            }
            .title-area p {
                margin: 4px 0 0 0;
                font-size: 11px;
                font-family: 'JetBrains Mono', monospace;
                color: #888;
            }
            .grid-container {
                display: flex;
                flex: 1;
                gap: 20px;
                flex-wrap: wrap;
            }
            .main-panel {
                flex: 2;
                min-width: 320px;
                background: var(--panel-bg);
                border: 1px solid #2a2a2a;
                border-radius: 12px;
                padding: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .side-panel {
                flex: 1;
                min-width: 250px;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            .status-card {
                background: var(--panel-bg);
                border: 1px solid #2a2a2a;
                border-radius: 10px;
                padding: 12px;
            }
            .status-card h3 {
                margin: 0 0 8px 0;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #ccc;
                border-left: 3px solid var(--cyan-active);
                padding-left: 8px;
            }
            .output-box {
                background: #050505;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 14px;
                font-size: 20px;
                letter-spacing: 1px;
                min-height: 50px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'JetBrains Mono', monospace;
                color: var(--cyan-active);
                text-shadow: 0 0 5x var(--cyan-glow);
            }
            .suggest-bar {
                display: flex;
                gap: 10px;
                margin-top: -6px;
            }
            .suggest-btn {
                background: #1e1e1e;
                border: 1px solid #333;
                border-radius: 6px;
                color: #ccc;
                padding: 6px 14px;
                cursor: pointer;
                font-size: 12px;
                flex: 1;
                text-align: center;
                transition: all 0.2s;
                font-family: 'Space Grotesk', sans-serif;
            }
            .suggest-btn:hover {
                background: var(--cyan-active);
                color: #000;
                font-weight: bold;
                border-color: var(--cyan-active);
                box-shadow: 0 0 8px var(--cyan-glow);
            }
            .keyboard-grid {
                display: grid;
                grid-template-columns: repeat(6, 1fr);
                gap: 8px;
                flex: 1;
            }
            .key-cell {
                background: #1a1a1a;
                border: 1px solid #2c2c2c;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 26px;
                font-weight: 800;
                aspect-ratio: 1;
                transition: all 0.15s ease-in-out;
                user-select: none;
                cursor: pointer;
                font-family: 'Space Grotesk', sans-serif;
            }
            .key-cell.scan-row {
                background: rgba(0, 255, 235, 0.15);
                border-color: rgba(0, 255, 235, 0.5);
                box-shadow: 0 0 5px var(--cyan-glow);
            }
            .key-cell.scan-cell {
                background: var(--cyan-active);
                color: #000 !important;
                border-color: var(--cyan-active);
                box-shadow: 0 0 15px var(--cyan-glow);
                transform: scale(1.05);
                z-index: 2;
            }
            .metrics-table {
                width: 100%;
                font-family: 'JetBrains Mono', monospace;
                font-size: 11px;
                border-collapse: collapse;
            }
            .metrics-table td {
                padding: 6px 4px;
                border-bottom: 1px solid #222;
            }
            .metrics-table td:last-child {
                text-align: right;
                font-weight: bold;
                color: #fff;
            }
            .indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #555;
                margin-right: 6px;
            }
            .indicator.active {
                background: var(--neon-green);
                box-shadow: 0 0 8px var(--neon-green);
                animation: pulse 1.5s infinite alternate;
            }
            .control-panel {
                display: flex;
                gap: 8px;
                margin-top: auto;
            }
            .ctrl-btn {
                background: #252525;
                border: 1px solid #444;
                color: #ccc;
                font-size: 11px;
                padding: 8px;
                border-radius: 6px;
                cursor: pointer;
                flex: 1;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                transition: background 0.2s;
            }
            .ctrl-btn:hover {
                background: #333;
                border-color: #00ffeb;
                color: #fff;
            }
            @keyframes pulse {
                from { transform: scale(1); opacity: 0.7; }
                to { transform: scale(1.2); opacity: 1; }
            }
            /* Trực quan sóng não EEG/EOG */
            .waveform-box {
                margin-top: 10px;
                background: #000;
                height: 60px;
                position: relative;
                border: 1px solid #222;
                border-radius: 6px;
                overflow: hidden;
            }
            .waveform-label {
                position: absolute;
                top: 4px;
                left: 4px;
                font-size: 8px;
                font-family: 'JetBrains Mono', monospace;
                color: #666;
                z-index: 2;
            }
            .plot-canvas {
                width: 100%;
                height: 100%;
            }
        </style>
    </head>
    <body>
        <header>
            <div class="title-area">
                <h1>NEUROTYPE</h1>
                <p>Hệ thống giao tiếp dựa trên tín hiệu điện mắt (EOG) và điện não (EEG)</p>
            </div>
            <div>
                <span class="indicator active"></span>
                <span style="font-size: 11px; font-family:'JetBrains Mono', monospace; color: #888;">CORE STATUS: ONLINE</span>
            </div>
        </header>

        <div class="grid-container">
            <!-- Khu vực gõ và bàn phím tương tác -->
            <div class="main-panel">
                <div class="output-box" id="typedOutput">Đang khởi tạo ứng dụng...</div>
                
                <div class="suggest-bar" id="suggestionsBox">
                    <button class="suggest-btn" onclick="selectSuggestion(0)">--</button>
                    <button class="suggest-btn" onclick="selectSuggestion(1)">--</button>
                    <button class="suggest-btn" onclick="selectSuggestion(2)">--</button>
                    <button class="suggest-btn" onclick="selectSuggestion(3)">--</button>
                    <button class="suggest-btn" onclick="selectSuggestion(4)">--</button>
                </div>

                <div class="keyboard-grid" id="keyboardGrid">
                    <!-- Bàn phím được render động bằng JS -->
                </div>
            </div>

            <!-- Bảng giám sát và các chỉ số BCI -->
            <div class="side-panel">
                <div class="status-card">
                    <h3>Trạng thái hệ thống</h3>
                    <table class="metrics-table">
                        <tr>
                            <td>Chế độ vận hành (Mode)</td>
                            <td id="sysMode" style="color:#00ffeb;">HYBRID</td>
                        </tr>
                        <tr>
                            <td>Bước quay quét hiện tại</td>
                            <td id="scanState" style="color:#ffc107;">DỮ LIỆU...</td>
                        </tr>
                        <tr>
                            <td>Nguồn tín hiệu chính</td>
                            <td>WebSocket ESP32</td>
                        </tr>
                    </table>
                </div>

                <div class="status-card">
                    <h3>Chỉ số Nghiên cứu Khoa học (ITR)</h3>
                    <table class="metrics-table">
                        <tr>
                            <td>Tốc độ gõ chữ (CPM)</td>
                            <td id="statCPM" style="color:#00ffeb;">0 CPM</td>
                        </tr>
                        <tr>
                            <td>Tốc độ từ trung bình (WPM)</td>
                            <td id="statWPM" style="color:#ffc107;">0.0 WPM</td>
                        </tr>
                        <tr>
                            <td>Tốc độ truyền tin (ITR)</td>
                            <td id="statITR" style="color:#00ff00;">0.0 b/m</td>
                        </tr>
                        <tr>
                            <td>Số lần sửa sai (Typos)</td>
                            <td id="statTypos" style="color:#ff3b30;">0</td>
                        </tr>
                    </table>
                </div>

                <div class="status-card">
                    <h3>Phân tách cử động sinh học</h3>
                    <table class="metrics-table">
                        <tr>
                            <td>Lớp phán quyết (Label)</td>
                            <td id="currentPredLabel" style="color:#00f;"><span class="indicator"></span> NEUTRAL</td>
                        </tr>
                        <tr>
                            <td>Chớp mắt (Blink)</td>
                            <td id="blinkActive" style="color:#aaa;">NO</td>
                        </tr>
                        <tr>
                            <td>Độ tập trung (EEG Var)</td>
                            <td id="eegAttention" style="color:#aaa;">LOW</td>
                        </tr>
                    </table>
                </div>

                <div class="status-card" style="flex:1; display:flex; flex-direction:column; gap:6px;">
                    <h3>Hiển thị dữ liệu sườn điện cực</h3>
                    <div class="waveform-box">
                        <span class="waveform-label">KÊNH A1 - ĐIỆN VÕNG MẠC MẮT (EOG)</span>
                        <canvas id="eogCanvas" class="plot-canvas"></canvas>
                    </div>
                    <div class="waveform-box">
                        <span class="waveform-label">KÊNH A0 - ĐIỆN NÃO TRƯỚC (EEG)</span>
                        <canvas id="eegCanvas" class="plot-canvas"></canvas>
                    </div>
                    
                    <div class="control-panel" style="margin-top: 10px;">
                        <button class="ctrl-btn" onclick="triggerManualBlink(1)">Blink Thường</button>
                        <button class="ctrl-btn" onclick="triggerManualBlink(2)">Blink Nháy Đúp</button>
                        <button class="ctrl-btn" onclick="triggerManualMove('LEFT')">Trái (Left)</button>
                        <button class="ctrl-btn" onclick="triggerManualMove('RIGHT')">Phải (Right)</button>
                    </div>
                    
                    <div class="control-panel" style="margin-top: 6px;">
                        <button class="ctrl-btn" style="background:#420000; color:#ff8888;" onclick="resetText()">Reset (Xóa hết)</button>
                        <button class="ctrl-btn" style="background:#5c5c00; color:#ffffaa;" onclick="deleteLastChar()">Xóa Chữ</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Định nghĩa bàn phím 6x6
            const GRID = [
                ['A','B','C','D','E','F'],
                ['G','H','I','J','K','L'],
                ['M','N','O','P','Q','R'],
                ['S','T','U','V','W','X'],
                ['Y','Z','0','1','2','3'],
                ['4','5','6','7','8','9']
            ];

            // Cấu hình quét và bộ nhớ
            let textBuffer = "";
            let currentWord = "";
            let scanRow = 0;
            let scanCol = 0;
            let currentMode = "ROW"; // ROW hoặc COL
            let systemMode = "HYBRID";
            let eyeTrackingState = "DISABLED";
            let lastBlinkTime = 0;
            let lastGazeTime = 0;
            let typosCount = 0;
            let totalBciSelections = 0;

            // Chu kỳ quét tối ưu (EOG + Blink Scanner: quét dòng 1000ms, quét cột 600ms)
            let scanIntervalMs = 1000;
            let activeTimer = null;

            // Lịch sử Buffer vẽ đồ thị dạng sóng
            const maxPoints = 120;
            const eogHistory = Array(maxPoints).fill(0);
            const eegHistory = Array(maxPoints).fill(0);

            // BẢNG ÁNH XẠ TELEX TIẾNG VIỆT HOÀN CHỈNH CHO NHÀ NGHIÊN CỨU
            const CLEAN_VOWELS = {
                'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
                'ắ': 'ă', 'ằ': 'ă', 'ẳ': 'ă', 'ẵ': 'ă', 'ặ': 'ă',
                'ấ': 'â', 'ầ': 'â', 'ẩ': 'â', 'ẫ': 'â', 'ậ': 'â',
                'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
                'ế': 'ê', 'ề': 'ê', 'ể': 'ê', 'ễ': 'ê', 'ệ': 'ê',
                'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
                'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
                'ố': 'ô', 'ồ': 'ô', 'ổ': 'ô', 'ỗ': 'ô', 'ộ': 'ô',
                'ớ': 'ơ', 'ờ': 'ơ', 'ở': 'ơ', 'ỡ': 'ơ', 'ợ': 'ơ',
                'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
                'ứ': 'ư', 'ừ': 'ư', 'ử': 'ư', 'ữ': 'ư', 'ự': 'ư',
                'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
                'Á': 'A', 'À': 'A', 'Ả': 'A', 'Ã': 'A', 'Ạ': 'A',
                'Ắ': 'Ă', 'Ằ': 'Ă', 'Ẳ': 'Ă', 'Ẵ': 'Ă', 'Ặ': 'Ă',
                'Ấ': 'Â', 'Ầ': 'Â', 'Ẩ': 'Â', 'Ẫ': 'Â', 'Ậ': 'Â',
                'É': 'E', 'È': 'E', 'Ẻ': 'E', 'Ẽ': 'E', 'Ẹ': 'E',
                'Ế': 'Ê', 'Ề': 'Ê', 'Ể': 'Ê', 'Ễ': 'Ê', 'Ệ': 'Ê',
                'Í': 'I', 'Ì': 'I', 'Ỉ': 'I', 'Ĩ': 'I', 'Ị': 'I',
                'Ó': 'O', 'Ò': 'O', 'Ỏ': 'O', 'Õ': 'O', 'Ọ': 'O',
                'Ố': 'Ô', 'Ồ': 'Ô', 'Ổ': 'Ô', 'Ỗ': 'Ô', 'Ộ': 'Ô',
                'Ớ': 'Ơ', 'Ờ': 'Ơ', 'Ở': 'Ơ', 'Ỡ': 'Ơ', 'Ợ': 'Ơ',
                'Ú': 'U', 'Ù': 'U', 'Ủ': 'U', 'Ũ': 'U', 'Ụ': 'U',
                'Ứ': 'Ư', 'Ừ': 'Ư', 'Ử': 'Ư', 'Ữ': 'Ư', 'Ự': 'Ư',
                'Ý': 'Y', 'Ỳ': 'Y', 'Ỷ': 'Y', 'Ỹ': 'Y', 'Ỵ': 'Y'
            };

            const TONE_MAPS = {
                's': {
                    'a': 'á', 'ă': 'ắ', 'â': 'ấ', 'e': 'é', 'ê': 'ế', 'i': 'í',
                    'o': 'ó', 'ô': 'ố', 'ơ': 'ớ', 'u': 'ú', 'ư': 'ứ', 'y': 'ý',
                    'A': 'Á', 'Ă': 'Ắ', 'Â': 'Ấ', 'E': 'É', 'Ê': 'Ế', 'I': 'Í',
                    'O': 'Ó', 'Ô': 'Ố', 'Ơ': 'Ớ', 'U': 'Ú', 'Ư': 'Ứ', 'Y': 'Ý'
                },
                'f': {
                    'a': 'à', 'ă': 'ằ', 'â': 'ầ', 'e': 'è', 'ê': 'ề', 'i': 'ì',
                    'o': 'ò', 'ô': 'ồ', 'ơ': 'ờ', 'u': 'ù', 'ư': 'ừ', 'y': 'ỳ',
                    'A': 'À', 'Ă': 'Ằ', 'Â': 'Ầ', 'E': 'È', 'Ê': 'Ề', 'I': 'Ì',
                    'O': 'Ò', 'Ô': 'Ồ', 'Ơ': 'Ờ', 'U': 'Ú', 'Ư': 'Ừ', 'Y': 'Ỳ'
                },
                'r': {
                    'a': 'ả', 'ă': 'ẳ', 'â': 'ẩ', 'e': 'ẻ', 'ê': 'ể', 'i': 'ỉ',
                    'o': 'ỏ', 'ô': 'ổ', 'ơ': 'ở', 'u': 'ủ', 'ư': 'ử', 'y': 'ỷ',
                    'A': 'Ả', 'Ă': 'Ẳ', 'Â': 'Ẩ', 'E': 'Ẻ', 'Ê': 'Ể', 'I': 'Ỉ',
                    'O': 'Ỏ', 'Ô': 'Ổ', 'Ơ': 'Ở', 'U': 'Ủ', 'Ư': 'Ứ', 'Y': 'Ý'
                },
                'x': {
                    'a': 'ã', 'ă': 'ẵ', 'â': 'ẫ', 'e': 'ẽ', 'ê': 'ễ', 'i': 'ĩ',
                    'o': 'õ', 'ô': 'ỗ', 'ơ': 'ỡ', 'u': 'ũ', 'ư': 'ữ', 'y': 'ỹ',
                    'A': 'Ã', 'Ă': 'Ẵ', 'Â': 'Ẫ', 'E': 'Ẽ', 'Ê': 'Ễ', 'I': 'Ĩ',
                    'O': 'Õ', 'Ô': 'Ỗ', 'Ơ': 'Ỡ', 'U': 'Ũ', 'Ư': 'Ữ', 'Y': 'Ỹ'
                },
                'j': {
                    'a': 'ạ', 'ă': 'ặ', 'â': 'ậ', 'e': 'ẹ', 'ê': 'ệ', 'i': 'ị',
                    'o': 'ọ', 'ô': 'ộ', 'ơ': 'ợ', 'u': 'ụ', 'ư': 'ự', 'y': 'ỵ',
                    'A': 'Ạ', 'Ă': 'Ặ', 'Â': 'Ậ', 'E': 'Ẹ', 'Ê': 'Ệ', 'I': 'Ị',
                    'O': 'Ọ', 'Ô': 'Ộ', 'Ơ': 'Ợ', 'U': 'Ụ', 'Ư': 'Ự', 'Y': 'Ỵ'
                }
            };

            function translateWordTelex(word) {
                if (!word) return "";
                let s = word;
                s = s.replace(/dd/gi, (m) => m.toLowerCase() === 'dd' ? 'đ' : 'Đ');
                s = s.replace(/aa/gi, (m) => m.toLowerCase() === 'aa' ? 'â' : 'Â');
                s = s.replace(/ee/gi, (m) => m.toLowerCase() === 'ee' ? 'ê' : 'Ê');
                s = s.replace(/oo/gi, (m) => m.toLowerCase() === 'oo' ? 'ô' : 'Ô');
                s = s.replace(/uw/gi, (m) => m.toLowerCase() === 'uw' ? 'ư' : 'Ư');
                s = s.replace(/ow/gi, (m) => m.toLowerCase() === 'ow' ? 'ơ' : 'Ơ');
                s = s.replace(/aw/gi, (m) => m.toLowerCase() === 'aw' ? 'ă' : 'Ă');

                const lastChar = s[s.length - 1]?.toLowerCase();
                const hasVowels = /[aeiouyâăêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵAEIOUYÂĂÊÔƠƯ]/g.test(s);
                
                if (hasVowels && s.length > 1 && ['s', 'f', 'r', 'x', 'j', 'z'].includes(lastChar)) {
                    s = s.slice(0, -1);
                    const vowelIndices = [];
                    for (let i = 0; i < s.length; i++) {
                        if (/[aeiouyâăêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵAEIOUYÂĂÊÔƠƯ]/.test(s[i])) {
                            vowelIndices.push(i);
                        }
                    }
                    if (vowelIndices.length > 0) {
                        const chars = s.split('');
                        for (const idx of vowelIndices) {
                            const c = chars[idx];
                            if (CLEAN_VOWELS[c]) {
                                chars[idx] = CLEAN_VOWELS[c];
                            }
                        }
                        s = chars.join('');

                        let targetIdx = vowelIndices[0];
                        if (vowelIndices.length === 2) {
                            const v1 = s[vowelIndices[0]].toLowerCase();
                            const v2 = s[vowelIndices[1]].toLowerCase();
                            const hasHat1 = ['â', 'ê', 'ô', 'ă', 'ơ', 'ư'].includes(v1);
                            const hasHat2 = ['â', 'ê', 'ô', 'ă', 'ơ', 'ư'].includes(v2);

                            if (hasHat1 && !hasHat2) targetIdx = vowelIndices[0];
                            else if (hasHat2 && !hasHat1) targetIdx = vowelIndices[1];
                            else if (v1 === 'ư' && v2 === 'ơ') targetIdx = vowelIndices[1];
                            else if (v1 === 'u' && v2 === 'â') targetIdx = vowelIndices[1];
                            else if (v1 === 'u' && v2 === 'ô') targetIdx = vowelIndices[1];
                            else if (v1 === 'i' && v2 === 'ê') targetIdx = vowelIndices[1];
                            else if (v1 === 'y' && v2 === 'ê') targetIdx = vowelIndices[1];
                            else {
                                const isEndingInYOrI = ['i', 'y'].includes(v2);
                                if (isEndingInYOrI && v1 !== 'u') targetIdx = vowelIndices[0];
                                else targetIdx = vowelIndices[1];
                            }
                        } else if (vowelIndices.length === 3) {
                            targetIdx = vowelIndices[1];
                        }

                        if (lastChar !== 'z') {
                            const charsWithTone = s.split('');
                            const baseVowel = charsWithTone[targetIdx];
                            const toned = TONE_MAPS[lastChar]?.[baseVowel];
                            if (toned) {
                                charsWithTone[targetIdx] = toned;
                                s = charsWithTone.join('');
                            }
                        }
                    }
                }
                return s;
            }

            function applyTelex(str) {
                if (!str) return "";
                const words = str.split(" ");
                if (words.length === 0) return "";
                words[words.length - 1] = translateWordTelex(words[words.length - 1]);
                return words.join(" ");
            }

            // Khởi tạo đồ thị
            function createGrid() {
                const gridEl = document.getElementById("keyboardGrid");
                gridEl.innerHTML = "";
                for (let r = 0; r < 6; r++) {
                    for (let c = 0; c < 6; c++) {
                        const cell = document.createElement("div");
                        cell.className = "key-cell";
                        cell.id = `cell-${r}-${c}`;
                        cell.textContent = GRID[r][c];
                        // Cho phép bấm trực tiếp để dự phòng test chuột
                        cell.onclick = () => selectKeyDirectly(r, c);
                        gridEl.appendChild(cell);
                    }
                }
            }

            // Đồng bộ vẽ Layout quét sườn dòng/cột
            function updateScanHighlight() {
                for (let r = 0; r < 6; r++) {
                    for (let c = 0; c < 6; c++) {
                        const cell = document.getElementById(`cell-${r}-${c}`);
                        if (!cell) continue;
                        cell.className = "key-cell"; // xóa class cũ

                        if (currentMode === "ROW") {
                            if (r === scanRow) {
                                cell.classList.add("scan-row");
                            }
                        } else if (currentMode === "COL") {
                            if (r === scanRow) {
                                if (c === scanCol) {
                                    cell.classList.add("scan-cell");
                                } else {
                                    cell.classList.add("scan-row");
                                }
                            }
                        }
                    }
                }
            }

            // Máy quét dòng và cột tự động chạy ngầm
            function autoScanStep() {
                if (currentMode === "ROW") {
                    scanRow = (scanRow + 1) % 6;
                } else if (currentMode === "COL") {
                    scanCol = (scanCol + 1) % 6;
                }
                updateScanHighlight();
                document.getElementById("scanState").textContent = `${currentMode} (Row:${scanRow + 1}, Col:${scanCol + 1})`;
            }

            function startTimer() {
                clearInterval(activeTimer);
                let speed = currentMode === "ROW" ? 1000 : 600;
                activeTimer = setInterval(autoScanStep, speed);
            }

            // Kênh kích hoạt lựa chọn khi chớp mắt (Blink)
            function executeSelection() {
                if (currentMode === "ROW") {
                    // Chuyển sang quét cột của dòng hiện tại
                    currentMode = "COL";
                    scanCol = 0;
                    startTimer();
                    updateScanHighlight();
                } else if (currentMode === "COL") {
                    // Chọn được ký tự cuối
                    const chosenChar = GRID[scanRow][scanCol];
                    textBuffer += chosenChar;
                    
                    // Thực hiện quy tắc telex
                    textBuffer = applyTelex(textBuffer);
                    
                    // Cập nhật lên ô chữ hiển thị và truyền lên máy học
                    updateTypedText();
                    
                    // Chuyển trả về quét dòng bắt đầu
                    currentMode = "ROW";
                    startTimer();
                    updateScanHighlight();
                }
            }

            let startTime = null;

            function updateTypedText() {
                const output = document.getElementById("typedOutput");
                output.textContent = textBuffer || "[Nhìn Trái/Phải để chuyển từ, Chớp mắt để bắt đầu gõ...]";
                
                // Phục vụ Word Predictor
                const lastWordMatch = textBuffer.trim().split(" ");
                currentWord = lastWordMatch[lastWordMatch.length - 1] || "";

                if (textBuffer && startTime === null) {
                    startTime = Date.now();
                }

                // Tính WPM, CPM, ITR
                let cpm = 0;
                let wpm = 0;
                let itr = 0;
                if (startTime !== null && textBuffer.length > 0) {
                    const durationMins = (Date.now() - startTime) / 60000;
                    if (durationMins > 0.008) {
                        cpm = Math.round(textBuffer.length / durationMins);
                        wpm = parseFloat((cpm / 5).toFixed(1));

                        // ITR
                        const totalSteps = textBuffer.length + typosCount;
                        const acc = totalSteps > 0 ? textBuffer.length / totalSteps : 1.0;
                        if (acc > 0.05) {
                            const symbolAlphabetSize = 36;
                            const partAlphabet = Math.log2(symbolAlphabetSize);
                            const partAcc = acc * Math.log2(acc);
                            const partErr = (1 - acc) > 0 ? (1 - acc) * Math.log2((1 - acc) / (symbolAlphabetSize - 1)) : 0;
                            itr = Math.max(0, (partAlphabet + partAcc + partErr) * cpm);
                        }
                    }
                }

                document.getElementById("statCPM").textContent = cpm + " CPM";
                document.getElementById("statWPM").textContent = wpm + " WPM";
                document.getElementById("statITR").textContent = itr.toFixed(1) + " b/m";
                document.getElementById("statTypos").textContent = typosCount;

                // Đồng bộ hóa với Flask Backend
                fetch("/api/update_text", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({text: textBuffer})
                }).catch(() => {});
            }

            // Nháy đúp (Double blink) hoặc phán quyết cử chỉ xóa để hiệu chỉnh sai số
            function executeDeletion() {
                if (textBuffer.length > 0) {
                    textBuffer = textBuffer.substring(0, textBuffer.length - 1);
                    typosCount += 1;
                    updateTypedText();
                }
            }

            // API gọi đồng bộ
            async function fetchState() {
                try {
                    const res = await fetch("/api/state");
                    const data = await res.json();
                    
                    // 1. Cập nhật nhãn phân loại từ SVM
                    const pred = data.prediction;
                    const predLabelEl = document.getElementById("currentPredLabel");
                    const blinkActiveEl = document.getElementById("blinkActive");
                    const eegAttentionEl = document.getElementById("eegAttention");
                    
                    document.getElementById("sysMode").textContent = data.system_mode;
                    
                    // Hiển thị nhãn chính kèm màu cảnh báo
                    let color = "#888";
                    if (pred.label === "LEFT") color = "#00ffff";
                    else if (pred.label === "RIGHT") color = "#ffaa00";
                    else if (pred.label === "BLINK") color = "#00ffeb";
                    
                    predLabelEl.style.color = color;
                    predLabelEl.innerHTML = `<span class="indicator active" style="background:${color}; box-shadow:0 0 8px ${color}"></span> ${pred.label} (${(pred.confidence * 100).toFixed(0)}%)`;
                    
                    // Trạng thái phụ chớp mắt & tập trung
                    blinkActiveEl.style.color = pred.blink_detected ? "#00ff00" : "#aaa";
                    blinkActiveEl.textContent = pred.blink_detected ? "YES - ACTIVATING" : "NO";
                    
                    eegAttentionEl.style.color = pred.attention_high ? "#00ff00" : "#aaa";
                    eegAttentionEl.textContent = pred.attention_high ? "HIGH" : "LOW";

                    // 3. Tận dụng phán quyết dự báo sóng não thực hiện trực tiếp hành vi
                    // Tránh dồn dập sự kiện bằng khóa chống rung (debounce logic nội hàm)
                    if (pred.label === "BLINK" && !pred.processed) {
                        let now = Date.now();
                        if (now - lastBlinkTime > 850) {
                            lastBlinkTime = now;
                            executeSelection();
                            fetch("/api/mark_processed", {method: "POST"}).catch(() => {});
                            
                            // Ghi nhận nhật ký mẫu dữ liệu thực nghiệm EOG/EEG
                            fetch("/api/add_log", {
                                method: "POST",
                                headers: {"Content-Type": "application/json"},
                                body: JSON.stringify({
                                    timestamp: new Date().toLocaleTimeString(),
                                    eog: pred.eog_raw,
                                    eeg: pred.eeg_raw,
                                    label: "BLINK"
                                })
                            }).catch(() => {});
                        }
                    } else if (pred.label === "LEFT" || pred.label === "RIGHT") {
                        let now = Date.now();
                        if (now - lastGazeTime > 1200) {
                            lastGazeTime = now;
                            fetch("/api/add_log", {
                                method: "POST",
                                headers: {"Content-Type": "application/json"},
                                body: JSON.stringify({
                                    timestamp: new Date().toLocaleTimeString(),
                                    eog: pred.eog_raw,
                                    eeg: pred.eeg_raw,
                                    label: pred.label
                                })
                            }).catch(() => {});
                        }
                    }

                    // 4. Cập nhật từ gợi ý predictive text
                    updateSuggestions(data.suggestions);

                    // 5. Cập nhật vẽ đồ thị thô từ dữ liệu ESP32
                    updateWaveforms(pred.eog_raw, pred.eeg_raw);

                } catch (e) {
                    console.error("Lỗi cập nhật API state:", e);
                }
            }

            function updateSuggestions(list) {
                const suggBox = document.getElementById("suggestionsBox");
                const btns = suggBox.getElementsByTagName("button");
                for (let i = 0; i < 5; i++) {
                    if (list && list[i]) {
                        btns[i].textContent = list[i];
                        btns[i].style.visibility = "visible";
                    } else {
                        btns[i].style.visibility = "hidden";
                    }
                }
            }

            async function selectSuggestion(index) {
                const suggBox = document.getElementById("suggestionsBox");
                const textWord = suggBox.getElementsByTagName("button")[index].textContent;
                
                // Thay thế từ hiện tại đang gõ dở bằng từ được chọn
                const words = textBuffer.trim().split(" ");
                if (words.length > 0) {
                    words[words.length - 1] = textWord;
                    textBuffer = words.join(" ") + " ";
                } else {
                    textBuffer = textWord + " ";
                }
                updateTypedText();
                
                // Gọi POST lên server báo cáo chọn từ học tập
                try {
                    await fetch(`/api/select_suggestion/${index}`, {method: "POST"});
                } catch(e) {}
            }

            // Gọi các API can thiệp giả lập thủ công nhanh chóng bằng nút bấm
            async function triggerManualBlink(type) {
                // type = 1: Single, type = 2: Double
                if (type === 1) {
                    executeSelection();
                } else {
                    executeDeletion();
                }
                try {
                    await fetch("/api/manual_blink", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({type: type})
                    });
                } catch(e){}
            }

            async function triggerManualMove(dir) {
                // Thay đổi cursor hoặc nhảy hàng
                if (dir === "LEFT") {
                    if (currentMode === "ROW") {
                        scanRow = (scanRow - 1 + 6) % 6;
                    } else {
                        scanCol = (scanCol - 1 + 6) % 6;
                    }
                } else {
                    if (currentMode === "ROW") {
                        scanRow = (scanRow + 1) % 6;
                    } else {
                        scanCol = (scanCol + 1) % 6;
                    }
                }
                updateScanHighlight();
                try {
                    await fetch("/api/manual_move", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({dir: dir})
                    });
                } catch(e){}
            }

            function selectKeyDirectly(r, c) {
                scanRow = r;
                scanCol = c;
                currentMode = "COL";
                executeSelection();
            }

            async function resetText() {
                textBuffer = "";
                updateTypedText();
                try {
                    await fetch("/api/reset", {method: "POST"});
                } catch(e){}
            }

            async function deleteLastChar() {
                executeDeletion();
                try {
                    await fetch("/api/delete_last", {method: "POST"});
                } catch(e){}
            }

            // Vẽ Canvas giả lập dải tần thô EOG / EEG
            const eogCtx = document.getElementById("eogCanvas").getContext("2d");
            const eegCtx = document.getElementById("eegCanvas").getContext("2d");

            function updateWaveforms(newEog, newEeg) {
                // Thêm mẫu vào hàng đợi history
                eogHistory.push(newEog || (Math.random() * 20 - 10));
                eegHistory.push(newEeg || (Math.random() * 10 - 5));
                
                if (eogHistory.length > maxPoints) eogHistory.shift();
                if (eegHistory.length > maxPoints) eegHistory.shift();

                drawGraph(eogCtx, eogHistory, "#00ffeb", 1500);
                drawGraph(eegCtx, eegHistory, "#00ff00", 500);
            }

            function drawGraph(ctx, data, color, scaleRange) {
                const width = ctx.canvas.width;
                const height = ctx.canvas.height;
                ctx.clearRect(0, 0, width, height);
                
                ctx.strokeStyle = "#333";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, height / 2);
                ctx.lineTo(width, height / 2);
                ctx.stroke();

                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                
                for (let i = 0; i < data.length; i++) {
                    const x = (i / (maxPoints - 1)) * width;
                    // Chuẩn hóa y theo biên độ dải đo sinh học
                    const valNormalized = data[i] / scaleRange;
                    const y = (height / 2) - (valNormalized * (height / 2) * 0.8);
                    
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            }

            // Thiết lập dọn khởi điểm màn hình
            window.onload = () => {
                createGrid();
                updateScanHighlight();
                startTimer();
                updateTypedText();
                
                // Chạy vòng lặp thăm dò API State cực nhanh 100ms
                setInterval(fetchState, 100);

                // Đồng bộ resize đồ thị tương đương canvas độ phân giải cao
                const resizeCanvas = (canvas) => {
                    canvas.width = canvas.parentElement.clientWidth;
                    canvas.height = canvas.parentElement.clientHeight;
                };
                resizeCanvas(document.getElementById("eogCanvas"));
                resizeCanvas(document.getElementById("eegCanvas"));
            };
        </script>
    </body>
    </html>
    """
    return render_template_string(inline_html)

@app.route("/api/state", methods=["GET"])
def get_state():
    """Nhận và gộp trạng thái EOG, EEG phục vụ đồng bộ giao diện (Eye Tracker đã loại bỏ)"""
    # Lấy nhãn dự đoán biosignal mới nhất
    pred = latest_prediction.copy()
    
    eye_status = "DISABLED"
    pred["system_mode"] = "EOG_ONLY"

    # Lấy gợi ý từ vựng theo tiền tố từ đang gõ dở
    last_word = ""
    words = app_state["text"].strip().split(" ")
    if words:
        last_word = words[-1]
        
    suggestions = predictor.predict(last_word, top_n=5)

    return jsonify({
        "cursor_row": app_state["cursor_row"],
        "cursor_col": app_state["cursor_col"],
        "scan_mode": app_state["scan_mode"],
        "prediction": pred,
        "suggestions": suggestions,
        "eye_direction": eye_status,
        "system_mode": pred.get("system_mode", "EOG_ONLY"),
        "text": app_state["text"]
    })

@app.route("/api/update_text", methods=["POST"])
def update_text():
    """Đồng bộ hóa văn bản tức thì từ Client-Side"""
    data = request.json or {}
    text_val = data.get("text")
    if text_val is not None:
        app_state["text"] = text_val
    return jsonify({"status": "update_success", "text": app_state["text"]})

@app.route("/api/mark_processed", methods=["POST"])
def mark_processed():
    """Đánh dấu sự kiện chớp mắt đã xử lý xong để tránh xung đột gõ chữ rác"""
    latest_prediction["processed"] = True
    latest_prediction["blink_detected"] = False
    latest_prediction["label"] = "NEUTRAL"
    return jsonify({"status": "mark_success"})

@app.route("/api/add_log", methods=["POST"])
def add_log():
    """Ghi nhận nhật ký mẫu dữ liệu EEG/EOG lâm sàng phục vụ đi thi nghiên cứu khoa học"""
    data = request.json or {}
    timestamp = data.get("timestamp", time.strftime("%H:%M:%S"))
    eog_val = data.get("eog", 0)
    eeg_val = data.get("eeg", 0)
    label_val = data.get("label", "NEUTRAL")
    
    csv_file = "dataset_log.csv"
    try:
        file_exists = os.path.exists(csv_file)
        with open(csv_file, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timestamp,eog,eeg,label\n")
            f.write(f"{timestamp},{eog_val},{eeg_val},{label_val}\n")
    except Exception as ex:
        print(f"[DatasetLog] Lỗi lưu CSV: {ex}")
        
    return jsonify({"status": "log_success"})

@app.route("/api/reset", methods=["POST", "GET"])
def reset_state():
    app_state["text"] = ""
    app_state["cursor_row"] = 0
    app_state["cursor_col"] = 0
    app_state["scan_mode"] = "ROW"
    return jsonify({"status": "reset_success"})

@app.route("/api/delete_last", methods=["POST", "GET"])
def delete_last_char():
    if len(app_state["text"]) > 0:
        app_state["text"] = app_state["text"][:-1]
    return jsonify({"status": "delete_success", "text": app_state["text"]})

@app.route("/api/select_suggestion/<int:index>", methods=["POST"])
def select_suggestion(index):
    """Chọn từ đề xuất gợi ý của Word_Predictor"""
    last_word = ""
    words = app_state["text"].strip().split(" ")
    if not words:
        return jsonify({"status": "no_words"})

    last_word = words[-1]
    suggestions = predictor.predict(last_word, top_n=5)
    
    if suggestions and index < len(suggestions):
        chosen = suggestions[index]
        # Thay từ cuối cùng bằng từ hoàn thiện
        words[-1] = chosen
        app_state["text"] = " ".join(words) + " "
        # Tăng trọng số từ vựng
        predictor.learn_word(chosen)
        return jsonify({"status": "success", "learned": chosen, "text": app_state["text"]})
        
    return jsonify({"status": "index_out_of_bound"})

@app.route("/api/manual_move", methods=["POST"])
def manual_move():
    """Di chuyển con trỏ quét bằng lệnh test"""
    data = request.json or {}
    direction = data.get("dir", "RIGHT")
    
    if app_state["scan_mode"] == "ROW":
        if direction == "LEFT":
            app_state["cursor_row"] = (app_state["cursor_row"] - 1 + 6) % 6
        else:
            app_state["cursor_row"] = (app_state["cursor_row"] + 1) % 6
    else:
        if direction == "LEFT":
            app_state["cursor_col"] = (app_state["cursor_col"] - 1 + 6) % 6
        else:
            app_state["cursor_col"] = (app_state["cursor_col"] + 1) % 6
            
    return jsonify({
        "status": "moved",
        "row": app_state["cursor_row"],
        "col": app_state["cursor_col"]
    })

@app.route("/api/manual_blink", methods=["POST"])
def manual_blink():
    """Hành vi nháy mắt mô phỏng từ xa"""
    data = request.json or {}
    blink_type = data.get("type", 1) # 1: Single, 2: Double
    
    # 1: Chớp đơn -> Chọn phần tử
    # 2: Chớp kép -> Xóa phần tử cuối
    if blink_type == 1:
        # Giả lập sườn tín hiệu cho máy học
        latest_prediction["label"] = "BLINK"
        latest_prediction["blink_detected"] = True
    else:
        latest_prediction["label"] = "NEUTRAL"
        latest_prediction["blink_detected"] = False
        if len(app_state["text"]) > 0:
            app_state["text"] = app_state["text"][:-1]
            
    return jsonify({"status": "blink_received", "text": app_state["text"]})

if __name__ == "__main__":
    print("[Mắt/Não] Đang khởi chạy Server Flask cổng 5000...")
    # Chạy Flask ở port 5000, host toàn cục cho phép ESP32 kết nối tự nhiên
    app.run(host="0.0.0.0", port=5000)
