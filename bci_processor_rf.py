# -*- coding: utf-8 -*-
"""
=== FILE: bci_processor_rf.py ===
NeuroType - BCI Processor using EOG + EEG features and Random Forest classifier.
"""

import json
import time
import threading
import numpy as np
import websocket
import joblib
import os

# Biến global lưu kết quả phân loại mới nhất cho Flask truy cập
latest_prediction = {
    "label": "NEUTRAL",
    "confidence": 1.0,
    "blink_detected": False,
    "attention_high": False,
    "eog_raw": 0,
    "eeg_raw": 0,
    "system_mode": "HYBRID"
}

class BCIProcessor:
    def __init__(self, ip="192.168.1.105", n_estimators=10):
        self.ip = ip
        self.ws_url = f"ws://{ip}:81"
        self.eog_buffer = []
        self.eeg_buffer = []
        self.buffer_size = 200
        self.lock = threading.Lock()
        self.running = False
        self.ws = None
        self.n_estimators = n_estimators
        
        # Load model Random Forest nếu có
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.load_model()

    def load_model(self):
        model_path = "bci_model_rf.pkl"
        if os.path.exists(model_path):
            try:
                payload = joblib.load(model_path)
                self.model = payload.get("model")
                self.scaler = payload.get("scaler")
                self.model_loaded = True
                print("[BCI] Đã tải model Random Forest và bộ chuẩn hóa Scaler thành công.")
            except Exception as e:
                print(f"[BCI] Lỗi tải model Random Forest: {e}")
        else:
            print("[BCI] Không tìm thấy bci_model_rf.pkl. Hệ thống sẽ dùng quy tắc ngưỡng cơ bản.")

    def extract_features(self, eog_win, eeg_win):
        """
        Trích xuất 7 đặc trưng:
        - EOG (4): mean, std, peak-to-peak (ptp), line integral (tích phân đường)
        - EEG (3): mean, std, variance (var)
        """
        eog_arr = np.array(eog_win)
        eeg_arr = np.array(eeg_win)
        
        # Đặc trưng EOG
        eog_mean = float(np.mean(eog_arr))
        eog_std = float(np.std(eog_arr))
        eog_ptp = float(np.ptp(eog_arr))
        # Tích phân đường: sum(|x[i] - x[i-1]|)
        eog_diff = np.abs(np.diff(eog_arr))
        eog_line_integral = float(np.sum(eog_diff)) if len(eog_diff) > 0 else 0.0
        
        # Đặc trưng EEG
        eeg_mean = float(np.mean(eeg_arr))
        eeg_std = float(np.std(eeg_arr))
        eeg_var = float(np.var(eeg_arr))
        
        features = [
            eog_mean, eog_std, eog_ptp, eog_line_integral,
            eeg_mean, eeg_std, eeg_var
        ]
        return features

    def detect_blink(self, eog_win):
        """Phát hiện chớp mắt (peak-to-peak EOG > 150)"""
        if len(eog_win) < 10:
            return False
        ptp = np.ptp(eog_win)
        return bool(ptp > 150)

    def detect_attention(self, eeg_win):
        """Phát hiện độ tập trung mạnh (variance EEG > 100)"""
        if len(eeg_win) < 10:
            return False
        var = np.var(eeg_win)
        return bool(var > 100)

    def classify(self, features):
        """Nhận diện nhãn (LEFT, RIGHT, NEUTRAL) từ đặc trưng trích xuất"""
        if self.model_loaded and self.model and self.scaler:
            try:
                # Chuẩn hóa đặc trưng
                feat_scaled = self.scaler.transform([features])
                pred = self.model.predict(feat_scaled)[0]
                
                # Tính độ tin cậy thông qua xác suất của quẩn thể cây quyết định
                try:
                    probs = self.model.predict_proba(feat_scaled)[0]
                    confidence = float(np.max(probs))
                except Exception:
                    confidence = 1.0
                
                # Trả về dưới dạng chuẩn (chỉ gom về nhóm điều khiển mắt)
                if pred in ["LEFT", "RIGHT", "NEUTRAL"]:
                    return pred, confidence
                return "NEUTRAL", 1.0
            except Exception as e:
                print(f"[BCI] Lỗi phân loại Random Forest: {e}")
        
        # Fallback bằng phân tích dựa trên ngưỡng (Rule-based)
        eog_mean = features[0]
        eog_ptp = features[2]
        
        # Ngưỡng dịch chuyển mắt EOG đơn giản
        if eog_ptp > 80:
            if eog_mean > 30:
                return "RIGHT", 0.8
            elif eog_mean < -30:
                return "LEFT", 0.8
        return "NEUTRAL", 1.0

    def process_window(self):
        """Xử lý buffer hiện tại khi có đủ số mẫu"""
        with self.lock:
            if len(self.eog_buffer) < self.buffer_size or len(self.eeg_buffer) < self.buffer_size:
                return
            
            eog_win = list(self.eog_buffer)
            eeg_win = list(self.eeg_buffer)
            
            # Cập nhật giá trị raw cuối cùng
            latest_eog = eog_win[-1]
            latest_eeg = eeg_win[-1]

        # Trích xuất đặc trưng và đưa ra quyết định
        features = self.extract_features(eog_win, eeg_win)
        blink = self.detect_blink(eog_win)
        attention = self.detect_attention(eeg_win)
        label, conf = self.classify(features)
        
        global latest_prediction
        latest_prediction["eog_raw"] = latest_eog
        latest_prediction["eeg_raw"] = latest_eeg
        latest_prediction["blink_detected"] = blink
        latest_prediction["attention_high"] = attention
        
        # Ghi đè label nếu phát hiện blink (blink được coi là sự kiện kích hoạt tức thì)
        if blink:
            latest_prediction["label"] = "BLINK"
            latest_prediction["confidence"] = 1.0
        else:
            latest_prediction["label"] = label
            latest_prediction["confidence"] = conf

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            eog_val = float(data.get("e", 0))
            eeg_val = float(data.get("b", 0))
            
            with self.lock:
                self.eog_buffer.append(eog_val)
                self.eeg_buffer.append(eeg_val)
                
                # Giới hạn kích thước buffer trượt
                if len(self.eog_buffer) > self.buffer_size:
                    self.eog_buffer.pop(0)
                if len(self.eeg_buffer) > self.buffer_size:
                    self.eeg_buffer.pop(0)
            
            self.process_window()
        except Exception:
            pass

    def on_error(self, ws, error):
        print(f"[BCI] Lỗi kết nối WebSocket (RF): {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("[BCI] Đã đóng kết nối WebSocket (RF).")

    def on_open(self, ws):
        print(f"[BCI] Kết nối thành công đến ESP32 ({self.ip}) cho mô hình Random Forest!")

    def connect(self):
        self.running = True
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.run_forever()

def start_websocket(ip="192.168.1.105", n_estimators=10):
    """Khởi động WebSocket Client trong một Thread nền"""
    processor = BCIProcessor(ip, n_estimators)
    
    def thread_worker():
        while True:
            try:
                print(f"[BCI] Đang kết nối đến ESP32 tại {processor.ws_url} cho bộ giải mã Random Forest...")
                processor.connect()
            except Exception as e:
                print(f"[BCI] Lỗi kết nối lại RF: {e}")
            time.sleep(3)

    t = threading.Thread(target=thread_worker, daemon=True)
    t.start()
    return processor
