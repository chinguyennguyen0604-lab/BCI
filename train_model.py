# -*- coding: utf-8 -*-
"""
=== FILE: train_model.py ===
NeuroType - CLI training helper. Collects real-time EOG/EEG samples from ESP32,
extracts features, trains an SVM model, and outputs 'bci_model.pkl'.
"""

import os
import json
import time
import websocket
import threading
import numpy as np
import joblib

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class DataCollector:
    def __init__(self):
        self.ip = "192.168.1.105"
        self.ws = None
        self.running = False
        
        # Buffer tạm phục vụ gom mẫu thời gian thực
        self.eog_raw_data = []
        self.eeg_raw_data = []
        self.timestamps = []
        
        # Tập dữ liệu huấn luyện cuối: danh sách features và tương ứng class index/label
        self.X_data = []
        self.y_data = []
        
        # Định danh 6 chuyển thái sinh học cần thu thập
        self.classes = ["LEFT", "RIGHT", "BLINK", "HIGH_ATT", "LOW_ATT", "NEUTRAL"]
        
    def connect_esp32(self, ip):
        """Khởi chạy kết nối tạm đến WebSocket để bắt sóng"""
        self.ip = ip
        ws_url = f"ws://{ip}:81"
        websocket.enableTrace(False)
        
        def on_message(ws, message):
            if not self.running:
                return
            try:
                data = json.loads(message)
                eog_val = float(data.get("e", 0))
                eeg_val = float(data.get("b", 0))
                t_val = float(data.get("t", time.time() * 1000))
                
                self.eog_raw_data.append(eog_val)
                self.eeg_raw_data.append(eeg_val)
                self.timestamps.append(t_val)
            except Exception:
                pass

        def on_error(ws, error):
            print(f"\n[Collector] Lỗi kết nối: {error}")

        def on_close(ws, close_status_code, close_msg):
            print("\n[Collector] Đã ngắt kết nối với ESP32.")

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Chạy trong luồng phụ để không block màn hình tương tác CLI
        t = threading.Thread(target=self.ws.run_forever, daemon=True)
        t.start()
        
        print(f"[Collector] Kết nối thành công tới {ws_url}. Hãy chắc chắn thiết bị ESP32 sẵn sàng bật điện cực!")

    def collect(self, label, duration=5.0):
        """Thu thập dữ liệu thô cho một nhãn cụ thể trong một khoảng thời gian (giây)"""
        print(f"\n---> CHUẨN BỊ THU THẬP NHÃN: [{label}] <---")
        print("Vòng đếm ngược bắt đầu...")
        for i in range(3, 0, -1):
            print(f"Hành động sau {i}...")
            time.sleep(1)
            
        print(f"=== ĐANG THU THẬP... (Thực hiện hành động trong {duration} giây) ===")
        
        # Reset các bộ gom thô
        self.eog_raw_data = []
        self.eeg_raw_data = []
        self.timestamps = []
        
        # Kích hoạt trạng thái thu nhận
        self.running = True
        time.sleep(duration)
        self.running = False
        
        print(f"=== XONG! Thu được {len(self.eog_raw_data)} điểm mẫu ===")
        
        if len(self.eog_raw_data) < 100:
            print("[Cảnh báo] Lấy mẫu quá ít! Vui lòng kiểm tra lại đường truyền ESP32.")
            return False
            
        # Trích xuất đặc trưng với cửa sổ trượt (Kích thước cửa sổ = 50 mẫu, Bước nhảy = 25 mẫu)
        win_size = 50
        step = 25
        features_extracted = 0
        
        eog_arr = np.array(self.eog_raw_data)
        eeg_arr = np.array(self.eeg_raw_data)
        
        for start in range(0, len(eog_arr) - win_size, step):
            end = start + win_size
            eog_win = eog_arr[start:end]
            eeg_win = eeg_arr[start:end]
            
            # Trích xuất 7 đặc trưng sinh học
            # EOG mean, std, ptp, line integral
            eog_mean = float(np.mean(eog_win))
            eog_std = float(np.std(eog_win))
            eog_ptp = float(np.ptp(eog_win))
            eog_diff = np.abs(np.diff(eog_win))
            eog_li = float(np.sum(eog_diff)) if len(eog_diff) > 0 else 0.0
            
            # EEG mean, std, var
            eeg_mean = float(np.mean(eeg_win))
            eeg_std = float(np.std(eeg_win))
            eeg_var = float(np.var(eeg_win))
            
            feat = [eog_mean, eog_std, eog_ptp, eog_li, eeg_mean, eeg_std, eeg_var]
            
            self.X_data.append(feat)
            self.y_data.append(label)
            features_extracted += 1
            
        print(f"-> Tạo thành công {features_extracted} cửa sổ đặc trưng trượt từ lượt này.")
        return True

    def train(self):
        """Huấn luyện model SVM bằng scikit-learn"""
        if not HAS_SKLEARN:
            print("[Lỗi] Huấn luyện máy học thất bại. Scikit-learn chưa được cài đặt!")
            return False
            
        if len(self.X_data) < 20:
            print("[Lỗi] Không đủ mẫu để huấn luyện. Tổng quan cần thu ít nhất hơn 20 cửa sổ dữ liệu.")
            return False
            
        X = np.array(self.X_data)
        y = np.array(self.y_data)
        
        print("\n=== BẮT ĐẦU HUẤN LUYỆN MODEL SVM ===")
        print(f"Tổng số mẫu đặc trưng đầu vào: {X.shape[0]}")
        
        # 1. Chuẩn hóa dữ liệu StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. Định nghĩa hệ phân loại SVM
        model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
        
        # 3. Đánh giá chéo 3-Fold Cross-Validation
        try:
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
            print(f"Độ chính xác trung bình (3-Fold CV Accuracy): {np.mean(scores)*100:.2f}% (Từng phần: {scores})")
        except Exception as e:
            print(f"[Đánh giá] Quá trình cross-validation gặp ngoại lệ: {e}. Tiến hành bỏ qua kiểm thử chéo.")
            
        # 4. Fit toàn diện bộ phân loại
        model.fit(X_scaled, y)
        
        # In báo cáo phân loại chi tiết (Classification Report) trên tập huấn luyện
        try:
            y_pred = model.predict(X_scaled)
            print("\nBáo cáo hiệu năng phân loại trên tập dữ liệu học:")
            print(classification_report(y, y_pred))
        except Exception:
            pass
            
        # 5. Lưu kết cấu model SVM và Scaler
        self.save_model(model, scaler)
        return True

    def save_model(self, model, scaler, filename="bci_model.pkl"):
        """Lưu trữ mô hình"""
        try:
            payload = {
                "model": model,
                "scaler": scaler,
                "timestamp": time.time()
            }
            joblib.dump(payload, filename)
            print(f"[Model] Đã lưu thành công tệp huấn luyện chuẩn hóa tại {filename}!")
        except Exception as e:
            print(f"[Model] Lỗi ghi file .pkl: {e}")

def main():
    if not HAS_SKLEARN:
        print("⚠️ CẢNH BÁO: Thư viện 'scikit-learn' chưa được cài đặt.")
        print("Hãy chạy lệnh 'pip install scikit-learn joblib numpy websocket-client' trước.")
        return

    collector = DataCollector()
    print("=================================================================")
    print("     CHƯƠNG TRÌNH HUẤN LUYỆN CALIBRATION NEUROTYPE SVM SYSTEM    ")
    print("=================================================================")
    
    ip = input("Nhập IP của thiết bị ESP32 (ví dụ 192.168.1.105): ").strip()
    if not ip:
        ip = "192.168.1.105"
        
    collector.connect_esp32(ip)
    
    # Cho phép thiết lập số lần lặp lại lấy mẫu cho mỗi lớp cử động tương ứng
    try:
        repeats = int(input("Nhập số lượt thu nhận (mỗi nhãn lặp lại từ 1-5 lần, khuyến nghị: 3): ") or "3")
    except ValueError:
        repeats = 3
        
    print(f"\nHệ thống sẽ thu tập dữ liệu cho bấy nhiêu lớp: {collector.classes}")
    print("Chu kỳ mỗi nhãn sẽ thu thập kéo dài liên tục trong 5 giây.")
    input("\nBấm Enter để bắt đầu khi đã đeo các điện cực EOG/EEG đúng vị trí...")
    
    for label in collector.classes:
        for r in range(repeats):
            print(f"\n>>> THU THẬP NHÃN {label} - LƯỢT {r+1}/{repeats} <<<")
            success = collector.collect(label, duration=5.0)
            if not success:
                # Nếu lỗi đường truyền hoặc không thu nhận được mẫu, cho phép nhảy thử lại
                choice = input("Lỗi xảy ra, có thử lại lượt thu nhận này? (y/n): ")
                if choice.lower() == "y":
                    collector.collect(label, duration=5.0)
            time.sleep(1.5) # Khoảng nghỉ xả cơ mắt cho người thử nghiệm
            
    # Tiến hành train và xuất báo cáo lưu tệp
    collector.train()
    print("\n--- HOÀN TẤT THỬ NGHIỆM HUẤN LUYỆN NEUROTYPE ---")

if __name__ == "__main__":
    main()
