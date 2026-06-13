# -*- coding: utf-8 -*-
"""
=== FILE: profile_manager.py ===
NeuroType - Calibration Profile Manager for personalized thresholds.
Saves users' calibration metadata into 'profiles/' folder.
"""

import os
import json

class ProfileManager:
    def __init__(self, root_dir="profiles"):
        self.root_dir = root_dir
        # Tự động tạo thư mục rễ lưu trữ hồ sơ cấu hình
        if not os.path.exists(self.root_dir):
            try:
                os.makedirs(self.root_dir)
                print(f"[Profile] Đã tạo thư mục lưu trữ profile tại '{self.root_dir}'")
            except Exception as e:
                print(f"[Profile] Lỗi khởi tạo thư mục profile: {e}")

    def save_profile(self, user_name, data):
        """
        Lưu cấu hình người dùng vào tệp tin JSON riêng biệt
        Dữ liệu mẫu: {"blink_threshold": 150, "attention_threshold": 100, "eog_mean": 10.5, "eog_std": 5.2}
        """
        if not user_name:
            return False
        
        # Làm sạch tên file để tránh tấn công path traversal
        clean_name = "".join([c for c in user_name if c.isalnum() or c in ("-", "_")]).strip()
        if not clean_name:
            return False
            
        filepath = os.path.join(self.root_dir, f"{clean_name}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"[Profile] Đã lưu cấu hình người dùng '{clean_name}'")
            return True
        except Exception as e:
            print(f"[Profile] Lỗi lưu cấu hình cho '{clean_name}': {e}")
            return False

    def load_profile(self, user_name):
        """Tải profile và trả về dưới dạng dict, trả về None nếu không tồn tại"""
        if not user_name:
            return None
            
        clean_name = "".join([c for c in user_name if c.isalnum() or c in ("-", "_")]).strip()
        filepath = os.path.join(self.root_dir, f"{clean_name}.json")
        
        if not os.path.exists(filepath):
            return None
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[Profile] Tải thành công hồ sơ '{clean_name}'")
            return data
        except Exception as e:
            print(f"[Profile] Không thể load cấu hình người dùng '{clean_name}': {e}")
            return None

    def list_profiles(self):
        """Trả về danh sách tên cấu hình người dùng hiện lưu trong hệ thống"""
        if not os.path.exists(self.root_dir):
            return []
            
        profiles = []
        try:
            for file in os.listdir(self.root_dir):
                if file.endswith(".json"):
                    profiles.append(file[:-5]) # lấy tên trước phần mở rộng .json
        except Exception as e:
            print(f"[Profile] Lỗi liệt kê cấu hình: {e}")
            
        return sorted(profiles)

    def delete_profile(self, user_name):
        """Xóa hồ sơ người dùng chỉ định"""
        if not user_name:
            return False
            
        clean_name = "".join([c for c in user_name if c.isalnum() or c in ("-", "_")]).strip()
        filepath = os.path.join(self.root_dir, f"{clean_name}.json")
        
        if not os.path.exists(filepath):
            return False
            
        try:
            os.remove(filepath)
            print(f"[Profile] Đã xóa hồ sơ người dùng '{clean_name}'")
            return True
        except Exception as e:
            print(f"[Profile] Không thể xóa hồ sơ người dùng '{clean_name}': {e}")
            return False

# Tạo thực thể tĩnh dùng chung cho toàn bộ ứng dụng
profile_mgr = ProfileManager()
