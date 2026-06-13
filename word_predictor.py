# -*- coding: utf-8 -*-
"""
=== FILE: word_predictor.py ===
NeuroType - Predictive Text Engine for Vietnamese typing helper.
Supports frequency learning and JSON persistence.
"""

import json
import os

class WordPredictor:
    def __init__(self, filepath="words.json"):
        self.filepath = filepath
        # 100 từ khởi tạo thông dụng theo mô tả
        self.default_vocabulary = [
            "có", "không", "và", "của", "một", "những", "cho", "với", "tôi", "bạn", 
            "chúng", "ta", "này", "kia", "đó", "nay", "mai", "hôm", "qua", "năm", 
            "tháng", "ngày", "giờ", "phút", "giây", "làm", "đi", "chạy", "ngồi", "đứng", 
            "nói", "nghe", "thấy", "biết", "hiểu", "thương", "yêu", "ghét", "vui", "buồn", 
            "khóc", "cười", "ăn", "uống", "ngủ", "thức", "đọc", "viết", "học", "dạy", 
            "thi", "đỗ", "trượt", "đẹp", "xấu", "tốt", "cao", "thấp", "dài", "ngắn", 
            "nhanh", "chậm", "nóng", "lạnh", "to", "nhỏ", "nặng", "nhẹ", "đắt", "rẻ", 
            "giàu", "nghèo", "thông minh", "ngu dốt", "chăm chỉ", "lười biếng", 
            "thật thà", "gian dối", "tốt bụng", "độc ác", "hiền lành", "hung dữ", 
            "dũng cảm", "nhút nhát", "kiên nhẫn", "nóng vội"
        ]
        
        # Dictionary lưu trữ dạng {từ: tần suất_xuất_hiện}
        self.words = {}
        self.load_dictionary()

    def load_dictionary(self):
        """Đọc từ điển từ file words.json, nếu không tồn tại thì nạp 100 từ cơ bản"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.words = json.load(f)
                print(f"[Predictor] Đã tải thành công {len(self.words)} từ từ file.")
            except Exception as e:
                print(f"[Predictor] Lỗi đọc từ điển: {e}. Sử dụng từ điển mặc định.")
                self._initialize_defaults()
        else:
            self._initialize_defaults()

    def _initialize_defaults(self):
        self.words = {word.lower(): 1 for word in self.default_vocabulary}
        self.save_dictionary()
        print("[Predictor] Đã tạo từ điển mặc định với 100 từ gốc.")

    def save_dictionary(self):
        """Lưu tần suất từ vựng vào file JSON"""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.words, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[Predictor] Gặp lỗi khi lưu từ điển: {e}")

    def learn_word(self, word):
        """Học thêm một từ mới hoặc tăng trọng số nếu từ đã tồn tại"""
        if not word or not isinstance(word, str):
            return
        
        word_clean = word.strip().lower()
        if not word_clean:
            return
            
        if word_clean in self.words:
            self.words[word_clean] += 1
        else:
            self.words[word_clean] = 1
            
        # Thường xuyên lưu lại khi học được từ mới
        self.save_dictionary()

    def predict(self, prefix, top_n=5):
        """
        Dự đoán các gợi ý khả dĩ nhất bắt đầu bằng prefix
        Trả về danh sách các từ có tần suất cao giảm dần.
        """
        if not prefix:
            # Gợi ý các từ thông dụng nhất nếu prefix rỗng
            sorted_all = sorted(self.words.items(), key=lambda item: item[1], reverse=True)
            return [word for word, score in sorted_all[:top_n]]
            
        prefix_clean = prefix.strip().lower()
        candidates = []
        
        for word, score in self.words.items():
            if word.startswith(prefix_clean):
                candidates.append((word, score))
                
        # Sắp xếp theo tần suất xuất hiện giảm dần
        candidates.sort(key=lambda item: item[1], reverse=True)
        
        return [word for word, score in candidates[:top_n]]

    def get_vocabulary_size(self):
        """Trả về kích thước tập từ vựng hiện tại"""
        return len(self.words)
