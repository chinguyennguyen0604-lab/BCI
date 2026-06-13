import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Cpu, 
  Eye, 
  Settings, 
  Terminal, 
  FileText, 
  Copy, 
  Check, 
  RefreshCw, 
  User, 
  Delete,
  Zap,
  BarChart3,
  Layout,
  Info,
  Camera,
  Play,
  Database,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import {
  KnnClassifier,
  RandomForestClassifier,
  SvmClassifier,
  runModelEvaluation,
  BASELINE_TRAINING_DATA,
  extractFeatures,
  featuresToArray,
  BciFeatures,
  BciSample,
  applyNotchFilter50Hz,
  applyBandpassFilter
} from './bciProcessor';

// === ĐỊNH NGHĨA BÀN PHÍM CO-DI TRÌNH CHỈNH SỬA ===
const GRID = [
  ['A', 'B', 'C', 'D', 'E', 'F'],
  ['G', 'H', 'I', 'J', 'K', 'L'],
  ['M', 'N', 'O', 'P', 'Q', 'R'],
  ['S', 'T', 'U', 'V', 'W', 'X'],
  ['Y', 'Z', '0', '1', '2', '3'],
  ['4', '5', '6', '7', '8', '9'],
  ['⌫', '␣', '←', '→', '↺ RESET', '🗑️ CLEAR']
];

// === TỪ ĐIỂN 100 TỪ TIẾNG VIỆT THÔNG DỤNG ===
const VOCABULARY = [
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
];

// === CƠ CHẾ TELEX TIẾNG VIỆT LÂM SÀNG ===
function translateTelex(text: string): string {
  let result = text;
  
  // Ký tự tổ hợp cơ bản
  result = result.replace(/aa/gi, (m) => m === 'aa' ? 'â' : 'Â');
  result = result.replace(/ee/gi, (m) => m === 'ee' ? 'ê' : 'Ê');
  result = result.replace(/oo/gi, (m) => m === 'oo' ? 'ô' : 'Ô');
  result = result.replace(/aw/gi, (m) => m === 'aw' ? 'ă' : 'Ă');
  result = result.replace(/ow/gi, (m) => m === 'ow' ? 'ơ' : 'Ơ');
  result = result.replace(/uw/gi, (m) => m === 'uw' ? 'ư' : 'Ư');
  result = result.replace(/dd/gi, (m) => m === 'dd' ? 'đ' : 'Đ');
  
  // Dấu thanh: s (sắc), f (huyền), r (hỏi), x (ngã), j (nặng)
  const toneMap: { [key: string]: string } = {
    'as': 'á', 'af': 'à', 'ar': 'ả', 'ax': 'ã', 'aj': 'ạ',
    'âs': 'ấ', 'âf': 'ầ', 'âr': 'ẩu', 'âx': 'ẫ', 'âj': 'ậ',
    'ăs': 'ắ', 'ăf': 'ằ', 'ăr': 'ẳ', 'ăx': 'ẵ', 'ăj': 'ặ',
    'es': 'é', 'ef': 'è', 'er': 'ẻ', 'ex': 'ẽ', 'ej': 'ẹ',
    'ês': 'ế', 'êf': 'ề', 'êr': 'ể', 'êx': 'ễ', 'êj': 'ệ',
    'os': 'ó', 'of': 'ò', 'or': 'ỏ', 'ox': 'õ', 'oj': 'ọ',
    'ôs': 'ố', 'ôf': 'ồ', 'ôr': 'ổ', 'ôx': 'ỗ', 'ôj': 'ộ',
    'ơs': 'ớ', 'ơf': 'ờ', 'ơr': 'ở', 'ơx': 'ỡ', 'ơj': 'ợ',
    'us': 'ú', 'uf': 'ù', 'ur': 'ủ', 'ux': 'ũ', 'uj': 'ụ',
    'ưs': 'ứ', 'ưf': 'ừ', 'ưr': 'ử', 'ưx': 'ữ', 'ưj': 'ự',
    'is': 'í', 'if': 'ì', 'ir': 'ỉ', 'ix': 'ĩ', 'ij': 'ị',
    'ys': 'ý', 'yf': 'ỳ', 'yr': 'ỷ', 'yx': 'ỹ', 'yj': 'ỵ',
  };

  for (const [key, val] of Object.entries(toneMap)) {
    const regex = new RegExp(key, 'gi');
    result = result.replace(regex, (m) => {
      if (m === m.toUpperCase()) return val.toUpperCase();
      return val;
    });
  }

  return result;
}

// === CƠ CHẾ DỰ ĐOÁN TỪ THEO PREFIX ===
function getSuggestions(text: string, count = 5): string[] {
  const words = text.trim().split(/\s+/);
  const lastWord = words[words.length - 1]?.toLowerCase() || '';
  if (!lastWord) {
    return VOCABULARY.slice(0, count);
  }
  
  const matches = VOCABULARY.filter(w => w.startsWith(lastWord));
  if (matches.length === 0) {
    return VOCABULARY.slice(0, count);
  }
  return matches.slice(0, count);
}

// === MÃ NGUỒN CỦA 8 FILE ===
import { SOURCE_FILES } from './sourceFiles';

export default function App() {
  // === TRẠNG THÁI HIỂN THỊ CHÍNH ===
  const [typedText, setTypedText] = useState("");
  const [cursorPosition, setCursorPosition] = useState(0);
  const suggestions = getSuggestions(typedText.slice(0, cursorPosition));
  const [currentRow, setCurrentRow] = useState(0);
  const [currentCol, setCurrentCol] = useState(0);
  const [scanMode, setScanMode] = useState<"ROW" | "COL">("ROW");
  const [scanSpeedRow, setScanSpeedRow] = useState(1000); // ms
  const [scanSpeedCol, setScanSpeedCol] = useState(600); // ms

  // ĐỊNH NGHĨA RÀO QUÉT ĐỘNG CHO BÀN PHÍM TÙY BIẾN
  const SUGGESTIONS_ROW_INDEX = GRID.length; // index 7
  const TOTAL_SCAN_ROWS = GRID.length + 1;  // total 8 rows

  // BẢO VỆ CON TRỎ SOẠN THẢO TRONG GIỚI HẠN CHUỐI
  useEffect(() => {
    setCursorPosition((prev) => Math.max(0, Math.min(typedText.length, prev)));
  }, [typedText]);

  // Issue 1: State synchronization locks to prevent polling race loops
  const lastLocalWriteRef = useRef<number>(0);

  // Issue 5: Double-trigger hardware-level refractory prevention blocks
  const [refractoryActive, setRefractoryActive] = useState(false);
  const [refractoryTimeLeft, setRefractoryTimeLeft] = useState(0);

  // NCKH Research Metrics state
  const [typingStartTime, setTypingStartTime] = useState<number | null>(null);
  const [errorsCount, setErrorsCount] = useState(0);
  const [totalSelections, setTotalSelections] = useState(0);
  const [cpm, setCpm] = useState(0);
  const [wpm, setWpm] = useState(0);
  const [itr, setItr] = useState(0);

  // HÀNH VI CHỈNH SỬA TOÀN DIỆN KHOA HỌC
  const handleResetCursorAction = () => {
    lastLocalWriteRef.current = Date.now();
    setCurrentRow(0);
    setCurrentCol(0);
    setScanMode("ROW");
  };

  const handleClearAction = () => {
    lastLocalWriteRef.current = Date.now();
    setTypedText("");
    setCursorPosition(0);
    setTypingStartTime(null);
    setErrorsCount(0);
    setTotalSelections(0);
    setCpm(0);
    setWpm(0);
    setItr(0);
    fetch("/api/reset").catch(() => {});
  };

  const handleMoveCursorLeft = () => {
    lastLocalWriteRef.current = Date.now();
    setCursorPosition((prev) => Math.max(0, prev - 1));
  };

  const handleMoveCursorRight = () => {
    lastLocalWriteRef.current = Date.now();
    setCursorPosition((prev) => Math.min(typedText.length, prev + 1));
  };

  const handleTypeCharAction = (char: string) => {
    lastLocalWriteRef.current = Date.now();
    if (typingStartTime === null) {
      setTypingStartTime(Date.now());
    }
    setTypedText((prev) => {
      const left = prev.slice(0, cursorPosition);
      const right = prev.slice(cursorPosition);
      const translatedLeft = translateTelex(left + char);
      const diff = translatedLeft.length - left.length;
      setCursorPosition((curr) => curr + diff);
      return translatedLeft + right;
    });
  };

  const handleSpaceAction = () => {
    lastLocalWriteRef.current = Date.now();
    if (typingStartTime === null) {
      setTypingStartTime(Date.now());
    }
    setTypedText((prev) => {
      const left = prev.slice(0, cursorPosition);
      const right = prev.slice(cursorPosition);
      setCursorPosition(cursorPosition + 1);
      return left + " " + right;
    });
  };

  const handleDeleteAction = () => {
    lastLocalWriteRef.current = Date.now();
    // Record delete action as a clinical typo corrective event
    setErrorsCount((prev) => prev + 1);
    
    if (cursorPosition > 0) {
      setTypedText((prev) => {
        const left = prev.slice(0, cursorPosition - 1);
        const right = prev.slice(cursorPosition);
        setCursorPosition(cursorPosition - 1);
        return left + right;
      });
      fetch("/api/delete_last").catch(() => {});
    }
  };

  const handleSuggestionClick = (word: string) => {
    lastLocalWriteRef.current = Date.now();
    if (typingStartTime === null) {
      setTypingStartTime(Date.now());
    }
    setTypedText((prev) => {
      const left = prev.slice(0, cursorPosition);
      const right = prev.slice(cursorPosition);
      
      const lastSpaceIdx = left.lastIndexOf(" ");
      const prefix = lastSpaceIdx >= 0 ? left.slice(0, lastSpaceIdx + 1) : "";
      const newLeft = prefix + word + " ";
      setCursorPosition(newLeft.length);
      return newLeft + right;
    });
  };
  
  const [gazeDirection, setGazeDirection] = useState<"LEFT" | "CENTER" | "RIGHT">("CENTER");
  const [isBlinking, setIsBlinking] = useState(false);
  const [attentionLevel, setAttentionLevel] = useState(42); // 0-100% (EEG Var proxy)
  const [systemMode, setSystemMode] = useState<"HYBRID" | "EOG_ONLY">("HYBRID");
  
  // Hiệu chuẩn (Calibration Settings)
  const [blinkThreshold, setBlinkThreshold] = useState(150);
  const [attentionThreshold, setAttentionThreshold] = useState(100);
  const [baselineEog, setBaselineEog] = useState<number>(10);
  const [eegTriggerEnabled, setEegTriggerEnabled] = useState(true);
  const [profileName, setProfileName] = useState("Default_User");

  // Bộ chọn file code explorer
  const [activeCodeTab, setActiveCodeTab] = useState<keyof typeof SOURCE_FILES>("bci_processor_svm.py");
  const [codeCopied, setCodeCopied] = useState(false);
  const [activeUIPanel, setActiveUIPanel] = useState<"DASHBOARD" | "CODE_EXPLORER">("DASHBOARD");

  // Buffer vẽ canvas tín hiệu
  const eogCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const eegCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Sóng Biosignal ảo
  const signalPhaseRef = useRef(0);

  // Bộ lọc EOG Left/Right
  const [gazeThresholdLeft, setGazeThresholdLeft] = useState(0.42);
  const [gazeThresholdRight, setGazeThresholdRight] = useState(0.58);
  const [gazeStatus, setGazeStatus] = useState<string>("CENTER");

  // === TRẠNG THÁI SẠCH CỦA SK SKIN CONTACT ===
  const [electrodeContact, setElectrodeContact] = useState<"GOOD" | "FAIR" | "POOR">("GOOD");

  // === KIỂM SOÁT LUỒNG VÀ TRÁNH DỘI TÍN HIỆU BINH CHỨNG KHOA HỌC ===
  const [preventedAccidents, setPreventedAccidents] = useState<number>(0);
  const lastBlinkTimeRef = useRef<number>(0);
  const eventQueueRef = useRef<{ type: "BLINK" | "GAZE"; payload?: any }[]>([]);

  const pushEvent = (type: "BLINK" | "GAZE", payload?: any) => {
    eventQueueRef.current.push({ type, payload });
  };

  // --- EVENT QUEUE CONTROLLER & EEG COGNITIVE GATE (Sửa số 3 & EEG Gate) ---
  useEffect(() => {
    const processQueue = setInterval(() => {
      if (eventQueueRef.current.length === 0) return;
      const event = eventQueueRef.current.shift();
      if (!event) return;

      if (event.type === "BLINK") {
        const now = Date.now();
        const timeSinceLastBlink = now - lastBlinkTimeRef.current;
        
        // Sửa số 1: Debounce chớp mắt cứng rắn khóa dội tín hiệu (Refractory Protection)
        if (refractoryActive || timeSinceLastBlink < 800) {
          return;
        }

        // EEG CONFIRMATION MECHANISM (Blink + Focus > 55% mới chọn hoặc EEG hỗ trợ xác thực)
        const isManual = event.payload?.isManual;
        const satisfiesCognitiveGate = systemMode === "EOG_ONLY" || attentionLevel >= 55 || isManual;

        if (satisfiesCognitiveGate) {
          lastBlinkTimeRef.current = now;
          setLastBlinkTime(now);
          handleSelectAction();
        } else {
          // Ngăn ngừa khoảnh khắc nhầm lẫn và dội lỗi
          setPreventedAccidents(prev => prev + 1);
          showToast("⚠️ Chọn hỏng bị từ chối: Độ tập trung thấp (Focus < 55%)", "error");
        }
      } else if (event.type === "GAZE") {
        const dir = event.payload?.dir;
        if (dir && dir !== gazeDirection) {
          setGazeDirection(dir);
          setTimeout(() => setGazeDirection("CENTER"), 800);
        }
      }
    }, 75);
    return () => clearInterval(processQueue);
  }, [refractoryActive, attentionLevel, systemMode, currentRow, currentCol, scanMode, suggestions, gazeDirection]);

  // === TRẠNG THÁI BỘ PHÂN LỚP HỌC MÁY BCI ===
  const [selectedClassifier, setSelectedClassifier] = useState<"KNN" | "RF" | "SVM">("SVM");
  const [trainingData, setTrainingData] = useState<BciSample[]>(BASELINE_TRAINING_DATA);
  const [currentBciFeatures, setCurrentBciFeatures] = useState<BciFeatures>({
    variance: 80,
    amplitudeRange: 38,
    zeroCrossingRate: 0.09,
    alphaPower: 72,
    betaPower: 8,
    thetaPower: 20
  });
  const [predictedBciLabel, setPredictedBciLabel] = useState<string>("RELAX");
  const [predictedBciConfidence, setPredictedBciConfidence] = useState<number>(0.9);
  const [knnNeighbors, setKnnNeighbors] = useState<{ label: string; dist: number }[]>([]);
  const [rfVotes, setRfVotes] = useState<{ [key: string]: number }>({});
  const [svmScores, setSvmScores] = useState<{ [key: string]: number }>({});
  const [knnK, setKnnK] = useState<number>(3);
  const [rfTrees, setRfTrees] = useState<number>(8);
  const [customLabelInput, setCustomLabelInput] = useState<string>("FOCUS");

  // --- FALLBACK MODE & PROFILE HIỆU CHUẨN & SCIENTIFIC EVALUATION ---
  const [isFallbackEnabled, setIsFallbackEnabled] = useState(true);
  const [fallbackThreshold, setFallbackThreshold] = useState(65); // 65%
  const [forceFallbackMode, setForceFallbackMode] = useState(false);
  const [isFallbackActive, setIsFallbackActive] = useState(false);

  // States hiển thị tối ưu giao diện BCI demo
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isCameraSectionExpanded, setIsCameraSectionExpanded] = useState(false);
  const [isAutoScanActive, setIsAutoScanActive] = useState(true);

  // Multipliers for signal tuning
  const [leftGain, setLeftGain] = useState<number>(1.0);
  const [rightGain, setRightGain] = useState<number>(1.0);

  interface CalibrationProfile {
    id: string;
    name: string;
    blinkThreshold: number;
    attentionThreshold: number;
    leftGain: number;
    rightGain: number;
    systemMode: "HYBRID" | "EOG_ONLY";
    description: string;
    baselineEog?: number;
    leftThreshold?: number;
    rightThreshold?: number;
  }

  const PRESET_PROFILES: CalibrationProfile[] = [
    { id: "default", name: "Hồ Sơ Mặc Định", blinkThreshold: 150, attentionThreshold: 100, leftGain: 1.0, rightGain: 1.0, systemMode: "HYBRID", description: "Cấu hình tiêu chuẩn của hệ thống, đa dụng cho hầu hết người dùng mới.", baselineEog: 10, leftThreshold: 0.42, rightThreshold: 0.58 },
    { id: "high_sensitivity", name: "Nhạy Gaze Cao (EOG Tuned)", blinkThreshold: 110, attentionThreshold: 80, leftGain: 1.5, rightGain: 1.5, systemMode: "HYBRID", description: "Băng thông nhạy sáng cao, tăng ích biên độ di động hẹp lên 1.5x.", baselineEog: 8, leftThreshold: 0.40, rightThreshold: 0.60 },
    { id: "fast_blink", name: "Chớp Mắt Tốc Độ (No-Noise Blink)", blinkThreshold: 190, attentionThreshold: 110, leftGain: 0.8, rightGain: 0.8, systemMode: "HYBRID", description: "Lọc nhiễu chớp mắt bất động mạnh mẽ, giảm độ nhạy chớp chập để tránh gõ phím nhầm.", baselineEog: 12, leftThreshold: 0.44, rightThreshold: 0.56 },
    { id: "eog_only_mode", name: "Thuần Vận Động Mắt (EOG Only)", blinkThreshold: 130, attentionThreshold: 100, leftGain: 1.2, rightGain: 1.2, systemMode: "EOG_ONLY", description: "Tập trung hoàn toàn vào vận động cơ mắt EOG để hiệu chỉnh gõ chữ, tắt kích hoạt EEG.", baselineEog: 10, leftThreshold: 0.42, rightThreshold: 0.58 }
  ];

  const [profiles, setProfiles] = useState<CalibrationProfile[]>(PRESET_PROFILES);
  const [activeProfileId, setActiveProfileId] = useState<string>("default");
  const [newProfileNameInput, setNewProfileNameInput] = useState("");
  const [toastMessage, setToastMessage] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);

  const isMountedRef = useRef(false);

  // Load custom calibration profiles from local storage on mount
  useEffect(() => {
    isMountedRef.current = true;
    const saved = localStorage.getItem("neurotype_profiles");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          const merged = [...PRESET_PROFILES];
          parsed.forEach((p: any) => {
            if (p && p.id && !merged.some(m => m.id === p.id)) {
              merged.push(p);
            }
          });
          setProfiles(merged);
        }
      } catch (e) {
        console.error("Lỗi parse cấu hình cá nhân:", e);
      }
    }
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  // Auto Save active profile parameters to list and localStorage
  useEffect(() => {
    if (!isMountedRef.current) return;
    
    setProfiles(prev => {
      const idx = prev.findIndex(p => p.id === activeProfileId);
      if (idx === -1) return prev;
      
      const current = prev[idx];
      // Avoid infinite update if values are identical
      if (
        current.blinkThreshold === blinkThreshold &&
        current.attentionThreshold === attentionThreshold &&
        current.leftGain === leftGain &&
        current.rightGain === rightGain &&
        current.baselineEog === baselineEog &&
        current.leftThreshold === gazeThresholdLeft &&
        current.rightThreshold === gazeThresholdRight
      ) {
        return prev;
      }
      
      const updated = prev.map(p => {
        if (p.id === activeProfileId) {
          return {
            ...p,
            blinkThreshold,
            attentionThreshold,
            leftGain,
            rightGain,
            baselineEog,
            leftThreshold: gazeThresholdLeft,
            rightThreshold: gazeThresholdRight,
          };
        }
        return p;
      });
      
      // Save ONLY custom profiles to local storage
      const customOnly = updated.filter(p => !["default", "high_sensitivity", "fast_blink", "eog_only_mode"].includes(p.id));
      localStorage.setItem("neurotype_profiles", JSON.stringify(customOnly));
      
      return updated;
    });
  }, [blinkThreshold, attentionThreshold, leftGain, rightGain, baselineEog, gazeThresholdLeft, gazeThresholdRight, activeProfileId]);

  const showToast = (text: string, type: "success" | "error" | "info" = "success") => {
    setToastMessage({ text, type });
    const timer = setTimeout(() => {
      setToastMessage(prev => prev?.text === text ? null : prev);
    }, 4000);
    return () => clearTimeout(timer);
  };

  const [evaluationResult, setEvaluationResult] = useState<any>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  // Auto-evaluation triggered on any model state change
  useEffect(() => {
    setIsEvaluating(true);
    const timer = setTimeout(() => {
      const res = runModelEvaluation(selectedClassifier, trainingData, 5, knnK, rfTrees);
      setEvaluationResult(res);
      setIsEvaluating(false);
    }, 400); 
    return () => clearTimeout(timer);
  }, [selectedClassifier, trainingData, knnK, rfTrees]);

  // Trạng thái lưu thời gian sự kiện EOG/EEG cập nhật thực tế
  const [lastBlinkTime, setLastBlinkTime] = useState<number>(Date.now() - 17000);
  const [lastLeftGazeTime, setLastLeftGazeTime] = useState<number>(Date.now() - 25000);
  const [lastRightGazeTime, setLastRightGazeTime] = useState<number>(Date.now() - 38000);
  const [currentTimeTick, setCurrentTimeTick] = useState<number>(Date.now());

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentTimeTick(Date.now());
    }, 100);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (isBlinking) {
      setLastBlinkTime(Date.now());
    }
  }, [isBlinking]);

  useEffect(() => {
    if (gazeDirection === "LEFT") {
      setLastLeftGazeTime(Date.now());
    } else if (gazeDirection === "RIGHT") {
      setLastRightGazeTime(Date.now());
    }
  }, [gazeDirection]);

  const formatSecsAgo = (timestamp: number) => {
    const sec = (currentTimeTick - timestamp) / 1000;
    if (sec <= 0 || isNaN(sec)) return "0.0s ago";
    if (sec > 59.9) return "60s+ ago";
    return `${sec.toFixed(1)}s ago`;
  };

  // === DỰ ĐOÁN & PHÂN TÍCH HỌC MÁY BCI THỜI GIAN THỰC ===
  useEffect(() => {
    const scale = attentionLevel > 50 ? 1.6 : 0.7;
    const signal: number[] = [];
    const basePhase = signalPhaseRef.current || 0;
    
    // Mô phỏng 200 điểm tín hiệu thô
    for (let i = 0; i < 200; i++) {
      const w1 = Math.sin(i * 0.15 + basePhase) * 6 * scale;
      const w2 = Math.sin(i * 0.35 - basePhase * 1.5) * 2.5 * scale;
      const w3 = Math.cos(i * 0.05 + basePhase * 0.8) * 9 * (1.2 - scale/3);
      let val = w1 + w2 + w3;
      
      if (isBlinking) {
        const dist = Math.abs(i - 100);
        if (dist < 25) {
          val += (50 - dist * 2) * Math.sin((dist / 25) * Math.PI + Math.PI / 2);
        }
      }
      signal.push(val);
    }

    // Trích xuất tham số đặc trưng từ tín hiệu thô (Variance, alphaPower, betaPower...)
    const extracted = extractFeatures(signal, 250);

    // Xác định trạng thái của người dùng làm giàu đặc trưng
    const isLeft = gazeDirection === "LEFT" || gazeStatus === "LEFT";
    const isRight = gazeDirection === "RIGHT" || gazeStatus === "RIGHT";
    const isBlinkNow = isBlinking;

    let finalFeatures: BciFeatures = { ...extracted };

    if (isBlinkNow) {
      finalFeatures.variance = 1010 + Math.random() * 200;
      finalFeatures.amplitudeRange = 360 + Math.random() * 60;
      finalFeatures.zeroCrossingRate = 0.035;
      finalFeatures.alphaPower = 6;
      finalFeatures.betaPower = 8;
      finalFeatures.thetaPower = 86;
    } else if (isLeft) {
      finalFeatures.variance = 190 + Math.random() * 15;
      finalFeatures.amplitudeRange = 93 + Math.random() * 8;
      finalFeatures.zeroCrossingRate = 0.12;
      finalFeatures.alphaPower = 39;
      finalFeatures.betaPower = 21;
      finalFeatures.thetaPower = 40;
    } else if (isRight) {
      finalFeatures.variance = 195 + Math.random() * 15;
      finalFeatures.amplitudeRange = 91 + Math.random() * 8;
      finalFeatures.zeroCrossingRate = 0.13;
      finalFeatures.alphaPower = 37;
      finalFeatures.betaPower = 23;
      finalFeatures.thetaPower = 40;
    } else {
      if (attentionLevel > 50) {
        finalFeatures.variance = 285 + Math.random() * 25;
        finalFeatures.amplitudeRange = 124 + Math.random() * 12;
        finalFeatures.zeroCrossingRate = 0.19 + Math.random() * 0.02;
        finalFeatures.alphaPower = 14;
        finalFeatures.betaPower = 67;
        finalFeatures.thetaPower = 19;
      } else {
        finalFeatures.variance = 82 + Math.random() * 10;
        finalFeatures.amplitudeRange = 38 + Math.random() * 5;
        finalFeatures.zeroCrossingRate = 0.09 + Math.random() * 0.02;
        finalFeatures.alphaPower = 71;
        finalFeatures.betaPower = 9;
        finalFeatures.thetaPower = 20;
      }
    }

    setCurrentBciFeatures(finalFeatures);

    // 2. Chạy dự báo nhãn điện sinh học tương ứng
    try {
      let finalLabel = "RELAX";
      let finalConfidence = 0.5;
      let isFallbackTrig = false;

      let mlLabel = "RELAX";
      let mlConfidence = 0.5;

      if (selectedClassifier === "KNN") {
        const knn = new KnnClassifier(knnK);
        knn.train(trainingData);
        const res = knn.predict(finalFeatures);
        mlLabel = res.label;
        mlConfidence = res.confidence;
        setKnnNeighbors(res.distances || []);
      } else if (selectedClassifier === "RF") {
        const rf = new RandomForestClassifier(rfTrees, 4, 4);
        rf.train(trainingData);
        const res = rf.predict(finalFeatures);
        mlLabel = res.label;
        mlConfidence = res.confidence;
        setRfVotes(res.votes || {});
      } else {
        const svm = new SvmClassifier(80, 0.05, 0.01);
        svm.train(trainingData);
        const res = svm.predict(finalFeatures, trainingData);
        mlLabel = res.label;
        mlConfidence = res.confidence;
        setSvmScores(res.scores || {});
      }

      const lowConfidence = mlConfidence < (fallbackThreshold / 100);
      const isForced = forceFallbackMode;
      const insufficientData = trainingData.length < 3;

      if (isFallbackEnabled && (lowConfidence || isForced || insufficientData)) {
        isFallbackTrig = true;
        
        if (gazeDirection === "BLINK" || isBlinking) {
          finalLabel = "BLINK";
          finalConfidence = 1.0;
        } else if (gazeDirection === "LEFT" || gazeStatus === "LEFT") {
          finalLabel = "LEFT";
          finalConfidence = 0.85;
        } else if (gazeDirection === "RIGHT" || gazeStatus === "RIGHT") {
          finalLabel = "RIGHT";
          finalConfidence = 0.85;
        } else if (attentionLevel > attentionThreshold) {
          finalLabel = "FOCUS";
          finalConfidence = 0.90;
        } else {
          finalLabel = "RELAX";
          finalConfidence = 0.95;
        }
      } else {
        finalLabel = mlLabel;
        finalConfidence = mlConfidence;
      }

      setPredictedBciLabel(finalLabel);
      setPredictedBciConfidence(finalConfidence);
      setIsFallbackActive(isFallbackTrig);
    } catch (err) {
      console.error("Lỗi phân lớp BCI:", err);
    }
  }, [
    attentionLevel,
    isBlinking,
    gazeDirection,
    gazeStatus,
    selectedClassifier,
    trainingData,
    knnK,
    rfTrees,
    isFallbackEnabled,
    fallbackThreshold,
    forceFallbackMode,
    attentionThreshold
  ]);

  // --- TRÌNH XỬ LÝ QUÉT BÀN PHÍM TỰ ĐỘNG ---
  useEffect(() => {
    if (!isAutoScanActive) return;

    const speed = scanMode === "ROW" ? scanSpeedRow : scanSpeedCol;
    const interval = setInterval(() => {
      if (scanMode === "ROW") {
        setCurrentRow((prev) => (prev + 1) % TOTAL_SCAN_ROWS);
      } else {
        const maxCols = currentRow === SUGGESTIONS_ROW_INDEX ? (suggestions.length || 5) : (GRID[currentRow]?.length || 6);
        setCurrentCol((prev) => (prev + 1) % maxCols);
      }
    }, speed);

    return () => clearInterval(interval);
  }, [isAutoScanActive, scanMode, scanSpeedRow, scanSpeedCol, currentRow, suggestions.length, TOTAL_SCAN_ROWS, SUGGESTIONS_ROW_INDEX]);

  // --- HỆ THỐNG BI-DIRECTIONAL STATE SYNCHRONISATION (Issue 1 & 2) ---
  // A. Đồng bộ từ máy chủ về giao diện (250ms Poll baselines)
  useEffect(() => {
    let active = true;
    const pollServerState = async () => {
      try {
        const response = await fetch("/api/state");
        if (!response.ok) return;
        const data = await response.json();
        if (!active) return;

        // Đồng bộ dữ liệu văn bản từ Backend là nguồn dữ liệu duy nhất (Issue Sửa số 2)
        // Độ trễ đồng bộ tối ưu giảm xuống còn 400ms để đảm bảo mượt mà hơn
        const delayGap = Date.now() - lastLocalWriteRef.current;
        if (delayGap > 400 && data.text !== undefined && data.text !== typedText) {
          setTypedText(data.text);
          setCursorPosition(data.text.length);
        }

        // Đọc tín hiệu Blink từ dự đoán Phần cứng / Server ảo
        if (data.prediction && data.prediction.blink_detected && !data.prediction.processed) {
          pushEvent("BLINK", { isManual: false });
          // Đánh dấu đã nhận để tránh trùng lặp gõ nhầm (Debounce & processed handshaking)
          fetch("/api/state", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              prediction: { ...data.prediction, processed: true }
            })
          }).catch(() => {});
        }

        if (!isAutoScanActive) {
          if (data.cursor_row !== undefined && data.cursor_row !== currentRow) {
            setCurrentRow(data.cursor_row);
          }
          if (data.cursor_col !== undefined && data.cursor_col !== currentCol) {
            setCurrentCol(data.cursor_col);
          }
        }

        if (data.scan_mode !== undefined && data.scan_mode !== scanMode) {
          setScanMode(data.scan_mode);
        }
        if (data.system_mode !== undefined && data.system_mode !== systemMode) {
          setSystemMode(data.system_mode);
        }
        if (data.attention_level !== undefined && data.attention_level !== attentionLevel) {
          setAttentionLevel(data.attention_level);
        }
        if (data.electrode_contact !== undefined && data.electrode_contact !== electrodeContact) {
          setElectrodeContact(data.electrode_contact);
        }
      } catch (err) {
        // Safe failover for simulated standalone mode
      }
    };

    const pollInterval = setInterval(pollServerState, 250); // MED-OPTIMAL THRESHOLD
    return () => {
      active = false;
      clearInterval(pollInterval);
    };
  }, [typedText, isAutoScanActive, currentRow, currentCol, scanMode, systemMode, attentionLevel, electrodeContact]);

  // C. ĐỒNG BỘ DỮ LIỆU THỰC TỪ PHẦN CỨNG (WebSocket -> Python Flask)
  // Phục vụ đúng luồng: [AD8232] → [ADS1115] → [ESP32] → [WiFi] → [Python Flask] → [WebSocket] → [Web Browser]
  useEffect(() => {
    // Thay cổng 5000 bằng cổng Flask server thực tế của bạn
    const WS_URL = "ws://localhost:5000/bci-stream"; 
    let ws: WebSocket | null = null;
    let reconnectTimer: NodeJS.Timeout;

    const connectWebSocket = () => {
      try {
        ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {
          console.log("✅ Đã kết nối với Python Flask WebSocket");
          showToast("Đã kết nối với phần cứng BCI (Flask)", "success");
          setElectrodeContact("GOOD");
        };

        ws.onmessage = (event) => {
          try {
            const sensorData = JSON.parse(event.data);
            
            // 1. NHẬN DỮ LIỆU RAW (Từ ADS1115) & VẼ LÊN CANVAS
            if (sensorData.eeg_raw !== undefined && sensorData.eog_raw !== undefined) {
              // Cập nhật bộ đệm để vẽ biểu đồ cho eegCanvasRef và eogCanvasRef
              // Ví dụ: addDataToBuffer(sensorData.eeg_raw)
            }

            // 2. NHẬN KẾT QUẢ ĐÃ LỌC / DỰ ĐOÁN TỪ PYTHON
            if (sensorData.gaze_direction) {
               // Python đã dự đoán hướng mắt (LEFT, RIGHT, CENTER, BLINK)
               if (sensorData.gaze_direction === "BLINK") {
                 setIsBlinking(true);
                 setTimeout(() => setIsBlinking(false), 300);
               } else {
                 setGazeDirection(sensorData.gaze_direction);
               }
            }

            if (sensorData.attention_level !== undefined) {
               // Python tính toán Bandpass, Alpha/Beta ratio và trả về
               setAttentionLevel(sensorData.attention_level);
            }
          } catch (e) {
            console.error("Lỗi parse dữ liệu WebSocket:", e);
          }
        };

        ws.onclose = () => {
          console.log("❌ Mất kết nối phần cứng. Đang thử lại...");
          setElectrodeContact("POOR");
          reconnectTimer = setTimeout(connectWebSocket, 3000); // Tự động kết nối lại
        };
      } catch (err) {
        console.error("Lỗi khởi tạo JS WebSocket", err);
      }
    };

    // Bỏ comment dòng dưới để kích hoạt kết nối WebSocket thực tế khi chạy với Flask
    // connectWebSocket();

    return () => {
      if (ws) ws.close();
      clearTimeout(reconnectTimer);
    };
  }, []);

  // B. Đẩy đồng bộ từ giao diện lên máy chủ khi có thay đổi nội bộ
  useEffect(() => {
    const elapsedSinceWrite = Date.now() - lastLocalWriteRef.current;
    if (elapsedSinceWrite < 1500) {
      const serverPushDelay = setTimeout(() => {
        fetch("/api/state", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: typedText,
            cursor_row: currentRow,
            cursor_col: currentCol,
            scan_mode: scanMode,
            system_mode: systemMode,
            attention_level: attentionLevel,
            electrode_contact: electrodeContact,
            prediction: {
              label: predictedBciLabel,
              confidence: predictedBciConfidence,
              system_mode: systemMode
            }
          })
        }).catch(() => {});
      }, 80);
      return () => clearTimeout(serverPushDelay);
    }
  }, [typedText, currentRow, currentCol, scanMode, systemMode, attentionLevel, electrodeContact, predictedBciLabel, predictedBciConfidence]);

  // --- BẬT ĐẾM NGƯỢC KHÓA CHỐNG DỘI TÍN HIỆU (Issue 5 - 900ms Lockout) ---
  useEffect(() => {
    if (!refractoryActive) return;
    const refTimer = setInterval(() => {
      setRefractoryTimeLeft((prev) => {
        if (prev <= 100) {
          setRefractoryActive(false);
          clearInterval(refTimer);
          return 0;
        }
        return prev - 100;
      });
    }, 100);
    return () => clearInterval(refTimer);
  }, [refractoryActive]);

  // --- ĐO ĐẠC CHỈ SỐ LÂM SÀNG BCI / NGHIÊN CỨU KHOA HỌC (WPM, CPM, ERROR, ITR) ---
  useEffect(() => {
    if (!typedText) {
      setCpm(0);
      setWpm(0);
      setItr(0);
      return;
    }

    let originTime = typingStartTime;
    if (originTime === null) {
      originTime = Date.now();
      setTypingStartTime(originTime);
    }

    const elapsedMinutes = (Date.now() - originTime) / 60000;
    if (elapsedMinutes > 0.008) {
      const computedCpm = Math.round(typedText.length / elapsedMinutes);
      const computedWpm = parseFloat((computedCpm / 5).toFixed(1));
      setCpm(computedCpm);
      setWpm(computedWpm);

      // Thước đo ITR học thuật (Information Transfer Rate) tinh lọc cho 26 chữ cái & 10 số (N=36)
      const totalSteps = typedText.length + errorsCount;
      const accuracyRate = totalSteps > 0 ? typedText.length / totalSteps : 1.0;

      let bVal = 0;
      if (accuracyRate > 0.05) {
        const symbolAlphabetSize = 36;
        const partAlphabet = Math.log2(symbolAlphabetSize);
        const partAcc = accuracyRate * Math.log2(accuracyRate);
        const partErr = (1 - accuracyRate) > 0 
          ? (1 - accuracyRate) * Math.log2((1 - accuracyRate) / (symbolAlphabetSize - 1)) 
          : 0;
        bVal = Math.max(0, (partAlphabet + partAcc + partErr) * computedCpm);
      }
      setItr(parseFloat(bVal.toFixed(1)));
    }
  }, [typedText, errorsCount, typingStartTime]);

  // --- HOẠT ĐỘNG PHÁN QUYẾT EYE TRACKING ĐÃ ĐƯỢC CHUYỂN HOÀN TOÀN SANG BỘ LỌC EOG/EEG CỦA PHẦN CỨNG ---
  useEffect(() => {
    // Không sử dụng bộ phân tích dựa trên camera nữa. Dự đoán được truyền qua WebSocket/Phần cứng hoặc bàn phím mô phỏng.
  }, []);

  // --- RENDERING HOẠT ĐỘNG SÓNG SINH HỌC ---
  useEffect(() => {
    let animationId: number;

    const draw = () => {
      signalPhaseRef.current += 0.05;

      const eogCanvas = eogCanvasRef.current;
      if (eogCanvas) {
        const ctx = eogCanvas.getContext('2d');
        if (ctx) {
          const w = eogCanvas.width;
          const h = eogCanvas.height;
          ctx.clearRect(0, 0, w, h);
          
          ctx.strokeStyle = '#111827';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, h / 2);
          ctx.lineTo(w, h / 2);
          ctx.stroke();

          // Lấy mẫu tín hiệu thô EOG thô (Có nhiễu cơ EMG và nguồn điện 50Hz) (Issue 3)
          const rawEog = [];
          for (let x = 0; x < w; x++) {
            let yNoise = Math.sin(x * 0.08 + signalPhaseRef.current) * 4;
            
            if (gazeDirection === 'LEFT') {
              yNoise -= 15 * Math.sin((x / w) * Math.PI);
            } else if (gazeDirection === 'RIGHT') {
              yNoise += 15 * Math.sin((x / w) * Math.PI);
            }

            if (isBlinking) {
              const peakPos = w / 2;
              const dist = Math.abs(x - peakPos);
              if (dist < 20) {
                yNoise -= (50 - dist * 2.5) * Math.sin((dist / 20) * Math.PI + Math.PI / 2);
              }
            }
            
            // Tạo nhiễu điện cơ EMG thô và nhiễu 50Hz hum
            const emgNoise = (Math.random() - 0.5) * 5.5;
            const powerlineNoise = Math.sin(x * 1.25) * 3.5;
            rawEog.push(yNoise + emgNoise + powerlineNoise);
          }

          // Áp dụng bộ lọc phần mềm DSP thông dải 0.5Hz - 40Hz và cắt tần Notch 50Hz
          const notchedEog = applyNotchFilter50Hz(rawEog);
          const filteredEog = applyBandpassFilter(notchedEog, 0.1, 30);

          // Vẽ tín hiệu thô EOG (Đỏ nhạt, biểu lộ nhiễu)
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.25)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let x = 0; x < w; x++) {
            const y = h / 2 + rawEog[x];
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();

          // Vẽ tín hiệu EOG sạch sau xử lý (Xanh dương nhạt nổi bật)
          ctx.strokeStyle = '#00f3ff';
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          for (let x = 0; x < w; x++) {
            const y = h / 2 + filteredEog[x];
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }

      const eegCanvas = eegCanvasRef.current;
      if (eegCanvas) {
        const ctx = eegCanvas.getContext('2d');
        if (ctx) {
          const w = eegCanvas.width;
          const h = eegCanvas.height;
          ctx.clearRect(0, 0, w, h);

          ctx.strokeStyle = '#111827';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, h / 2);
          ctx.lineTo(w, h / 2);
          ctx.stroke();

          // Lấy mẫu EEG thô (Có sóng hài 50Hz nhiễu điện lưới bọc ngoài)
          const rawEeg = [];
          const scale = attentionLevel > 50 ? 1.5 : 0.6;
          for (let x = 0; x < w; x++) {
            const wave1 = Math.sin(x * 0.15 + signalPhaseRef.current * 2) * 5 * scale;
            const wave2 = Math.sin(x * 0.35 - signalPhaseRef.current * 4) * 2 * scale;
            const wave3 = Math.cos(x * 0.05 + signalPhaseRef.current) * 8 * (1 - scale/3);
            
            // Nhiễu nguồn 50Hz và DC baseline drift trôi dạt chậm
            const powerlineNoise = Math.sin(x * 1.25) * 4.5;
            const dcDrift = Math.sin(x * 0.01) * 5.0;
            rawEeg.push(wave1 + wave2 + wave3 + powerlineNoise + dcDrift);
          }

          // Áp dụng bộ lọc phần mềm DSP lọc dải và chặn dải 50Hz
          const notchedEeg = applyNotchFilter50Hz(rawEeg);
          const filteredEeg = applyBandpassFilter(notchedEeg, 0.5, 40);

          // Vẽ tín hiệu thô EEG (Đỏ nhạt nhiễu bám)
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.25)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let x = 0; x < w; x++) {
            const y = h / 2 + rawEeg[x];
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();

          // Vẽ tín hiệu sạch EEG (Màu ngọc lục bảo Emerald)
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          for (let x = 0; x < w; x++) {
            const y = h / 2 + filteredEeg[x];
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }

      animationId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationId);
  }, [gazeDirection, isBlinking, attentionLevel]);

  // --- KÍCH HOẠT LỰA CHỌN GÕ PHÍM ---
  const handleSelectAction = () => {
    // Ngăn chặn dội tín hiệu (Issue 5 - Double blink hardware refractory protection)
    if (refractoryActive) {
      showToast("🔒 Khóa dội tín hiệu (Refractory Lock) đang hoạt động!", "info");
      return;
    }

    // Thiết lập khóa chống dội thời gian hiệu dụng 900ms
    setRefractoryActive(true);
    setRefractoryTimeLeft(900);
    setTotalSelections((prev) => prev + 1);
    lastLocalWriteRef.current = Date.now();

    setIsBlinking(true);
    setTimeout(() => setIsBlinking(false), 300);

    if (scanMode === "ROW") {
      setScanMode("COL");
      setCurrentCol(0);
    } else {
      if (currentRow === SUGGESTIONS_ROW_INDEX) {
        const word = suggestions[currentCol];
        if (word) {
          handleSuggestionClick(word);
        }
      } else {
        const char = GRID[currentRow]?.[currentCol];
        if (char === "⌫") {
          handleDeleteAction();
        } else if (char === "␣") {
          handleSpaceAction();
        } else if (char === "↺ RESET") {
          handleResetCursorAction();
        } else if (char === "🗑️ CLEAR") {
          handleClearAction();
        } else if (char === "←") {
          handleMoveCursorLeft();
        } else if (char === "→") {
          handleMoveCursorRight();
        } else if (char) {
          handleTypeCharAction(char);
        }
      }
      setScanMode("ROW");
    }
  };

  // Keyboard layout hotkeys
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        pushEvent("BLINK", { isManual: true });
      } else if (e.code === "Backspace") {
        e.preventDefault();
        handleDeleteAction();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        setGazeDirection("LEFT");
        setTimeout(() => setGazeDirection("CENTER"), 800);
        if (scanMode === "ROW") {
          setCurrentRow((prev) => (prev - 1 + TOTAL_SCAN_ROWS) % TOTAL_SCAN_ROWS);
        } else {
          const maxCols = currentRow === SUGGESTIONS_ROW_INDEX ? (suggestions.length || 5) : (GRID[currentRow]?.length || 6);
          setCurrentCol((prev) => (prev - 1 + maxCols) % maxCols);
        }
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        setGazeDirection("RIGHT");
        setTimeout(() => setGazeDirection("CENTER"), 800);
        if (scanMode === "ROW") {
          setCurrentRow((prev) => (prev + 1) % TOTAL_SCAN_ROWS);
        } else {
          const maxCols = currentRow === SUGGESTIONS_ROW_INDEX ? (suggestions.length || 5) : (GRID[currentRow]?.length || 6);
          setCurrentCol((prev) => (prev + 1) % maxCols);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [scanMode, currentRow, currentCol, suggestions.length, TOTAL_SCAN_ROWS, SUGGESTIONS_ROW_INDEX]);

  // Bộ sao chép mã code
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCodeCopied(true);
    setTimeout(() => setCodeCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-cyber-bg text-slate-100 flex flex-col font-sans selection:bg-cyber-accent selection:text-black">
      
      {/* HEADER ĐIỀU HÀNH */}
      <header className="border-b border-slate-900 bg-cyber-panel/80 backdrop-blur px-6 py-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-cyan-950/50 rounded-xl border border-cyan-500/30 text-cyber-accent shadow-[0_0_15px_rgba(0,243,255,0.15)]">
            <Cpu className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-cyber-accent to-blue-400">
                NEUROTYPE SYSTEM
              </h1>
              <span className="text-[9px] bg-cyan-950 text-cyber-accent border border-cyan-500/30 px-2 py-0.5 rounded-full font-mono">
                v1.0-Hybrid
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5 max-w-xl leading-normal">
              Hệ thống sử dụng tín hiệu điện mắt (EOG) làm kênh điều khiển chính, kết hợp tín hiệu EEG vùng trán để hỗ trợ xác thực và giảm lỗi chọn nhầm.
            </p>
          </div>
        </div>

        {/* Chuyển đổi TAB UI */}
        <div className="flex items-center gap-2 w-full md:w-auto">
          <button
            onClick={() => setActiveUIPanel("DASHBOARD")}
            className={`flex-1 md:flex-none flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold cursor-pointer border transition-all ${
              activeUIPanel === "DASHBOARD"
                ? "bg-cyber-accent text-slate-950 border-cyber-accent shadow-[0_0_12px_rgba(0,243,255,0.3)] font-bold"
                : "bg-slate-950 border-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            <Layout className="w-3.5 h-3.5" />
            BÀN PHÍM CO-WRITING
          </button>
          
          <button
            onClick={() => setActiveUIPanel("CODE_EXPLORER")}
            className={`flex-1 md:flex-none flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold cursor-pointer border transition-all ${
              activeUIPanel === "CODE_EXPLORER"
                ? "bg-cyber-accent text-slate-950 border-cyber-accent shadow-[0_0_12px_rgba(0,243,255,0.3)] font-bold"
                : "bg-slate-950 border-slate-800 text-slate-400 hover:text-white"
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            FILE MÃ NGUỒN ({Object.keys(SOURCE_FILES).length})
          </button>
        </div>
      </header>

      {/* DASHBOARD HOẠT ĐỘNG CHÍNH */}
      <main className="flex-1 w-full px-4 md:px-6 py-4 md:py-6 grid grid-cols-1 gap-6">
        
        {activeUIPanel === "DASHBOARD" ? (
          <div className="grid grid-cols-1 gap-6">
            
            {/* THÀNH PHẦN 1: PANEL TỔNG QUAN SK STATUS (HCI View) */}
            <div className="bg-slate-900/40 border border-slate-900 rounded-xl p-4 px-6 flex flex-wrap items-center justify-between gap-4 font-sans text-sm shadow-sm max-w-4xl mx-auto w-full animate-fadeIn">
              <div className="flex items-center gap-3">
                <span className="flex h-3 w-3 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-400 font-bold tracking-wider text-xs">HỆ THỐNG SẴN SÀNG</span>
              </div>

              {/* Trạng thái tín hiệu tinh giản theo yêu cầu tối ưu giáo trình */}
              <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono font-bold">
                <span className="text-emerald-400 bg-emerald-950/20 px-2.5 py-1 rounded border border-emerald-900/30 flex items-center gap-1.5 shadow-inner">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  EEG Ready
                </span>
                <span className="text-emerald-400 bg-emerald-950/20 px-2.5 py-1 rounded border border-emerald-900/30 flex items-center gap-1.5 shadow-inner">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  EOG Ready
                </span>
                <span className="text-cyan-400 bg-cyan-950/20 px-2.5 py-1 rounded border border-cyan-900/30 flex items-center gap-1.5 shadow-inner">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                  Calibration Complete
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-slate-300">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs">Chế độ: <strong className="text-white bg-slate-800 px-2 py-0.5 rounded text-[11px]">{systemMode === "HYBRID" ? "EOG + EEG" : "EOG Fallback"}</strong></span>
                </div>
              </div>
            </div>

            {/* THÀNH PHẦN 2: BỐ CỤC CHUYÊN BIỆT */}
            <div className="flex flex-col gap-6 items-center w-full max-w-4xl mx-auto">
              
              {/* --- MÀN HÌNH CHÍNH (TEXT AREA & MATRIX) --- */}
              <div className="w-full flex flex-col gap-6">
                
                {/* 1. TEXT AREA / THƯ VIỆN GÕ TEXT */}
                <div className="bg-cyber-panel border border-slate-900 rounded-xl p-5 shadow-lg flex flex-col gap-4">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] font-mono text-slate-400 tracking-widest uppercase">
                      🖥️ VĂN BẢN ĐÃ GÕ (CO-WRITING OUTPUT)
                    </span>
                    <div className="flex gap-2">
                       <button
                        onClick={handleClearAction}
                        className="p-1 px-2.5 bg-slate-950/80 hover:bg-rose-950 border border-slate-800 text-rose-400 rounded text-[9px] font-mono font-bold uppercase cursor-pointer"
                        title="Xóa toàn bộ văn bản"
                      >
                        Xóa sạch
                      </button>
                      <button
                        onClick={() => copyToClipboard(typedText)}
                        className="p-1 px-2.5 bg-slate-950/80 hover:bg-cyan-950 border border-slate-800 text-cyber-accent rounded text-[9px] font-mono font-bold uppercase flex items-center gap-1 cursor-pointer"
                      >
                        <Copy className="w-2.5 h-2.5" />
                        Sao Chép
                      </button>
                    </div>
                  </div>

                  {/* Ô hiển thị văn bản */}
                  <div className="relative bg-slate-950 border border-slate-900/60 rounded-xl p-4.5 min-h-[90px] shadow-inner select-all font-sans">
                    {typedText ? (
                      <p className="text-base font-medium leading-relaxed tracking-wide text-cyan-50">
                        {typedText.slice(0, cursorPosition)}
                        <span className="animate-pulse font-extrabold text-cyber-accent bg-cyan-700/50 px-[1px] ml-[1px]">|</span>
                        {typedText.slice(cursorPosition)}
                      </p>
                    ) : (
                      <div className="text-slate-500 text-xs py-1.5 space-y-1 font-mono leading-relaxed">
                        <p className="font-bold text-slate-300">Hệ thống đang quét bàn phím.</p>
                        <ul className="list-disc pl-4 space-y-1 text-[11px] text-slate-400">
                          <li>Chớp mắt để xác nhận lựa chọn</li>
                          <li>Liếc trái/phải để điều hướng</li>
                          <li>Space dùng cho chế độ mô phỏng</li>
                        </ul>
                      </div>
                    )}
                  </div>
                  
                  {/* WORD PREDICTIVE ROW */}
                  <div className="space-y-1 pt-1">
                    <div className="text-[9px] font-mono text-slate-500 uppercase tracking-widest">
                      Gợi Ý Từ Tiếng Việt (Vietnamese Predictive Text):
                    </div>
                    <div className="grid grid-cols-5 gap-1.5 font-sans">
                      {suggestions.map((word, idx) => {
                        const isSelectedRow = currentRow === SUGGESTIONS_ROW_INDEX;
                        const isSelectedCol = isSelectedRow && currentCol === idx;
                        return (
                          <button
                            key={idx}
                            onClick={() => handleSuggestionClick(word)}
                            className={`
                              py-2 rounded-lg text-xs font-semibold tracking-wide border cursor-pointer select-none transition-all duration-100 uppercase text-center truncate
                              ${isSelectedRow
                                ? isSelectedCol
                                  ? 'bg-amber-400 border-amber-400 text-black font-extrabold scale-[1.03] shadow-[0_0_10px_rgba(251,191,36,0.5)]'
                                  : 'bg-amber-950/40 border-amber-900/50 text-amber-300 font-bold'
                                : 'bg-slate-950 border-slate-900 text-slate-400 hover:text-white'
                              }
                            `}
                          >
                            {word}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* 2. MATRIX SPELLER KEYBOARD CARD */}
                <div className="bg-cyber-panel border border-slate-900 rounded-xl p-5 shadow-lg flex flex-col gap-4">
                  <div className="flex justify-between items-center border-b border-slate-950 pb-2">
                    <div className="flex items-center gap-2">
                      <Layout className="w-4 h-4 text-cyber-accent" />
                      <h3 className="text-xs font-bold tracking-wider uppercase font-mono text-slate-300">
                        MÀN HÌNH MA TRẬN PHÍM BCI (6x6 Matrix Speller)
                      </h3>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => {
                          setIsAutoScanActive(!isAutoScanActive);
                          showToast(
                            !isAutoScanActive
                              ? "Kích hoạt Quét tự động (Auto Scanning)"
                              : "Đã chuyển sang Quét thủ công (Manual Scanning). Hãy dùng các mũi tên hoặc nút liếc để dịch chuyển dòng/cột.",
                            "info"
                          );
                        }}
                        className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded border transition-all cursor-pointer ${
                          isAutoScanActive
                            ? "bg-cyan-950/45 text-cyan-400 border-cyan-800 hover:bg-cyan-900/50"
                            : "bg-red-950/30 text-rose-400 border-rose-900 hover:bg-rose-900/20"
                        }`}
                      >
                        {isAutoScanActive ? "🟢 AUTO SCAN" : "⚪ MANUAL SCAN"}
                      </button>
                      <span className="text-[9px] font-mono text-amber-400 bg-amber-950/30 border border-amber-900/50 px-2 py-0.5 rounded font-bold uppercase">
                        Mode: {scanMode === "ROW" ? "Quét Dòng" : "Quét Cột"}
                      </span>
                    </div>
                  </div>

                  {/* Grid 6x6 Ma trận */}
                  <div className="grid grid-cols-6 gap-2">
                    {GRID.map((rowArr, rIdx) => {
                      const isRowScanned = scanMode === "ROW" && currentRow === rIdx;
                      const isColScanned = scanMode === "COL" && currentRow === rIdx;

                      return rowArr.map((char, cIdx) => {
                        const isElementScanned = (scanMode === "ROW" && isRowScanned) || (scanMode === "COL" && isColScanned && currentCol === cIdx);
                        const isSelectedElement = (currentRow === rIdx && currentCol === cIdx);

                        if (char === "") return <div key={`empty-${rIdx}-${cIdx}`} className="bg-transparent" />;

                        // Tính màu phím theo chức năng đặc biệt
                        const isSpecial = ["⌫", "␣", "↺ RESET", "🗑️ CLEAR", "←", "→"].includes(char);
                        let fontClass = "text-xs font-bold font-mono";
                        if (char.length > 1) fontClass = "text-[9px] font-extrabold font-mono";

                        let btnTheme = "bg-slate-950/90 text-slate-300 border-slate-900 hover:text-white";
                        if (isSpecial) {
                          btnTheme = "bg-slate-900 shadow text-cyan-400 border-slate-800";
                        }

                        // Theme thay đổi nếu được quét trùng
                        if (isElementScanned) {
                          if (scanMode === "COL" && isSelectedElement) {
                            // Phím đang được nhắm tới cụ thể
                            btnTheme = "bg-cyber-accent border-cyber-accent text-slate-950 font-black scale-102 shadow-[0_0_12px_rgba(0,243,255,0.6)]";
                          } else {
                            // Toàn bộ dòng được quét
                            btnTheme = "bg-cyan-950/45 border-cyan-500/50 text-cyber-accent font-extrabold";
                          }
                        }

                        return (
                          <button
                            key={`${rIdx}-${cIdx}`}
                            onClick={() => {
                              setCurrentRow(rIdx);
                              setCurrentCol(cIdx);
                              setScanMode("ROW");
                              if (char === "⌫") {
                                handleDeleteAction();
                              } else if (char === "␣") {
                                handleSpaceAction();
                              } else if (char === "↺ RESET") {
                                handleResetCursorAction();
                              } else if (char === "🗑️ CLEAR") {
                                handleClearAction();
                              } else if (char === "←") {
                                handleMoveCursorLeft();
                              } else if (char === "→") {
                                handleMoveCursorRight();
                              } else if (char) {
                                handleTypeCharAction(char);
                              }
                            }}
                            className={`
                              py-3 rounded-lg border text-center transition-all duration-100 cursor-pointer min-h-[44px] flex items-center justify-center select-none
                              ${btnTheme} ${fontClass}
                            `}
                          >
                            {char}
                          </button>
                        );
                      });
                    })}
                  </div>

                  {/* THIẾT LẬP TẤT CẢ SPEED điều khiển */}
                  <div className="grid grid-cols-2 gap-4 border-t border-slate-950 pt-3 text-[10px] font-mono">
                    <div className="space-y-1">
                      <div className="flex justify-between text-slate-400">
                        <span>Nhịp quét dòng (Row speed):</span>
                        <span className="text-cyan-400 font-bold">{(scanSpeedRow/1000).toFixed(1)}s</span>
                      </div>
                      <input 
                        type="range" 
                        min="400" 
                        max="2000" 
                        step="100"
                        value={scanSpeedRow}
                        onChange={(e) => setScanSpeedRow(Number(e.target.value))}
                        className="w-full h-1 accent-cyber-accent bg-slate-950 cursor-pointer rounded"
                      />
                    </div>
                    <div className="space-y-1">
                      <div className="flex justify-between text-slate-400">
                        <span>Nhịp quét cột (Col speed):</span>
                        <span className="text-cyan-400 font-bold">{(scanSpeedCol/1000).toFixed(1)}s</span>
                      </div>
                      <input 
                        type="range" 
                        min="300" 
                        max="1500" 
                        step="100"
                        value={scanSpeedCol}
                        onChange={(e) => setScanSpeedCol(Number(e.target.value))}
                        className="w-full h-1 accent-cyber-accent bg-slate-950 cursor-pointer rounded"
                      />
                    </div>
                  </div>
                </div>

              </div>
              
              {/* --- ADVANCED SCIENTIFIC REPORT --- */}
              <details className="w-full bg-cyber-panel/30 border border-slate-900 rounded-xl shadow-lg mt-4 text-slate-400 font-mono text-[10px] [&_summary::-webkit-details-marker]:hidden">
                <summary className="flex items-center gap-2 p-4 cursor-pointer select-none border-b border-transparent hover:bg-slate-900/50 transition-colors rounded-xl font-bold uppercase tracking-widest text-[#00f3ff]">
                  <ChevronDown className="w-4 h-4" />
                  Advanced Scientific Report (Technical Details)
                </summary>
                <div className="p-4 pt-0 grid grid-cols-1 md:grid-cols-2 gap-6 border-t border-slate-900/50 mt-1">
                  
                  {/* CỘT TRÁI (KỸ THUẬT) */}
                  <div className="flex flex-col gap-4 pt-4">
                    <div className="flex items-center gap-2 border-b border-slate-950 pb-1.5">
                      <Activity className="w-4 h-4 text-emerald-400" />
                      <span className="font-bold uppercase tracking-wider text-slate-300">BIOPHYSICAL SIGNALS</span>
                    </div>
                    
                    {/* CHẢT SÓNG EEG */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center px-1">
                        <span className="text-[10px] text-slate-400 font-bold uppercase">Sóng Trán EEG</span>
                        <span className="text-[9px] text-cyber-green bg-[#10b981]/15 px-1.5 py-0.5 rounded font-extrabold text-emerald-400">
                          Focus: {attentionLevel}%
                        </span>
                      </div>
                      <div className="bg-slate-950 rounded p-1 border border-slate-900 relative">
                        <canvas ref={eegCanvasRef} width={250} height={70} className="w-full h-[70px]" />
                        <span className="absolute bottom-1 right-2 text-[8px] text-slate-600">Gain: 1.0x | Qual: 98%</span>
                      </div>
                    </div>

                    {/* CHẢT SÓNG EOG */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center px-1">
                        <span className="text-[10px] text-slate-400 font-bold uppercase">Xung Vận Mắt EOG</span>
                        <span className="text-[9px] text-cyber-accent bg-[#00f3ff]/15 px-1.5 py-0.5 rounded font-extrabold text-[#00f3ff]">
                          {isBlinking ? "BLINK Spike" : "Baseline"}
                        </span>
                      </div>
                      <div className="bg-slate-950 rounded p-1 border border-slate-900 relative">
                        <canvas ref={eogCanvasRef} width={250} height={70} className="w-full h-[70px]" />
                        <span className="absolute bottom-1 right-2 text-[8px] text-slate-600">Filter: 50Hz Notch</span>
                      </div>
                    </div>
                    
                    <div className="space-y-1 text-slate-500 leading-relaxed bg-slate-950/45 p-2 rounded border border-slate-900">
                      <div className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 bg-[#00f3ff] rounded-full" /> EOG: Motor channel (Vận động cơ mắt, chớp mắt/nhìn trái/phải)</div>
                      <div className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" /> EEG: Cognitive channel (Attention tracking)</div>
                      <div className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 bg-yellow-400 rounded-full" /> Impedance: {electrodeContact === "GOOD" ? "12kΩ - Khá tốt" : "45kΩ - Cần chỉnh"}</div>
                    </div>
                  </div>

                  {/* CỘT PHẢI (KỸ THUẬT) */}
                  <div className="flex flex-col gap-4 pt-4">
                
                {/* 1. AI DETECTION & CONFIDENCE VIEW */}
                <div className="bg-cyber-panel border border-slate-900 rounded-xl p-4 shadow-lg flex flex-col gap-3.5">
                  <div className="flex justify-between items-center border-b border-slate-950 pb-2">
                    <div className="flex items-center gap-1.5">
                      <Cpu className="w-4 h-4 text-amber-500 animate-pulse" />
                      <h3 className="text-xs font-bold font-mono uppercase text-slate-350">
                        AI CONFIDENCE & STATE
                      </h3>
                    </div>
                    <span className="text-[8px] font-mono text-emerald-400 bg-emerald-950/45 border border-emerald-950 px-1.5 py-0.5 rounded font-bold uppercase">
                      ONLINE
                    </span>
                  </div>

                  {/* SYSTEM MODE SELECTION */}
                  <div className="bg-slate-950/90 p-3 rounded-lg border border-slate-900 flex flex-col gap-1.5 font-mono">
                    <div className="text-[9px] text-slate-500 uppercase tracking-wider font-extrabold pb-0.5">
                      SYSTEM MODE
                    </div>
                    <div className="grid grid-cols-2 gap-1 mt-0.5">
                      <button
                        onClick={() => {
                          lastLocalWriteRef.current = Date.now();
                          setSystemMode("HYBRID");
                          setEegTriggerEnabled(true);
                          showToast("Đã kích hoạt chế độ HYBRID (EOG + EEG)", "success");
                        }}
                        className={`py-1.5 px-2 rounded text-[9px] font-bold text-left border cursor-pointer transition-all ${
                          systemMode === "HYBRID"
                            ? "bg-cyan-950 text-cyan-400 border-cyan-500 shadow-[0_0_8px_rgba(0,243,255,0.2)]"
                            : "bg-slate-950/80 border-slate-900 text-slate-500 hover:text-slate-300"
                        }`}
                      >
                        {systemMode === "HYBRID" ? "🟢" : "⚪"} EOG+EEG
                      </button>
                      <button
                        onClick={() => {
                          lastLocalWriteRef.current = Date.now();
                          setSystemMode("EOG_ONLY");
                          setEegTriggerEnabled(false);
                          showToast("Đã chuyển sang chế độ EOG Fallback", "info");
                        }}
                        className={`py-1.5 px-2 rounded text-[9px] font-bold text-left border cursor-pointer transition-all ${
                          systemMode === "EOG_ONLY"
                            ? "bg-amber-955 text-amber-500 border-amber-600 shadow-[0_0_8px_rgba(245,158,11,0.2)]"
                            : "bg-slate-950/80 border-slate-900 text-slate-500 hover:text-slate-300"
                        }`}
                      >
                        {systemMode === "EOG_ONLY" ? "🟢" : "⚪"} EOG Fallback
                      </button>
                    </div>
                  </div>

                  <div className="bg-slate-950/90 p-3 rounded-lg border border-slate-900 flex flex-col gap-2 font-mono text-[10px] text-slate-400 leading-relaxed">
                    <div className="flex justify-between border-b border-slate-950/40 pt-1 pb-1">
                      <span>CURRENT STATE:</span>
                      <strong className={`font-bold tracking-wider ${
                        predictedBciLabel === "FOCUS" ? "text-emerald-400 animate-pulse" :
                        predictedBciLabel === "RELAX" ? "text-cyan-400" :
                        predictedBciLabel === "BLINK" ? "text-pink-400 animate-pulse" :
                        predictedBciLabel === "LEFT" || predictedBciLabel === "RIGHT" ? "text-yellow-400 animate-pulse" : "text-slate-350"
                      }`}>
                        {predictedBciLabel}
                      </strong>
                    </div>

                    <div className="flex justify-between items-center border-b border-slate-950/40 pb-1">
                      <span>CONFIDENCE:</span>
                      <strong className="text-white font-bold">{(predictedBciConfidence * 100).toFixed(0)}%</strong>
                    </div>

                    <div className="w-full bg-slate-900 h-1 rounded-full overflow-hidden mt-1">
                      <div 
                        className={`h-full transition-all duration-350 ${
                          predictedBciLabel === "FOCUS" ? "bg-emerald-400" :
                          predictedBciLabel === "RELAX" ? "bg-cyan-400" :
                          predictedBciLabel === "BLINK" ? "bg-pink-400" : "bg-yellow-400"
                        }`}
                        style={{ width: `${predictedBciConfidence * 100}%` }}
                      />
                    </div>
                  </div>



                  {/* THỐNG KÊ LÂM SÀNG BCI / NCKH CORE METRICS */}
                  <div className="bg-slate-950/90 p-3 rounded-lg border border-slate-900 flex flex-col gap-2.5 font-mono">
                    <div className="text-[9px] text-slate-500 uppercase tracking-wider font-extrabold pb-0.5 border-b border-slate-900/60 flex justify-between">
                      <span>📊 CLINICAL WRITING METRICS</span>
                      <span className="text-cyan-400 text-[8px] font-bold">N=36 SYMBOLS</span>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div className="bg-slate-900/40 p-2 rounded border border-slate-950 flex flex-col">
                        <span className="text-[8px] text-slate-500">SPEED CPM:</span>
                        <strong className="text-cyan-400 text-sm font-bold tracking-tight">{cpm} <span className="text-[8px] text-slate-500 font-normal font-mono">c/min</span></strong>
                      </div>
                      <div className="bg-slate-900/40 p-2 rounded border border-slate-950 flex flex-col">
                        <span className="text-[8px] text-slate-500">SPEED WPM:</span>
                        <strong className="text-emerald-400 text-sm font-bold tracking-tight">{wpm} <span className="text-[8px] text-slate-500 font-normal font-mono">w/min</span></strong>
                      </div>
                      <div className="bg-slate-900/40 p-2 rounded border border-slate-950 flex flex-col">
                        <span className="text-[8px] text-slate-500">CORRECTIVE DEL:</span>
                        <span className="text-red-400 text-sm font-bold tracking-tight">{errorsCount} <span className="text-[8px] text-slate-500 font-normal font-mono font-bold">typos</span></span>
                      </div>
                      <div className="bg-slate-900/40 p-2 rounded border border-slate-950 flex flex-col">
                        <span className="text-[8px] text-slate-500">ITR SPECTRUM:</span>
                        <strong className="text-amber-500 text-sm font-bold tracking-tight">{itr} <span className="text-[8px] text-slate-500 font-normal font-mono">bits/m</span></strong>
                      </div>
                    </div>

                  </div>




                </div>

                {/* 2. CALIBRATION & USER PROFILE CARD */}
                <div className="bg-cyber-panel border border-slate-900 rounded-xl p-4 shadow-lg flex flex-col gap-3.5">
                  <div className="flex items-center justify-between border-b border-slate-950 pb-2">
                    <div className="flex items-center gap-1.5">
                      <User className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-xs font-bold font-mono uppercase text-slate-350">
                        USER PROFILE MANAGER
                      </h3>
                    </div>
                  </div>
                  
                  <div className="text-[10px] text-slate-500 font-mono italic leading-normal bg-slate-950 p-2.5 rounded border border-slate-900">
                    User Profile Manager. Save personal calibration settings.
                  </div>

                  {/* CHỌN HỒ SƠ */}
                  <div className="space-y-2 font-mono">
                    <div className="space-y-1">
                      <label className="text-[9px] text-slate-500 uppercase tracking-wider block">Chọn Hồ Sơ Hoạt Động:</label>
                      <div className="flex gap-1.5">
                        <select 
                          value={activeProfileId} 
                          onChange={(e) => {
                            const selectedId = e.target.value;
                            const prof = profiles.find(p => p.id === selectedId);
                            if (prof) {
                              setActiveProfileId(selectedId);
                              setBlinkThreshold(prof.blinkThreshold);
                              setAttentionThreshold(prof.attentionThreshold);
                              setLeftGain(prof.leftGain);
                              setRightGain(prof.rightGain);
                              setSystemMode(prof.systemMode);
                              setProfileName(prof.name);
                              if (prof.baselineEog !== undefined) setBaselineEog(prof.baselineEog);
                              if (prof.leftThreshold !== undefined) setGazeThresholdLeft(prof.leftThreshold);
                              if (prof.rightThreshold !== undefined) setGazeThresholdRight(prof.rightThreshold);
                            }
                          }}
                          className="flex-1 bg-slate-950 border border-slate-900 rounded p-1.5 text-xs text-white focus:outline-none focus:border-cyan-400 cursor-pointer"
                        >
                          {profiles.map(p => (
                            <option key={p.id} value={p.id}>{p.name} {["default", "high_sensitivity", "fast_blink", "eog_only_mode"].includes(p.id) ? "🔒" : "👤"}</option>
                          ))}
                        </select>
                        
                        {!["default", "high_sensitivity", "fast_blink", "eog_only_mode"].includes(activeProfileId) && (
                          <button
                            onClick={() => {
                              setProfiles(prev => prev.filter(p => p.id !== activeProfileId));
                              const def = profiles.find(p => p.id === "default")!;
                              setActiveProfileId("default");
                              setBlinkThreshold(def.blinkThreshold);
                              setAttentionThreshold(def.attentionThreshold);
                              setLeftGain(def.leftGain);
                              setRightGain(def.rightGain);
                              setSystemMode(def.systemMode);
                              setProfileName(def.name);
                              if (def.baselineEog !== undefined) setBaselineEog(def.baselineEog);
                              if (def.leftThreshold !== undefined) setGazeThresholdLeft(def.leftThreshold);
                              if (def.rightThreshold !== undefined) setGazeThresholdRight(def.rightThreshold);
                            }}
                            className="px-2 bg-rose-950/50 border border-rose-900 text-rose-400 hover:bg-rose-900 hover:text-white rounded text-[9px] font-bold cursor-pointer"
                          >
                            Xóa
                          </button>
                        )}
                      </div>
                    </div>

                    {/* ELECTRODE CONTACT DIAGNOSTICS */}
                    <div className="space-y-1">
                      <label className="text-[9px] text-slate-500 uppercase tracking-wider block">Electrode Connection Diagnostics:</label>
                      <div className="bg-slate-950 p-2.5 rounded border border-slate-900 font-mono text-[9px] leading-relaxed space-y-1">
                        <div className="flex justify-between items-center text-slate-400">
                          <span>Electrode Impedance:</span>
                          <span className={electrodeContact === "GOOD" ? "text-emerald-400 font-bold" : electrodeContact === "FAIR" ? "text-yellow-400 font-bold" : "text-rose-400 font-bold"}>
                            {electrodeContact === "GOOD" ? "4.2 kΩ (Pass)" : electrodeContact === "FAIR" ? "12.8 kΩ (Alert)" : "48.5 kΩ (High)"}
                          </span>
                        </div>
                        <div className="flex justify-between items-center border-t border-slate-900/30 pt-1 text-slate-400">
                          <span>Hardware Status:</span>
                          <span className={electrodeContact === "GOOD" ? "text-emerald-400 font-bold animate-pulse" : electrodeContact === "FAIR" ? "text-yellow-400" : "text-rose-500 font-bold"}>
                            {electrodeContact === "GOOD" ? "🟢 GOOD" : electrodeContact === "FAIR" ? "🟡 FAIR" : "🔴 POOR"}
                          </span>
                        </div>
                        <div className="flex justify-between items-center pt-0.5 text-slate-400">
                          <span>Signal Quality Index:</span>
                          <span className="text-white font-bold">{electrodeContact === "GOOD" ? "98%" : electrodeContact === "FAIR" ? "65%" : "20%"}</span>
                        </div>
                      </div>
                    </div>

                    {/* HIỆU CHỈNH GIAIN & THRESHOLD */}
                    <div className="grid grid-cols-2 gap-2 pt-1">
                      <div className="space-y-0.5">
                        <label className="text-[8px] text-slate-500 uppercase">Gain L:</label>
                        <input 
                          type="range" 
                          min="0.5" 
                          max="2.5" 
                          step="0.1"
                          value={leftGain}
                          onChange={(e) => setLeftGain(Number(e.target.value))}
                          className="w-full accent-cyan-400 bg-slate-950 h-1 cursor-pointer"
                        />
                        <span className="text-[8px] text-cyan-400 font-bold">{leftGain.toFixed(1)}x</span>
                      </div>
                      <div className="space-y-0.5">
                        <label className="text-[8px] text-slate-500 uppercase">Gain R:</label>
                        <input 
                          type="range" 
                          min="0.5" 
                          max="2.5" 
                          step="0.1"
                          value={rightGain}
                          onChange={(e) => setRightGain(Number(e.target.value))}
                          className="w-full accent-cyan-400 bg-slate-950 h-1 cursor-pointer"
                        />
                        <span className="text-[8px] text-cyan-400 font-bold">{rightGain.toFixed(1)}x</span>
                      </div>
                    </div>

                    <div className="space-y-1 mt-1">
                      <div className="flex justify-between text-[9px] text-slate-500">
                        <span>EOG Blink Threshold:</span>
                        <span>{blinkThreshold} (Relative)</span>
                      </div>
                      <input 
                        type="range" 
                        min="80" 
                        max="300" 
                        value={blinkThreshold}
                        onChange={(e) => setBlinkThreshold(Number(e.target.value))}
                        className="w-full accent-cyan-400 bg-slate-950 h-1 cursor-pointer"
                      />
                    </div>

                    <div className="mt-1 border-t border-slate-900/40 pt-1.5 mb-2">
                      <div className="space-y-0.5">
                        <label className="text-[8px] text-slate-500 uppercase block">Baseline EOG (μV):</label>
                        <input 
                          type="range" 
                          min="1" 
                          max="40" 
                          step="1"
                          value={baselineEog}
                          onChange={(e) => setBaselineEog(Number(e.target.value))}
                          className="w-full accent-cyan-450 bg-slate-950 h-1 cursor-pointer"
                        />
                        <span className="text-[8.5px] text-cyan-400 font-bold">{baselineEog} μV</span>
                      </div>
                    </div>

                    {/* LƯU TRỮ NEW PROFILE */}
                    <div className="space-y-1 pt-1.5 border-t border-slate-900">
                      <label className="text-[8px] text-slate-500 uppercase font-bold block">Tạo hồ sơ mới:</label>
                      <div className="flex gap-1">
                        <input 
                          type="text" 
                          placeholder="Tên profile..." 
                          value={newProfileNameInput}
                          onChange={(e) => setNewProfileNameInput(e.target.value)}
                          className="flex-1 bg-slate-950 border border-slate-900 rounded p-1 text-[10px] text-white focus:outline-none focus:border-cyan-400"
                        />
                        <button
                          onClick={() => {
                            if (!newProfileNameInput.trim()) {
                              showToast("Nhập tên hồ sơ!", "error");
                              return;
                            }
                            const newId = "custom_" + Date.now();
                            const newProf = {
                              id: newId,
                              name: newProfileNameInput.trim(),
                              blinkThreshold,
                              attentionThreshold,
                              leftGain,
                              rightGain,
                              systemMode,
                              description: `Profile của ${newProfileNameInput.trim()} lưu lúc ${new Date().toLocaleTimeString()}`,
                              baselineEog,
                              leftThreshold: gazeThresholdLeft,
                              rightThreshold: gazeThresholdRight,
                            };
                            setProfiles(prev => [...prev, newProf]);
                            setActiveProfileId(newId);
                            setProfileName(newProfileNameInput.trim());
                            showToast(`Đã lưu profile "${newProfileNameInput.trim()}"!`, "success");
                            setNewProfileNameInput("");
                          }}
                          className="p-1 px-2.5 bg-cyan-900 border border-cyan-700 text-cyan-200 text-[9px] font-bold uppercase rounded cursor-pointer transition hover:bg-cyan-800"
                        >
                          Lưu
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
                  
                  </div> {/* Close CỘT PHẢI */}
                </div> {/* Close Grid grid-cols-2 */}
                </details>

            </div> {/* Close THÀNH PHẦN 2 */}

          </div>
        ) : (
          /* TRÌNH KHÁM PHÁ FILE VÀ HOÀN THIỆN ĐẦU OUT */
          <div className="bg-cyber-panel border border-slate-900 rounded-xl p-5 shadow-lg flex flex-col gap-4">
            
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-slate-950 pb-4">
              <div>
                <h2 className="text-lg font-bold flex items-center gap-2">
                  <Terminal className="text-cyber-accent w-5 h-5" />
                  KHO FILE MÃ NGUỒN TOÀN DIỆN (8 FILES)
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  Đã định hình chính xác đầy đủ sườn tệp cấu trúc theo thiết kế hệ thống NeuroType của bạn.
                </p>
              </div>
              <div className="flex items-center gap-2 w-full md:w-auto">
                <button
                  onClick={() => copyToClipboard(SOURCE_FILES[activeCodeTab])}
                  className="bg-slate-950 border border-slate-900 hover:bg-slate-900 text-slate-300 text-xs px-4 py-2.5 rounded-lg flex items-center justify-center gap-2 transition cursor-pointer flex-1 md:flex-none font-bold font-mono"
                >
                  {codeCopied ? (
                    <>
                      <Check className="w-4 h-4 text-green-400" />
                      ĐÃ SAO CHÉP MÃ TỆP
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      SAO CHÉP MÃ FILE
                    </>
                  )}
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-start">
              
              {/* Danh sách tệp bên trái */}
              <div className="md:col-span-4 flex flex-col gap-1">
                <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider mb-2 px-1">Danh mục sườn thiết kế:</span>
                {Object.keys(SOURCE_FILES).map((fileName) => {
                  return (
                    <button
                      key={fileName}
                      onClick={() => {
                        setActiveCodeTab(fileName as keyof typeof SOURCE_FILES);
                        setCodeCopied(false);
                      }}
                      className={`
                        w-full text-left px-3 py-2.5 rounded-lg text-xs font-mono transition-all duration-150 flex items-center justify-between cursor-pointer border
                        ${activeCodeTab === fileName
                          ? 'bg-slate-900/80 text-cyber-accent font-bold border-cyan-900/50 shadow-inner'
                          : 'bg-slate-950 border-transparent text-slate-400 hover:text-white hover:bg-slate-950/80'
                        }
                      `}
                    >
                      <span className="flex items-center gap-2 truncate">
                        <FileText className={`w-3.5 h-3.5 ${activeCodeTab === fileName ? 'text-cyber-accent' : 'text-slate-500'}`} />
                        {fileName}
                      </span>
                      <span className="text-[9px] bg-slate-950 border border-slate-900 text-slate-600 px-1.5 py-0.5 rounded font-mono">
                        {fileName.endsWith('.py') ? 'Python' : fileName.endsWith('.ino') ? 'C++' : 'Config'}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Khối hiển thị dòng mã */}
              <div className="md:col-span-8">
                <div className="bg-slate-950 rounded-xl overflow-hidden border border-slate-900 shadow-inner relative flex flex-col">
                  
                  {/* File specs indicator */}
                  <div className="bg-slate-900 px-4 py-2 text-xs font-mono text-slate-500 flex justify-between items-center border-b border-slate-950 select-none">
                    <span>Path: /{activeCodeTab}</span>
                    <span>{SOURCE_FILES[activeCodeTab].split('\n').length} lines</span>
                  </div>

                  {/* Code Block Container */}
                  <div className="p-4 overflow-x-auto font-mono text-xs text-slate-300 leading-relaxed max-h-[500px] overflow-y-auto">
                    <pre className="whitespace-pre">{SOURCE_FILES[activeCodeTab]}</pre>
                  </div>
                </div>
              </div>

            </div>

          </div>
        )}

      </main>

      {/* FOOTER HỆ THỐNG */}
      <footer className="mt-auto border-t border-slate-950 py-5 bg-black/90 px-6 text-center select-none">
        <p className="text-[11px] text-slate-600 font-mono">
          © 2026 NEUROTYPE CO-WRITER · Thiết kế hoàn thiện cho đề tài BCI (EOG + EEG) · Bản quyền thuộc tổ chức nghiên cứu.
        </p>
      </footer>

      {/* FLOATING TOAST NOTIFICATION IFRAME-SAFE */}
      <AnimatePresence>
        {toastMessage && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-6 right-6 z-50 flex items-center gap-2.5 px-4.5 py-3 rounded-lg border shadow-2xl font-mono text-xs text-white max-w-sm backdrop-blur-md"
            style={{
              backgroundColor:
                toastMessage.type === "success"
                  ? "rgba(6, 78, 59, 0.95)"
                  : toastMessage.type === "error"
                  ? "rgba(153, 27, 27, 0.95)"
                  : "rgba(30, 58, 138, 0.95)",
              borderColor:
                toastMessage.type === "success"
                  ? "rgb(16, 185, 129)"
                  : toastMessage.type === "error"
                  ? "rgb(239, 68, 68)"
                  : "rgb(59, 130, 246)",
            }}
          >
            <span className="shrink-0">
              {toastMessage.type === "success" ? "🟢" : toastMessage.type === "error" ? "🔴" : "ℹ️"}
            </span>
            <span className="font-semibold leading-tight">{toastMessage.text}</span>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}
