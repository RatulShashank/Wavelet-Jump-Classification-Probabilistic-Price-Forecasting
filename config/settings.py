from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    SYMBOL: str = "BTCUSDT"
    BINANCE_VISION_BASE: str = "https://data.binance.vision"
    BASE_URL: str = "https://fapi.binance.com"
    MAX_RETRIES: int = 3
    BACKOFF_BASE_MS: int = 1000
    # Range: April 2024 to March 2026 (Last 2 Years)
    TRAIN_START: str = "2024-04-01"
    TRAIN_END:   str = "2025-03-31"
    VAL_START:   str = "2025-04-01"
    VAL_END:     str = "2025-09-30"
    TEST_START:  str = "2025-10-01"
    TEST_END:    str = "2026-03-31"
    # Jump Detection
    JUMP_SIGMA_WINDOW: int = 30
    JUMP_THRESHOLD: float = 4.0
    JUMP_CLUSTER_GAP: int = 10
    # Wavelets
    WAVELET_J: int = 6
    WAVELET_NAME: str = 'db4'
    WAVELET_FEATURE_COUNT: int = 29
    # Microstructure
    CVD_SLOPE_WINDOW: int = 15
    # Volume Profile
    VOL_PROFILE_BINS: int = 100
    VOL_PROFILE_VA_PCT: float = 0.7
    # Regimes
    MACRO_HMM_STATES: int = 4
    MICRO_EFF_WINDOW: int = 15
    # Labeling Thresholds
    LABEL_ENDO_VOL_BUILDUP: float = 1.1
    LABEL_ENDO_SCAT_ASYM: float = 0.5
    LABEL_ANTI_VOL_BUILDUP: float = 1.2
    LABEL_ANTI_TREND_ALIGN: float = 0.5
    # Calibration
    TARGET_BRIER_SCORE: float = 0.2
    # Confirmed from prototype training set
    BUCKET_STRONG_UP:   float =  0.493
    BUCKET_MILD_UP:     float =  0.112
    BUCKET_FLAT_LO:     float = -0.131
    BUCKET_MILD_DOWN:   float = -0.512

CONFIG = Config()
