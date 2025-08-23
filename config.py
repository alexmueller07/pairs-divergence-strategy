import os 

# === Alpaca ===
# Need .env file
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# === Discovery / Data ===
PERIOD = "1mo"    
INTERVAL = "5m"
CORRELATION_THRESHOLD = 0.85
MIN_VALID_RATIO = 0.50  
DOWNLOAD_THREADING = True

# === Live Trading Bars ===
LIVE_PERIOD_MIN = "2d"  
LIVE_INTERVAL   = "1m"  

# === Signal / Strategy Params ===
ROLLING_WINDOW = 60     
ENTRY_Z        = 2.0         
EXIT_Z         = 0.5  
COOLDOWN_BARS  = 10 

# === Risk / Sizing ===
MAX_OPEN_PAIRS            = 10  
PER_PAIR_MAX_DOLLAR       = 5_000
ACCOUNT_RISK_CAP_PCT      = 0.50 
MIN_RISK_PCT              = 0.01     
MAX_RISK_PCT              = 0.05 
SIGNAL_STRENGTH_CAP       = 2.0

# === Stop Time ===
EOD_FLATTEN_HHMM_EST = (15, 00)


LOG_EVERY_X_SECS = 60
