#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SANTO GRIAL – IQ Option BOT (PAPER)
Features: volume, MACD confluence, delta-atr dinâmico, outside-bar filter
Grava cada trade em trades.csv
"""
import os, time, threading, requests, csv
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    from backports.zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from iqoptionapi.stable_api import IQ_Option
from dotenv import load_dotenv

load_dotenv()

# ========================= CONFIG & ENV =========================
TZ = ZoneInfo("America/Sao_Paulo")

def env_bool(k, d=False):
    return str(os.getenv(k, str(d))).lower() in ("1","true","yes","y","on")

def env_list(k, sep=",", cast=str):
    v = os.getenv(k, "")
    return [cast(x.strip()) for x in v.split(sep) if x.strip()]

IQ_EMAIL        = os.getenv("IQ_EMAIL", "")
IQ_PASSWORD     = os.getenv("IQ_PASSWORD", "")
IQ_MODE         = os.getenv("IQ_MODE", "PRACTICE")

TELEGRAM_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHATID = os.getenv("TELEGRAM_CHAT_ID", "")

ASSETS          = env_list("ASSETS")
TIMEFRAMES      = env_list("TIMEFRAMES")
EXPIRATION_CANDLES = int(os.getenv("EXPIRATION_CANDLES", "1"))

SG_FAST         = int(os.getenv("SG_FAST", "1"))
SG_SLOW         = int(os.getenv("SG_SLOW", "34"))
SG_SIGNAL       = int(os.getenv("SG_SIGNAL", "5"))

ONLY_TREND      = env_bool("ONLY_TREND", True)
EMA_TREND       = int(os.getenv("EMA_TREND", "100"))

USE_BB_CONTEXT  = env_bool("USE_BB_CONTEXT", True)
SMA_BB          = int(os.getenv("SMA_BB", "20"))
BB_STD          = float(os.getenv("BB_STD", "3"))

COOLDOWN_SAME_BAR = env_bool("COOLDOWN_SAME_BAR", True)
VERBOSE         = env_bool("VERBOSE", False)
TICK_LOG_MINIMAL= env_bool("TICK_LOG_MINIMAL", True)
USE_MG1         = env_bool("USE_MG1", False)

ENABLE_TELEGRAM_COMMANDS = env_bool("ENABLE_TELEGRAM_COMMANDS", True)
POLL_INTERVAL_SEC        = int(os.getenv("POLL_INTERVAL_SEC", "2"))

# ---------- NOVOS FILTROS ----------
USE_MIN_BUFFER_DELTA   = env_bool("USE_MIN_BUFFER_DELTA", True)
MIN_BUFFER_DELTA       = float(os.getenv("MIN_BUFFER_DELTA", "0.00005"))

USE_BODY_CONFIRM       = env_bool("USE_BODY_CONFIRM", True)
MIN_BODY_PCT           = float(os.getenv("MIN_BODY_PCT", "40"))

USE_HTF_CONFLUENCE     = env_bool("USE_HTF_CONFLUENCE", True)
HTF_FOR_CONFLUENCE     = os.getenv("HTF_FOR_CONFLUENCE", "M5")

USE_ATR_FILTER         = env_bool("USE_ATR_FILTER", True)
ATR_PERIOD             = int(os.getenv("ATR_PERIOD", "14"))
MIN_ATR                = float(os.getenv("MIN_ATR", "0.0008"))

USE_CONSOLIDATION      = env_bool("USE_CONSOLIDATION", True)
CONS_LOOKBACK          = int(os.getenv("CONS_LOOKBACK", "10"))
CONS_BODY_PCT_MAX      = float(os.getenv("CONS_BODY_PCT_MAX", "35"))
CONS_REQUIRE_ALTERNATE = env_bool("CONS_REQUIRE_ALTERNATE", True)

SR_LOOKBACK            = int(os.getenv("SR_LOOKBACK", "50"))

USE_MIN_PAYOUT         = env_bool("USE_MIN_PAYOUT", False)
MIN_PAYOUT             = float(os.getenv("MIN_PAYOUT", "0.80"))

# 1) Volume
USE_VOLUME_CONFIRM   = env_bool("USE_VOLUME_CONFIRM", True)
VOLUME_PERIOD        = int(os.getenv("VOLUME_PERIOD", "20"))
VOLUME_STDDEV        = float(os.getenv("VOLUME_STDDEV", "1.0"))

# 3) MACD
USE_MACD_CONFLUENCE  = env_bool("USE_MACD_CONFLUENCE", True)
MACD_TF              = os.getenv("MACD_TF", "M5")
MACD_FAST            = int(os.getenv("MACD_FAST", "12"))
MACD_SLOW            = int(os.getenv("MACD_SLOW", "26"))
MACD_SIGNAL          = int(os.getenv("MACD_SIGNAL", "9"))

# 4) Delta-ATR dinâmico
MIN_BUFFER_DELTA_ATR_RATIO = float(os.getenv("MIN_BUFFER_DELTA_ATR_RATIO", "0.15"))

# 5) Outside-bar
REJECT_OUTSIDE_BAR   = env_bool("REJECT_OUTSIDE_BAR", True)

# CSV
CSV_FILE = os.getenv("CSV_FILE", "trades.csv")

# ========================= TERMINAL STYLE =========================
GREEN = "\033[38;5;46m"
RESET = "\033[0m"
BOLD  = "\033[1m"

def now_sp():          return datetime.now(TZ)
def ts_hhmm(dt):       return dt.strftime("%H:%M")
def dt_full(dt):       return dt.strftime("%Y-%m-%d %H:%M:%S")

def tf_to_secs(tf):
    m = {"M1":60, "M5":300, "M15":900}
    if tf not in m:
        raise ValueError(f"TF inválido: {tf} (use M1,M5,M15)")
    return m[tf]

def print_banner():
    head = (GREEN + BOLD +
            "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ INÍCIO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n" +
            "┃ 🔔 SANTO GRIAL — Bot (PAPER)                                          ┃\n" +
            "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛" +
            RESET)
    print(head)
    print(f"▶️  Ativos: {', '.join(ASSETS) if ASSETS else '-'}")
    print(f"⏱  TFs: {', '.join(TIMEFRAMES) if TIMEFRAMES else '-'}")
    print(f"🧮  SG: fast={SG_FAST}, slow={SG_SLOW}, signal={SG_SIGNAL}")
    print(f"⏳  Expiração: {EXPIRATION_CANDLES} candle(s)")
    print(f"📈  Trend filter (EMA{EMA_TREND}): {ONLY_TREND}")
    print(f"📎  BB contexto: {USE_BB_CONTEXT} (SMA={SMA_BB}, σ={BB_STD})")
    print(f"🎲  Martingale (MG1): {USE_MG1}")
    print("— Filtros avançados —")
    print(f"• Força de cruzamento: {USE_MIN_BUFFER_DELTA} (min={MIN_BUFFER_DELTA})")
    print(f"• Corpo confirmado: {USE_BODY_CONFIRM} (min {MIN_BODY_PCT:.0f}% do range)")
    print(f"• Confluência HTF: {USE_HTF_CONFLUENCE} (HTF={HTF_FOR_CONFLUENCE})")
    print(f"• ATR mínimo: {USE_ATR_FILTER} (ATR{ATR_PERIOD} ≥ {MIN_ATR})")
    print(f"• Evitar consolidação: {USE_CONSOLIDATION}")
    print(f"• Volume filter: {USE_VOLUME_CONFIRM}")
    print(f"• MACD confluence: {USE_MACD_CONFLUENCE}")
    print(f"• Delta-ATR ratio: {MIN_BUFFER_DELTA_ATR_RATIO}")
    print(f"• Reject outside-bar: {REJECT_OUTSIDE_BAR}")
    print(f"📝  Log curto por candle: {TICK_LOG_MINIMAL} | Verbose: {VERBOSE}")
    print(f"🌎  TZ: America/Sao_Paulo")
    print(f"⚠️  PAPER — não envia ordens")
    print(f"🔌  IQ Option: {IQ_MODE}")
    print("")

# ========================= TELEGRAM =========================
def tg_send(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHATID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHATID, "text": text}, timeout=10)
    except Exception as e:
        print(f"[Telegram] erro: {e}")

def fmt_signal_msg(tf, asset, when_dt, side, bb_ctx=None):
    lines = [
        "Entrada confirmada",
        f"⏳ {tf}",
        f"📊 {asset}",
        f"🕐 {ts_hhmm(when_dt)}",
        f"🔃 {side}",
    ]
    if bb_ctx:
        lines.append(bb_ctx)
    return "\n".join(lines)

def fmt_result_msg(tf, asset, end_dt, side, wl, mg_tag=""):
    return "\n".join([
        "Resultado",
        "",
        f"⏳ {tf}",
        f"📊 {asset}",
        f"🕐 {ts_hhmm(end_dt)}",
        f"🔃 {side}",
        f"Resultado: {wl} ✅{mg_tag}" if wl=="WIN" else f"Resultado: {wl} ❌{mg_tag}"
    ])

# ========================= INDICADORES =========================
def sma(arr, period):
    if len(arr) < period: return np.full(len(arr), np.nan)
    return pd.Series(arr).rolling(period).mean().to_numpy()

def ema(arr, period):
    if len(arr) < period: return np.full(len(arr), np.nan)
    return pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()

def wma(arr, period):
    if len(arr) < period: return np.full(len(arr), np.nan)
    s = pd.Series(arr)
    w = np.arange(1, period+1, dtype=float)
    return s.rolling(period).apply(lambda x: np.dot(x, w)/w.sum(), raw=True).to_numpy()

def bollinger(close, period=20, stds=3.0):
    s = pd.Series(close)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std(ddof=0)
    upper = mid + stds*std
    lower = mid - stds*std
    return mid.to_numpy(), upper.to_numpy(), lower.to_numpy()

def atr_from_ohlc(high, low, close, period=14):
    h = pd.Series(high); l = pd.Series(low); c = pd.Series(close)
    prev_close = c.shift(1)
    tr = pd.concat([h-l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().to_numpy()

def macd(close, fast=12, slow=26, signal=9):
    s = pd.Series(close)
    ema_f = s.ewm(span=fast).mean()
    ema_s = s.ewm(span=slow).mean()
    macd_l = ema_f - ema_s
    signal_l = macd_l.ewm(span=signal).mean()
    hist = macd_l - signal_l
    return macd_l.to_numpy(), signal_l.to_numpy(), hist.to_numpy()

def santo_grial_signals(close, fast=1, slow=34, signal=5):
    sma_fast = sma(close, fast)
    sma_slow = sma(close, slow)
    buffer1 = sma_fast - sma_slow
    buffer2 = wma(buffer1, signal)
    s1 = pd.Series(buffer1)
    s2 = pd.Series(buffer2)
    buy  = (s1 > s2) & (s1.shift(1) <= s2.shift(1))
    sell = (s1 < s2) & (s1.shift(1) >= s2.shift(1))
    return buy.to_numpy(), sell.to_numpy(), buffer1, buffer2

# ========================= ESTADO & TRAVA GLOBAL =========================
history = []
wins = 0; losses = 0
last_bar_time = {}
cooldown_flag = {}
global_trade_lock = threading.Lock()
active_signal = None

# ========================= IQ OPTION =========================
def iq_connect():
    Iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
    Iq.connect()
    Iq.change_balance(IQ_MODE)
    for _ in range(30):
        if Iq.check_connect():
            break
        time.sleep(1)
    if not Iq.check_connect():
        raise RuntimeError("Falha ao conectar na IQ Option.")
    return Iq

def get_candles(Iq, asset, seconds, count=500, endtime=None):
    if endtime is None:
        endtime = int(time.time())
    return Iq.get_candles(asset, seconds, count, endtime)

def get_payout(Iq, asset):
    try:
        return None
    except Exception:
        return None

# ========================= TELEGRAM POLLER =========================
def make_telegram_poller(Iq):
    _last_update_id = {"v": None}
    def poll():
        nonlocal Iq
        global wins, losses, active_signal
        if not ENABLE_TELEGRAM_COMMANDS or not TELEGRAM_TOKEN or not TELEGRAM_CHATID:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
            params = {}
            if _last_update_id["v"] is not None:
                params["offset"] = _last_update_id["v"] + 1
            r = requests.get(url, params=params, timeout=10)
            data = r.json()
            if not data.get("ok"):
                return
            for up in data.get("result", []):
                _last_update_id["v"] = up["update_id"]
                msg = up.get("message") or up.get("edited_message")
                if not msg:
                    continue
                chat_id = msg["chat"]["id"]
                if str(chat_id) != str(TELEGRAM_CHATID):
                    continue
                text = (msg.get("text") or "").strip().lower()
                if text == "/ativos":
                    tg_send("Ativos monitorados:\n" + "\n".join([f"- {a}" for a in ASSETS]) + "\n\nTFs: " + ", ".join(TIMEFRAMES))
                elif text == "/resumo":
                    total = wins + losses
                    taxa = (wins/total*100) if total>0 else 0.0
                    status = "ocupado" if active_signal else "livre"
                    tg_send(f"Resumo\n\nWins: {wins}\nLosses: {losses}\nTaxa: {taxa:.2f}%\nEntradas: {total}\nFila global: {status}")
                elif text == "/historico":
                    ult = history[-10:]
                    if not ult:
                        tg_send("Histórico vazio.")
                    else:
                        lines=[]
                        for h in ult:
                            lines.append(f"{h['time']} | {h['tf']} | {h['asset']} | {h['side']} | {h['result']}")
                        tg_send("Últimos sinais:\n" + "\n".join(lines))
                elif text == "/status":
                    snaps = []
                    for asset in ASSETS:
                        for tf in TIMEFRAMES:
                            try:
                                sec = tf_to_secs(tf)
                                c = get_candles(Iq, asset, sec, 3)
                                if not c:
                                    snaps.append(f"{asset} {tf}: sem dados")
                                    continue
                                df = pd.DataFrame(c).sort_values("from")
                                last = df.iloc[-1]
                                close = float(last["close"])
                                tbar = datetime.fromtimestamp(int(last["to"]), TZ).strftime("%H:%M:%S")
                                snaps.append(f"{asset} {tf} | {tbar} | close={close}")
                            except Exception:
                                snaps.append(f"{asset} {tf}: erro ao ler")
                    if active_signal:
                        snaps.append(f"\n🧵 Em curso: {active_signal['asset']} {active_signal['tf']} {active_signal['side']}")
                    tg_send("Status\n" + "\n".join(snaps))
                elif text == "/resumo+":
                    tg_send(build_summary_plus())
        except Exception as e:
            print(f"[Telegram polling] erro: {e}")
    return poll

# ========================= AUXILIARES DE FEATURE/FILTROS =========================
def body_pct_dir(open_, close_, high_, low_):
    rng = max(high_ - low_, 1e-12)
    body = abs(close_ - open_)
    pct  = (body / rng) * 100.0
    direction = "BULL" if close_ > open_ else ("BEAR" if close_ < open_ else "DOJI")
    return pct, direction

def is_consolidating(opens, closes, highs, lows, lookback, body_pct_max, require_alternation=True):
    if len(closes) < lookback + 1:
        return False
    o = pd.Series(opens[-lookback:]); c = pd.Series(closes[-lookback:])
    h = pd.Series(highs[-lookback:]); l = pd.Series(lows[-lookback:])
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    pct = (body / rng) * 100.0
    small_bodies = pct.fillna(0) <= body_pct_max
    if require_alternation:
        dirs = np.sign(c - o).fillna(0).to_numpy()
        changes = np.sum(np.diff(dirs) != 0)
        return small_bodies.all() and changes >= int(lookback * 0.5)
    return small_bodies.all()

def dist_to_sr(price, highs, lows, lookback):
    if len(highs) < lookback + 1 or len(lows) < lookback + 1:
        return np.nan, np.nan, np.nan
    hh = np.max(highs[-lookback:]); ll = np.min(lows[-lookback:])
    d_hh = abs(price - hh); d_ll = abs(price - ll); d_sr = min(d_hh, d_ll)
    return d_sr, d_hh, d_ll

def htf_aligned(Iq, asset, htf_str, side):
    try:
        sec = tf_to_secs(htf_str)
    except Exception:
        return True
    c = get_candles(Iq, asset, sec, 300)
    if not c or len(c) < EMA_TREND + 5:
        return True
    df = pd.DataFrame(c).sort_values("from")
    close = df["close"].to_numpy(dtype=float)
    ema100 = ema(close, EMA_TREND)
    i = len(df) - 1
    if np.isnan(ema100[i-1]):
        return True
    if side == "CALL":
        return close[i-1] > ema100[i-1]
    else:
        return close[i-1] < ema100[i-1]

def collect_features_for_log(asset, tf, df, i, close, open_, high, low, buffer1, buffer2, ema100, atr_arr, payout):
    bdelta = float(abs(buffer1[i] - buffer2[i]))
    body_pct, dir_candle = body_pct_dir(open_[i], close[i], high[i], low[i])
    dist_ema = float(abs(close[i] - ema100[i])) if not np.isnan(ema100[i]) else np.nan
    d_sr, d_hh, d_ll = dist_to_sr(close[i], high, low, SR_LOOKBACK)
    atrv = float(atr_arr[i]) if atr_arr is not None and not np.isnan(atr_arr[i]) else np.nan
    return {
        "asset": asset,
        "tf": tf,
        "timestamp": int(df.iloc[i]["to"]),
        "hour": int(datetime.fromtimestamp(int(df.iloc[i]["to"]), TZ).strftime("%H")),
        "weekday": int(datetime.fromtimestamp(int(df.iloc[i]["to"]), TZ).strftime("%w")),
        "buffer_delta": bdelta,
        "body_pct": float(body_pct),
        "candle_dir": dir_candle,
        "dist_to_ema": dist_ema,
        "dist_to_sr": float(d_sr) if d_sr==d_sr else np.nan,
        "dist_to_hh": float(d_hh) if d_hh==d_hh else np.nan,
        "dist_to_ll": float(d_ll) if d_ll==d_ll else np.nan,
        "atr": atrv,
        "payout": payout if payout is None else float(payout)
    }

# ========================= CSV =========================
def append_csv(trade_dict):
    """Escreve linha no CSV assim que o trade termina."""
    fieldnames = ["date","asset","tf","side","entry","exit","result","buffer_delta","body_pct","atr","hour","weekday"]
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "date": trade_dict["time"],
            "asset": trade_dict["asset"],
            "tf": trade_dict["tf"],
            "side": trade_dict["side"],
            "entry": f"{trade_dict['entry_price']:.6f}",
            "exit": f"{trade_dict['exit_price']:.6f}",
            "result": trade_dict["result"],
            "buffer_delta": f"{trade_dict['features']['buffer_delta']:.7f}",
            "body_pct": f"{trade_dict['features']['body_pct']:.1f}",
            "atr": f"{trade_dict['features']['atr']:.6f}",
            "hour": trade_dict["features"]["hour"],
            "weekday": trade_dict["features"]["weekday"]
        })

# ========================= RESUMO+ =========================
def build_summary_plus():
    if not history:
        return "Resumo+\n\nSem operações ainda."
    df = pd.DataFrame(history)
    df["is_win"] = (df["result"] == "WIN").astype(int)
    by_asset = df.groupby("asset")["is_win"].mean().sort_values(ascending=False) * 100
    hours = [h["features"]["hour"] if h.get("features") else None for h in history]
    weekdays = [h["features"]["weekday"] if h.get("features") else None for h in history]
    df["hour"] = hours; df["weekday"] = weekdays
    by_hour = df.dropna(subset=["hour"]).groupby("hour")["is_win"].mean().sort_values(ascending=False) * 100
    by_wday = df.dropna(subset=["weekday"]).groupby("weekday")["is_win"].mean().sort_values(ascending=False) * 100
    def topn(s, n=7):
        lines=[]
        for k,v in s.head(n).items():
            lines.append(f"{k}: {v:.1f}%")
        return "\n".join(lines) if lines else "(sem dados)"
    txt = ["Resumo+"]
    wr = df["is_win"].mean()*100
    txt.append(f"\nWinrate geral: {wr:.1f}% (n={len(df)})")
    txt.append("\nPor ativo:"); txt.append(topn(by_asset, n=len(by_asset)))
    txt.append("\nPor hora (0-23):"); txt.append(topn(by_hour, n=len(by_hour)))
    label_day = {0:"Dom",1:"Seg",2:"Ter",3:"Qua",4:"Qui",5:"Sex",6:"Sáb"}
    if not by_wday.empty:
        by_wday.index = by_wday.index.map(lambda x: label_day.get(int(x), str(int(x))))
    txt.append("\nPor dia da semana:"); txt.append(topn(by_wday, n=len(by_wday)))
    return "\n".join(txt)

# ========================= CORE =========================
def analyze_and_signal(Iq, asset, tf):
    global wins, losses, active_signal
    sec = tf_to_secs(tf)
    key = (asset, tf)
    cooldown_flag.setdefault(key, None)
    last_bar_time.setdefault(key, 0)

    c = get_candles(Iq, asset, sec, 600)
    if not c or len(c) < max(EMA_TREND, SMA_BB, ATR_PERIOD, SR_LOOKBACK, SG_SLOW + SG_SIGNAL + 5):
        return
    df = pd.DataFrame(c).sort_values("from")
    close = df["close"].to_numpy(dtype=float)
    openp = df["open"].to_numpy(dtype=float)
    high  = df["max"].to_numpy(dtype=float)
    low   = df["min"].to_numpy(dtype=float)

    buy_sig, sell_sig, buffer1, buffer2 = santo_grial_signals(close, SG_FAST, SG_SLOW, SG_SIGNAL)
    ema100 = ema(close, EMA_TREND)
    atrArr = atr_from_ohlc(high, low, close, ATR_PERIOD) if USE_ATR_FILTER else None
    if USE_BB_CONTEXT:
        bbm, bbu, bbl = bollinger(close, SMA_BB, BB_STD)
    else:
        bbm = bbu = bbl = None

    i = len(df) - 1
    bar_close_epoch = int(df.iloc[i]["to"])

    if (bar_close_epoch != last_bar_time[key]) and TICK_LOG_MINIMAL and not VERBOSE:
        bar_dt = datetime.fromtimestamp(bar_close_epoch, TZ).strftime("%H:%M:%S")
        print(f"{bar_dt} {asset} {tf} | close={close[i-1]:.5f}")

    if COOLDOWN_SAME_BAR and cooldown_flag[key] == bar_close_epoch:
        return
    if bar_close_epoch == last_bar_time[key]:
        return

    side = "CALL" if buy_sig[i] else ("PUT" if sell_sig[i] else None)

    # 0) trend
    if side is not None and ONLY_TREND:
        if np.isnan(ema100[i-1]):
            side = None
        else:
            if side == "CALL" and not (close[i-1] > ema100[i-1]): side = None
            if side == "PUT"  and not (close[i-1] < ema100[i-1]): side = None

    # 1) volume
    if side is not None and USE_VOLUME_CONFIRM:
        vol = df["volume"].astype(float)
        vol_sma = sma(vol, VOLUME_PERIOD)
        vol_std = pd.Series(vol).rolling(VOLUME_PERIOD).std()
        if vol[i] < (vol_sma[i] + VOLUME_STDDEV * vol_std[i]):
            side = None

    # 3) macd
    if side is not None and USE_MACD_CONFLUENCE:
        try:
            sec_macd = tf_to_secs(MACD_TF)
        except Exception:
            sec_macd = None
        if sec_macd:
            c_macd = get_candles(Iq, asset, sec_macd, 300)
            if c_macd and len(c_macd) > max(MACD_FAST, MACD_SLOW) + 5:
                df_macd = pd.DataFrame(c_macd).sort_values("from")
                close_macd = df_macd["close"].to_numpy(dtype=float)
                _, _, hist = macd(close_macd, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
                if not np.isnan(hist[-2]):
                    if side == "CALL" and hist[-2] < 0:
                        side = None
                    if side == "PUT" and hist[-2] > 0:
                        side = None

    # 4) delta atr
    if side is not None and USE_MIN_BUFFER_DELTA:
        delta_abs = abs(buffer1[i] - buffer2[i])
        if atrArr is not None and not np.isnan(atrArr[i-1]):
            delta_min = MIN_BUFFER_DELTA_ATR_RATIO * atrArr[i-1]
            if delta_abs < delta_min:
                side = None

    # 5) outside bar
    if side is not None and REJECT_OUTSIDE_BAR:
        prev_dir = 1 if close[i-1] > openp[i-1] else -1
        curr_dir = 1 if close[i]   > openp[i]   else -1
        outside  = (high[i] > high[i-1]) and (low[i] < low[i-1])
        if outside and (curr_dir != prev_dir):
            side = None

    # body
    if side is not None and USE_BODY_CONFIRM:
        pct, dir_c = body_pct_dir(openp[i], close[i], high[i], low[i])
        if pct < MIN_BODY_PCT:
            side = None
        else:
            if side == "CALL" and dir_c != "BULL": side = None
            if side == "PUT"  and dir_c != "BEAR": side = None

    # atr
    if side is not None and USE_ATR_FILTER:
        if atrArr is None or np.isnan(atrArr[i-1]) or atrArr[i-1] < MIN_ATR:
            side = None

    # consolidação
    if side is not None and USE_CONSOLIDATION:
        if is_consolidating(openp, close, high, low, CONS_LOOKBACK, CONS_BODY_PCT_MAX, CONS_REQUIRE_ALTERNATE):
            side = None

    # htf
    if side is not None and USE_HTF_CONFLUENCE:
        if not htf_aligned(Iq, asset, HTF_FOR_CONFLUENCE, side):
            side = None

    if side is None:
        last_bar_time[key] = bar_close_epoch
        cooldown_flag[key] = bar_close_epoch
        return

    if active_signal is not None:
        last_bar_time[key] = bar_close_epoch
        cooldown_flag[key] = bar_close_epoch
        return

    acquired = global_trade_lock.acquire(blocking=False)
    if not acquired:
        last_bar_time[key] = bar_close_epoch
        cooldown_flag[key] = bar_close_epoch
        return

    try:
        if active_signal is not None:
            global_trade_lock.release()
            last_bar_time[key] = bar_close_epoch
            cooldown_flag[key] = bar_close_epoch
            return

        payout = get_payout(Iq, asset)
        features = collect_features_for_log(
            asset, tf, df, i, close, openp, high, low, buffer1, buffer2, ema100,
            atrArr if USE_ATR_FILTER else None, payout
        )

        active_signal = {"asset": asset, "tf": tf, "side": side, "sec": sec, "start_to": bar_close_epoch}
        entry_from = bar_close_epoch
        entry_dt   = datetime.fromtimestamp(entry_from, TZ)

        sleep_to_entry = max(0, entry_from - int(time.time()))
        if sleep_to_entry > 0:
            time.sleep(sleep_to_entry)

        msg = fmt_signal_msg(tf, asset, entry_dt, side, None)
        print(f"[{dt_full(now_sp())}] ENTRADA {asset} {tf} {side} (abertura)")
        tg_send(msg)

        def wait_and_fetch_target_candle(Iq, asset, sec, target_to):
            deadline = time.time() + sec*6
            while time.time() < deadline:
                c2 = get_candles(Iq, asset, sec, 20)
                if c2:
                    d2 = pd.DataFrame(c2).sort_values("from")
                    out = d2[d2["to"] == target_to]
                    if not out.empty:
                        out_row = out.iloc[-1]
                        entry = d2[d2["from"] == entry_from]
                        if not entry.empty:
                            entry_row = entry.iloc[0]
                        else:
                            idx = d2.index.get_loc(out.index[-1])
                            back = EXPIRATION_CANDLES
                            entry_row = d2.iloc[max(0, idx-back+1)]
                        return entry_row, out_row
                time.sleep(0.4)
            return None, None

        def resolve_result():
            global wins, losses, active_signal
            nonlocal side, asset, tf, sec, entry_from, features

            try:
                target_to = entry_from + sec*EXPIRATION_CANDLES
                sleep_seconds = max(0, target_to - int(time.time()) + 1)
                time.sleep(sleep_seconds)

                entry_row, out_row = wait_and_fetch_target_candle(Iq, asset, sec, target_to)
                if entry_row is None or out_row is None:
                    res = "LOSS"
                    tg_send(fmt_result_msg(tf, asset, now_sp(), side, res, ""))
                    return

                entry_price = float(entry_row["open"])
                exit_price  = float(out_row["close"])
                end_dt      = datetime.fromtimestamp(int(out_row["to"]), TZ)
                res = "WIN" if ((side=="CALL" and exit_price>entry_price) or (side=="PUT" and exit_price<entry_price)) else "LOSS"
                mg_tag = ""

                if res == "LOSS" and USE_MG1:
                    target_to2 = target_to + sec
                    sleep2 = max(0, target_to2 - int(time.time()) + 1)
                    time.sleep(sleep2)
                    entry_row2, out_row2 = wait_and_fetch_target_candle(Iq, asset, sec, target_to2)
                    if entry_row2 is not None and out_row2 is not None:
                        entry_price2 = float(entry_row2["open"])
                        exit_price2  = float(out_row2["close"])
                        exit_price   = exit_price2
                        end_dt       = datetime.fromtimestamp(int(out_row2["to"]), TZ)
                        res          = "WIN" if ((side=="CALL" and exit_price2>entry_price2) or (side=="PUT" and exit_price2<entry_price2)) else "LOSS"
                        mg_tag = " (1)"
                    else:
                        mg_tag = " (1)"

                history.append({
                    "time": ts_hhmm(end_dt),
                    "asset": asset,
                    "tf": tf,
                    "side": side,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "result": res,
                    "features": features
                })
                if res == "WIN":
                    wins += 1
                else:
                    losses += 1

                # ---------- salva CSV ----------
                append_csv(history[-1])

                tg_send(fmt_result_msg(tf, asset, end_dt, side, res, mg_tag))
            finally:
                active_signal = None
                try:
                    global_trade_lock.release()
                except RuntimeError:
                    pass

        threading.Thread(target=resolve_result, daemon=True).start()

    except Exception:
        active_signal = None
        try:
            global_trade_lock.release()
        except RuntimeError:
            pass
    finally:
        last_bar_time[key] = bar_close_epoch
        cooldown_flag[key] = bar_close_epoch

# ========================= MAIN =========================
def main():
    if not IQ_EMAIL or not IQ_PASSWORD:
        print("Defina IQ_EMAIL e IQ_PASSWORD no .env")
        return
    if not ASSETS or not TIMEFRAMES:
        print("Defina ASSETS e TIMEFRAMES no .env")
        return

    Iq = iq_connect()
    print(f"[{dt_full(now_sp())}] Conectado à IQ Option ({IQ_MODE})")
    print_banner()

    tg_send(
        "🔔 SANTO GRIAL — Bot (PAPER)\n"
        f"▶️ Ativos: {', '.join(ASSETS)}\n"
        f"⏱ TFs: {', '.join(TIMEFRAMES)}\n"
        f"🧮 SG: fast={SG_FAST}, slow={SG_SLOW}, signal={SG_SIGNAL}\n"
        f"⏳ Expiração: {EXPIRATION_CANDLES} candle(s)\n"
        f"📈 Trend filter (EMA{EMA_TREND}): {ONLY_TREND}\n"
        f"📎 BB contexto: {USE_BB_CONTEXT}\n"
        f"🎲 MG1: {USE_MG1}\n"
        f"🌎 TZ: America/Sao_Paulo\n"
        "⚠️ PAPER — nenhuma ordem na corretora."
    )

    poller = make_telegram_poller(Iq)

    try:
        last_poll = 0
        while True:
            now_epoch = time.time()

            for asset in ASSETS:
                for tf in TIMEFRAMES:
                    try:
                        analyze_and_signal(Iq, asset, tf)
                    except Exception as e:
                        print(f"[{asset} {tf}] erro: {e}")

            if ENABLE_TELEGRAM_COMMANDS and (now_epoch - last_poll > POLL_INTERVAL_SEC):
                poller()
                last_poll = now_epoch

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nEncerrado.")

if __name__ == "__main__":
    main()