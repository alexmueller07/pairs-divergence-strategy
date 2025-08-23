"""
Live pairs trading main loop with:
 - correlation + optional cointegration filtering
 - adaptive thresholds (optimized entry/exit) with overshoot & gap guards
 - abnormal-opposite-move gate with persistence
 - volatility-adjusted sizing
 - progressive scale-in / scale-out
 - stop-loss / give-up (no improvement) / max hold-time
 - periodic pair re-discovery intra-day
 - detailed structured entry/exit printouts
 - session hygiene (avoid first/last minutes)
 - overnight EOD flattening

Relies on config.py, get_related.py and strategy.py 
"""

import time
import math
import traceback
from datetime import datetime, time as dt_time, timedelta
from collections import defaultdict, deque
from pytz import timezone

import numpy as np
import pandas as pd
import pytz
import alpaca_trade_api as tradeapi

# =========================
# Tunables / Diagnostics
# =========================
DEBUG = True
MIN_ENTRY_Z_ABS = 1.00     
EXIT_Z_ABS_FLOOR = 0.30
ENTRY_OVERSHOOT = 1.00 
MIN_GAP_MULTIPLIER = 1.20 
MIN_GAP_DOLLARS_FLOOR = 0.05
SLIPPAGE_BUFFER = 0.01

MIN_SIGMA_MOVE = 0.0
ABNORMAL_PERSIST_BARS = 0

ENTRY_LEVEL_MULTIPLIERS = [1.00, 1.50, 2.00]
ENTRY_FRACTIONS        = [0.50, 0.30, 0.20]

EXIT_PARTIAL_MULTIPLIERS = [0.70, 0.50]  
EXIT_PARTIAL_FRACTIONS   = [0.50, 0.50]

STOP_LOSS_Z = 4.0 
MAX_HOLD_BARS = 4 * 60 
NO_IMPROVEMENT_BARS = 10
IMPROVE_EPS = 0.25

REDISCOVER_EVERY_MIN = 60

AVOID_AFTER_OPEN_MIN = 1
AVOID_BEFORE_CLOSE_MIN = 10

try:
    from statsmodels.tsa.stattools import coint
    HAVE_COINTEGRATION = True
except Exception:
    HAVE_COINTEGRATION = False

from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
    PERIOD, INTERVAL, CORRELATION_THRESHOLD, MIN_VALID_RATIO, DOWNLOAD_THREADING,
    LIVE_PERIOD_MIN, LIVE_INTERVAL,
    ROLLING_WINDOW, ENTRY_Z, EXIT_Z, COOLDOWN_BARS,
    MAX_OPEN_PAIRS, PER_PAIR_MAX_DOLLAR, ACCOUNT_RISK_CAP_PCT,
    MIN_RISK_PCT, MAX_RISK_PCT, SIGNAL_STRENGTH_CAP,
    EOD_FLATTEN_HHMM_EST, LOG_EVERY_X_SECS
)
from get_related import get_related_pairs
from strategy import download_pair_prices, rolling_beta, zscore

# Alpaca client
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

EST = pytz.timezone('US/Eastern')

# runtime state
pair_state = {}  
last_trade_bar_idx = defaultdict(lambda: -10_000)

# order throttle
ORDER_THROTTLE_SECONDS = 0.25

def safe_request(func, *args, retries=5, base_sleep=1.0, **kwargs):
    attempt = 0
    while attempt <= retries:
        try:
            return True, func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            wait = min(base_sleep * (2 ** (attempt - 1)), 10.0)
            print(f"safe_request attempt {attempt}/{retries} failed for {getattr(func, '__name__', func)} -> {e}")
            if attempt > retries:
                return False, e
            time.sleep(wait)
    return False, RuntimeError("safe_request unexpected")

def get_account_safe():
    ok, res = safe_request(api.get_account, retries=4, base_sleep=1.0)
    if ok: return res
    raise res

def list_positions_safe():
    ok, res = safe_request(api.list_positions, retries=4, base_sleep=1.0)
    if ok: return res
    raise res

def submit_order_safe(**kwargs):
    ok, res = safe_request(api.submit_order, retries=2, base_sleep=0.5, **kwargs)
    if ok: return res
    raise res

def now_est():
    return datetime.now(EST)

def compute_used_notional_from_positions(positions):
    total = 0.0
    for p in positions:
        try:
            total += abs(float(p.market_value))
        except Exception:
            pass
    return total

def dynamic_thresholds_from_corr_and_history(corr, z_hist_series):
    """
    Combine base scaling by correlation with an adaptive percentile of historical |z|
    to choose entry/exit more robustly.
    Rules:
      - base_entry = ENTRY_Z / |corr|
      - hist_entry = 90th percentile of |z_history|
      - chosen_entry = max(base_entry, hist_entry * 0.9)
      - exit: base_exit = EXIT_Z / |corr|, hist_exit = 50th percentile of |z_history|
      - chosen_exit = min(base_exit, max(base_exit * 0.5, hist_exit * 0.6))
    """
    c = max(abs(corr), 1e-6)
    base_entry = ENTRY_Z / c
    base_exit = EXIT_Z / c

    if z_hist_series is None or z_hist_series.dropna().shape[0] < max(10, ROLLING_WINDOW//2):
        return max(base_entry, MIN_ENTRY_Z_ABS), max(base_exit, EXIT_Z_ABS_FLOOR)

    abs_z = z_hist_series.dropna().abs()
    try:
        hist_entry = float(np.nanpercentile(abs_z, 90))
        hist_exit  = float(np.nanpercentile(abs_z, 50))
    except Exception:
        return max(base_entry, MIN_ENTRY_Z_ABS), max(base_exit, EXIT_Z_ABS_FLOOR)

    chosen_entry = max(base_entry, hist_entry * 0.9)
    chosen_exit  = min(base_exit, max(base_exit * 0.5, hist_exit * 0.6))
    chosen_entry = float(max(chosen_entry, MIN_ENTRY_Z_ABS))
    chosen_exit  = float(max(chosen_exit, EXIT_Z_ABS_FLOOR))
    return chosen_entry, chosen_exit

def confidence_weight(corr: float) -> float:
    lo, hi = CORRELATION_THRESHOLD, 1.0
    x = (abs(corr) - lo) / max(hi - lo, 1e-6)
    return float(np.clip(x, 0.0, 1.0))

def pair_position_count():
    return sum(1 for s in pair_state.values() if s.get("pos", 0) != 0)

def flatten_all(positions_cache=None):
    try:
        if positions_cache is None:
            positions = list_positions_safe()
        else:
            positions = positions_cache
        for p in positions:
            try:
                qty = int(abs(float(p.qty)))
                if qty <= 0:
                    continue
                side = "sell" if float(p.qty) > 0 else "buy"
                print(f"[FLATTEN] {p.symbol} {qty} shares ({side})")
                submit_order_safe(symbol=p.symbol, qty=qty, side=side, type="market", time_in_force="day")
                time.sleep(ORDER_THROTTLE_SECONDS)
            except Exception as ex:
                print(f"Flatten order error for {p.symbol}: {ex}")
    except Exception as e:
        print("Error flattening positions:", e)

def is_cointegrated(series_a: pd.Series, series_b: pd.Series, significance=0.05):
    if not HAVE_COINTEGRATION:
        return True 
    try:
        t_stat, pvalue, _ = coint(series_a.dropna(), series_b.dropna(), maxlag=1)
        return (pvalue <= significance)
    except Exception:
        return False

# Volatility adjusted notional calculation
def compute_target_notional(equity, base_risk_pct, strength, spread_std, vol1=None, vol2=None):
    """
    Produce dollar notional per leg:
      - Base: (equity * base_risk_pct * strength) / spread_std
      - Adjust per-leg sizing inversely with each ticker's volatility
    """
    if spread_std <= 0 or not np.isfinite(spread_std):
        return 0.0
    raw = (equity * base_risk_pct * strength) / spread_std
    scaled = raw * 0.6 

    if vol1 is not None and vol2 is not None and vol1 > 0 and vol2 > 0:
        avg_vol = (vol1 + vol2) / 2
        adj_factor = max(min(avg_vol / max(vol1, vol2, 1e-6), 2.0), 0.5)
        scaled *= adj_factor

    notional = min(scaled, PER_PAIR_MAX_DOLLAR)
    return float(notional if notional >= 50 else 0.0)

def within_session_window():
    """Skip first few minutes after open and last few before EOD flatten time."""
    hh, mm = EOD_FLATTEN_HHMM_EST
    now_t = now_est().time()
    open_t = dt_time(9, 30)
    avoid_start = (datetime.combine(datetime.today(), open_t) + timedelta(minutes=AVOID_AFTER_OPEN_MIN)).time()
    avoid_end   = (datetime.combine(datetime.today(), dt_time(hh, mm)) - timedelta(minutes=AVOID_BEFORE_CLOSE_MIN)).time()
    return (now_t >= avoid_start) and (now_t <= avoid_end)

# Main trade loop
def trade_loop():
    def discover_pairs():
        try:
            base_pairs = get_related_pairs(period=PERIOD, interval=INTERVAL,
                                           correlation_threshold=CORRELATION_THRESHOLD,
                                           min_valid_ratio=MIN_VALID_RATIO)
        except Exception as e:
            print("Failed discovering pairs:", e)
            return []

        if not base_pairs:
            print("No correlated pairs found.")
            return []

        print(f"Found {len(base_pairs)} correlated pairs. Running cointegration filter (recommended).")
        coint_pairs = []
        for (t1, t2, corr) in base_pairs:
            try:
                df_hist = download_pair_prices(t1, t2, period=PERIOD, interval=INTERVAL)
                la, lb = np.log(df_hist[t1]), np.log(df_hist[t2])
                if is_cointegrated(la, lb):
                    coint_pairs.append((t1, t2, corr))
            except Exception:
                pass

        if coint_pairs:
            pairs_ = coint_pairs
            print(f"{len(pairs_)} pairs remain after cointegration filter.")
        else:
            pairs_ = base_pairs
            print("Cointegration filter removed all pairs or statsmodels unavailable; proceeding with correlation-only pairs (not ideal).")

        pairs_.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs_

    pairs = discover_pairs()
    if not pairs:
        return

    print(f"Live trading {len(pairs)} pairs with correlation >= {CORRELATION_THRESHOLD}")

    last_discovery = now_est()

    # Main loop
    while True:
        try:
            # Periodically re-discover pairs intra-day
            if (now_est() - last_discovery).total_seconds() >= REDISCOVER_EVERY_MIN * 60:
                print("\n[INFO] Re-discovering pairs intra-day...")
                new_pairs = discover_pairs()
                if new_pairs:
                    pairs = new_pairs
                    print(f"[INFO] Using {len(pairs)} refreshed pairs.")
                last_discovery = now_est()

            hh, mm = EOD_FLATTEN_HHMM_EST
            if now_est().time() >= dt_time(hh, mm):
                print("EOD flatten time reached; closing all positions and stopping.")
                try:
                    positions = list_positions_safe()
                except Exception:
                    positions = None
                flatten_all(positions_cache=positions)
                break

            if not within_session_window():
                if DEBUG:
                    print("[DEBUG] Outside preferred session window; skipping iteration.")
                time.sleep(LOG_EVERY_X_SECS)
                continue

            try:
                account = get_account_safe()
                equity = float(account.equity)
            except Exception as e:
                print("Could not fetch account equity; skipping this loop iteration. Error:", e)
                time.sleep(LOG_EVERY_X_SECS)
                continue

            try:
                positions = list_positions_safe()
            except Exception as e:
                print("Could not fetch positions; continuing with empty positions cache:", e)
                positions = []

            used_notional = compute_used_notional_from_positions(positions)

            # Iterate pairs
            for (t1, t2, corr) in pairs:
                key = (t1, t2)
                try:
                    df = download_pair_prices(t1, t2, period=LIVE_PERIOD_MIN, interval=LIVE_INTERVAL)
                    if df is None or df.empty or t1 not in df.columns or t2 not in df.columns:
                        continue
                    if df.shape[0] < max(ROLLING_WINDOW + 5, 30):
                        continue

                    la, lb = np.log(df[t1]), np.log(df[t2])
                    beta_series = rolling_beta(la, lb, ROLLING_WINDOW).ffill()
                    if beta_series.dropna().empty:
                        continue
                    beta = float(beta_series.dropna().iloc[-1])

                    spread = la - beta * lb
                    z_series = zscore(spread, ROLLING_WINDOW)
                    if z_series.dropna().empty:
                        continue
                    z_last = float(z_series.dropna().iloc[-1])
                    spread_val_log = float(spread.iloc[-1])

                    p1 = float(df[t1].iloc[-1])
                    p2 = float(df[t2].iloc[-1])

                    ret1 = (df[t1].iloc[-1] / df[t1].iloc[-2] - 1.0) if df[t1].shape[0] > 1 else 0.0
                    ret2 = (df[t2].iloc[-1] / df[t2].iloc[-2] - 1.0) if df[t2].shape[0] > 1 else 0.0
                    vol1 = float(df[t1].pct_change().rolling(ROLLING_WINDOW).std().dropna().iloc[-1]) if df[t1].pct_change().dropna().shape[0] > 0 else 0.0
                    vol2 = float(df[t2].pct_change().rolling(ROLLING_WINDOW).std().dropna().iloc[-1]) if df[t2].pct_change().dropna().shape[0] > 0 else 0.0

                    st = pair_state.get(key, {
                        "pos": 0, "qty1": 0, "qty2": 0,
                        "entry_time": None, "entry_z": None, "entry_spread": None,
                        "corr": corr, "beta": beta, "last_exit_bar": None,
                        "scale_level": -1,
                        "worst_abs_z": 0.0,
                        "bars_since_entry": 0
                    })
                    st["beta"] = beta
                    st["corr"] = corr
                    pair_state[key] = st

                    gap_dollars = float(p1 - beta * p2)
                    hist_z = z_series.dropna()
                    entry_z_dynamic, exit_z_dynamic = dynamic_thresholds_from_corr_and_history(corr, hist_z)

                    spread_std_log_series = spread.rolling(ROLLING_WINDOW).std()
                    spread_std_log_last = float(spread_std_log_series.dropna().iloc[-1]) if not spread_std_log_series.dropna().empty else 0.0
                    spread_std = spread_std_log_last

                    bar_index = len(df) - 1

                    if st.get("last_exit_bar") is not None:
                        if bar_index - last_trade_bar_idx[key] < COOLDOWN_BARS:
                            continue

                    pos = st.get("pos", 0)

                    conf = confidence_weight(corr)
                    base_risk_pct = MIN_RISK_PCT + conf * (MAX_RISK_PCT - MIN_RISK_PCT)
                    strength = float(np.clip(abs(z_last) / max(entry_z_dynamic, 1e-6), 0.0, SIGNAL_STRENGTH_CAP))

                    target_notional = compute_target_notional(
                        equity, base_risk_pct, strength, spread_std, vol1, vol2
                    )
                    if target_notional <= 0:
                        if DEBUG:
                            print(f"[DEBUG skip] {t1}/{t2}: target_notional<=0 (z_last={z_last:.3f}, "
                                  f"entry_z_dyn={entry_z_dynamic:.3f}, spread_std={spread_std:.6g})")
                        continue

                    # === ENTRY / SCALE-IN conditions ===
                    enter_short = (z_last >= entry_z_dynamic)
                    enter_long  = (z_last <= -entry_z_dynamic)
                    want_entry  = (enter_short or enter_long)


                    if want_entry and (not np.isfinite(spread_std_log_last) or spread_std_log_last <= 0):
                        if DEBUG:
                            print(f"[DEBUG skip] {t1}/{t2}: spread_std_log_last <= 0")
                        continue

                    min_gap_log = entry_z_dynamic * spread_std_log_last
                    ref_price = max(p1, p2)
                    min_gap_dollars_approx = abs(min_gap_log) * ref_price

                    # gap must exceed threshold by multiplier + slippage buffer
                    min_gap_required = max(min_gap_dollars_approx * MIN_GAP_MULTIPLIER,
                                           MIN_GAP_DOLLARS_FLOOR) + SLIPPAGE_BUFFER

                    if want_entry and abs(gap_dollars) < min_gap_required:
                        if DEBUG:
                            print(f"[DEBUG skip] {t1}/{t2}: gap too small ({abs(gap_dollars):.4f} < {min_gap_required:.4f})")
                        want_entry = False

                    # enforce account-level cap before creating/adding a position
                    if want_entry:
                        add_notional = target_notional   # we will scale per level below
                        if (used_notional + 2 * add_notional) > (ACCOUNT_RISK_CAP_PCT * equity):
                            if DEBUG:
                                print(f"[DEBUG skip] {t1}/{t2}: account cap (used={used_notional:.0f}, "
                                      f"add={2*add_notional:.0f}, cap={ACCOUNT_RISK_CAP_PCT*equity:.0f})")
                            want_entry = False

                    next_level = st["scale_level"] + 1
                    eligible_level = -1
                    if want_entry and next_level < len(ENTRY_LEVEL_MULTIPLIERS):
                        need = entry_z_dynamic * ENTRY_LEVEL_MULTIPLIERS[next_level]
                        if abs(z_last) >= need:
                            eligible_level = next_level

                    if pos == 0 and eligible_level >= 0 and pair_position_count() < MAX_OPEN_PAIRS:
                        frac = ENTRY_FRACTIONS[eligible_level]
                        leg_notional = max(target_notional * frac, 0.0)

                        abs_beta = max(abs(beta), 1e-6)

                        qty1 = int(max(1, math.floor(leg_notional / max(p1, 1e-6))))
                        qty2 = int(max(1, math.floor((leg_notional * abs_beta) / max(p2, 1e-6))))

                        max_qty1 = max(1, int(math.floor(PER_PAIR_MAX_DOLLAR / max(p1, 1e-6))))
                        max_qty2 = max(1, int(math.floor(PER_PAIR_MAX_DOLLAR / max(p2, 1e-6))))
                        qty1 = max(1, min(qty1, max_qty1))
                        qty2 = max(1, min(qty2, max_qty2))

                        direction = "SHORT" if enter_short else "LONG"
                        leg1_action, leg2_action = ("SELL","BUY") if enter_short else ("BUY","SELL")

                        print()
                        print(f"Total position new {pair_position_count() + 1} " + "{")
                        print(f"ENTRY L{eligible_level}: {direction}  {t1}/{t2}  |  corr={corr:.4f}  "
                              f"| adaptive_entry_z={entry_z_dynamic:.3f}  |  adaptive_exit_z={exit_z_dynamic:.3f}")
                        print()
                        print(f"{leg1_action} {t1} @{p1:.4f} | qty = {qty1} | leg_notional ≈ ${qty1 * p1:,.2f}")
                        print(f"{leg2_action} {t2} @{p2:.4f} | qty = {qty2} | leg_notional ≈ ${qty2 * p2:,.2f}")
                        print()
                        print(f"z_score(last) = {z_last:.4f}  |  spread(log) = {spread_val_log:.6f}")
                        print(f"gap$ (p1 - beta*p2) = ${gap_dollars:.4f}  |  min_gap_req$ ≈ ${min_gap_required:.4f}")
                        print(f"beta = {beta:.6f}  |  spread_std_log = {spread_std:.6f}")
                        print()
                        print(f"confidence(from corr) = {conf:.3f}  |  signal_strength = {strength:.3f}")
                        print(f"target_notional_per_leg = ${target_notional:,.2f}  |  level_frac = {frac:.2f}")
                        print(f"stop_loss_z = {STOP_LOSS_Z:.2f}  |  max_hold_bars = {MAX_HOLD_BARS}  |  give_up bars = {NO_IMPROVEMENT_BARS}")
                        print("}")

                        try:
                            if enter_short:
                                submit_order_safe(symbol=t1, qty=qty1, side="sell", type="market", time_in_force="gtc")
                                time.sleep(ORDER_THROTTLE_SECONDS)
                                submit_order_safe(symbol=t2, qty=qty2, side="buy", type="market", time_in_force="gtc")
                            else:
                                submit_order_safe(symbol=t1, qty=qty1, side="buy", type="market", time_in_force="gtc")
                                time.sleep(ORDER_THROTTLE_SECONDS)
                                submit_order_safe(symbol=t2, qty=qty2, side="sell", type="market", time_in_force="gtc")

                            st["pos"] = -1 if enter_short else +1
                            st["qty1"], st["qty2"] = qty1, qty2
                            st["entry_time"] = datetime.now(EST)
                            st["entry_z"] = z_last
                            st["entry_spread"] = spread_val_log
                            st["worst_abs_z"] = abs(z_last)
                            st["bars_since_entry"] = 0
                            st["scale_level"] = eligible_level
                            pair_state[key] = st
                            last_trade_bar_idx[key] = bar_index
                            used_notional += (qty1 * p1 + qty2 * p2)
                        except Exception as e:
                            print(f"Order error entering {t1}-{t2}: {e}")
                            traceback.print_exc()

                    # Scale-in (already in a position)
                    elif pos != 0 and eligible_level >= 0 and eligible_level > st["scale_level"]:
                        frac = ENTRY_FRACTIONS[eligible_level]
                        leg_notional = max(target_notional * frac, 0.0)

                        abs_beta = max(abs(beta), 1e-6)

                        add_qty1 = int(max(1, math.floor(leg_notional / max(p1, 1e-6))))
                        add_qty2 = int(max(1, math.floor((leg_notional * abs_beta) / max(p2, 1e-6))))

                        max_add_qty1 = max(1, int(math.floor(PER_PAIR_MAX_DOLLAR / max(p1, 1e-6))))
                        max_add_qty2 = max(1, int(math.floor(PER_PAIR_MAX_DOLLAR / max(p2, 1e-6))))
                        add_qty1 = max(1, min(add_qty1, max_add_qty1))
                        add_qty2 = max(1, min(add_qty2, max_add_qty2))


                        # order in direction of current position
                        try:
                            if st["pos"] == -1:  # short spread (SELL t1, BUY t2)
                                submit_order_safe(symbol=t1, qty=add_qty1, side="sell", type="market", time_in_force="gtc")
                                time.sleep(ORDER_THROTTLE_SECONDS)
                                submit_order_safe(symbol=t2, qty=add_qty2, side="buy", type="market", time_in_force="gtc")
                            else:  # long spread (BUY t1, SELL t2)
                                submit_order_safe(symbol=t1, qty=add_qty1, side="buy", type="market", time_in_force="gtc")
                                time.sleep(ORDER_THROTTLE_SECONDS)
                                submit_order_safe(symbol=t2, qty=add_qty2, side="sell", type="market", time_in_force="gtc")

                            st["qty1"] += add_qty1
                            st["qty2"] += add_qty2
                            st["scale_level"] = eligible_level
                            last_trade_bar_idx[key] = bar_index
                            used_notional += (add_qty1 * p1 + add_qty2 * p2)

                            print(f"[SCALE-IN] {t1}/{t2} level {eligible_level} added "
                                  f"(+{add_qty1} {t1}, +{add_qty2} {t2}) | |z|={abs(z_last):.2f}")
                        except Exception as e:
                            print(f"Order error scale-in {t1}-{t2}: {e}")
                            traceback.print_exc()

                    # === EXIT / SCALE-OUT conditions ===
                    if pos != 0:
                        st["worst_abs_z"] = max(st["worst_abs_z"], abs(z_last))
                        st["bars_since_entry"] = st.get("bars_since_entry", 0) + 1

                        force_exit = False
                        reason = None
                        if abs(z_last) >= STOP_LOSS_Z:
                            force_exit = True
                            reason = f"stop_loss_z {z_last:.2f} >= {STOP_LOSS_Z}"

                        if not force_exit and st["bars_since_entry"] >= NO_IMPROVEMENT_BARS:
                            improved = (st["worst_abs_z"] - abs(z_last)) >= IMPROVE_EPS
                            if not improved:
                                force_exit = True
                                reason = f"no improvement {st['bars_since_entry']} bars (|z| stayed near {st['worst_abs_z']:.2f})"

                        if not force_exit and abs(z_last) <= exit_z_dynamic:
                            force_exit = True
                            reason = "z within exit threshold"

                        # partial scale-out on intermediate convergence
                        if not force_exit:
                            did_partial = False
                            for m, frac_out in zip(EXIT_PARTIAL_MULTIPLIERS, EXIT_PARTIAL_FRACTIONS):
                                if abs(z_last) <= entry_z_dynamic * m and st["scale_level"] >= 0:
                                    # close a fraction of current position
                                    part_qty1 = int(max(round(st.get('qty1', 0) * frac_out), 0))
                                    part_qty2 = int(max(round(st.get('qty2', 0) * frac_out), 0))
                                    if part_qty1 > 0 or part_qty2 > 0:
                                        try:
                                            if st['pos'] == +1:
                                                if part_qty1 > 0:
                                                    submit_order_safe(symbol=t1, qty=part_qty1, side="sell", type="market", time_in_force="gtc")
                                                    time.sleep(ORDER_THROTTLE_SECONDS)
                                                if part_qty2 > 0:
                                                    submit_order_safe(symbol=t2, qty=part_qty2, side="buy", type="market", time_in_force="gtc")
                                                    time.sleep(ORDER_THROTTLE_SECONDS)
                                            elif st['pos'] == -1:
                                                if part_qty1 > 0:
                                                    submit_order_safe(symbol=t1, qty=part_qty1, side="buy", type="market", time_in_force="gtc")
                                                    time.sleep(ORDER_THROTTLE_SECONDS)
                                                if part_qty2 > 0:
                                                    submit_order_safe(symbol=t2, qty=part_qty2, side="sell", type="market", time_in_force="gtc")
                                                    time.sleep(ORDER_THROTTLE_SECONDS)
                                            st['qty1'] -= part_qty1
                                            st['qty2'] -= part_qty2
                                            st['scale_level'] = max(-1, st['scale_level'] - 1)
                                            last_trade_bar_idx[key] = bar_index
                                            print(f"[SCALE-OUT] {t1}/{t2} partial close @|z|={abs(z_last):.2f} "
                                                  f"(-{part_qty1} {t1}, -{part_qty2} {t2})")
                                            did_partial = True
                                        except Exception as e:
                                            print(f"Order error partial exit {t1}-{t2}: {e}")
                                            traceback.print_exc()
                                    break 

                            # refresh state
                            pair_state[key] = st

                        if force_exit and pos != 0:
                            print()
                            print(f"Total position exit {pair_position_count()} " + "{")
                            print(f"EXIT: {('LONG' if st['pos']>0 else 'SHORT')} {t1}/{t2}  |  reason = {reason}")
                            print()
                            print(f"Close {t1} qty={st.get('qty1',0)}  |  Close {t2} qty={st.get('qty2',0)}")
                            print()
                            print(f"z_last = {z_last:.4f}  |  adaptive_exit_z = {exit_z_dynamic:.4f}  |  entry_z = {st.get('entry_z')}")
                            print(f"entry_time = {st.get('entry_time')}  |  entry_spread = {st.get('entry_spread')}")
                            print("}")

                            try:
                                if st['pos'] == +1:
                                    if st.get('qty1', 0) > 0:
                                        submit_order_safe(symbol=t1, qty=st['qty1'], side="sell", type="market", time_in_force="gtc")
                                        time.sleep(ORDER_THROTTLE_SECONDS)
                                    if st.get('qty2', 0) > 0:
                                        submit_order_safe(symbol=t2, qty=st['qty2'], side="buy", type="market", time_in_force="gtc")
                                        time.sleep(ORDER_THROTTLE_SECONDS)
                                elif st['pos'] == -1:
                                    if st.get('qty1', 0) > 0:
                                        submit_order_safe(symbol=t1, qty=st['qty1'], side="buy", type="market", time_in_force="gtc")
                                        time.sleep(ORDER_THROTTLE_SECONDS)
                                    if st.get('qty2', 0) > 0:
                                        submit_order_safe(symbol=t2, qty=st['qty2'], side="sell", type="market", time_in_force="gtc")
                                        time.sleep(ORDER_THROTTLE_SECONDS)
                                st['pos'] = 0
                                st['qty1'], st['qty2'] = 0, 0
                                st['last_exit_bar'] = df.index[-1]
                                last_trade_bar_idx[key] = bar_index
                                st['entry_time'] = None
                                st['entry_z'] = None
                                st['scale_level'] = -1
                                st['worst_abs_z'] = 0.0
                                st['bars_since_entry'] = 0
                                pair_state[key] = st
                            except Exception as e:
                                print(f"Order error exiting {t1}-{t2}: {e}")
                                traceback.print_exc()

                except Exception as e:
                    print(f"[WARN] Skipping pair {t1}-{t2} due to error: {e}")
                    traceback.print_exc()

            # End loop through pairs

            # Print summary of current open pairs with blocks and a blank line between them
            open_idx = 0
            for (t1, t2), st in pair_state.items():
                if st.get('pos', 0) != 0:
                    open_idx += 1
                    print()
                    print(f"Total position open {open_idx} " + "{")
                    print(f"{'LONG' if st['pos']>0 else 'SHORT'} {t1}/{t2}")
                    print(f"qtys: {st.get('qty1',0)} / {st.get('qty2',0)}")
                    print(f"beta={st.get('beta'):.6f} corr={st.get('corr'):.4f}")
                    print(f"entry_time={st.get('entry_time')} entry_z={st.get('entry_z')}")
                    print(f"scale_level={st.get('scale_level')} worst|z|={st.get('worst_abs_z'):.2f} bars_in={st.get('bars_since_entry',0)}")
                    print("}")

            print()
            print(f"===== Loop finished. Sleeping {LOG_EVERY_X_SECS} seconds =====")
            time.sleep(LOG_EVERY_X_SECS)

        except KeyboardInterrupt:
            print("KeyboardInterrupt - flattening and exiting.")
            try:
                positions = list_positions_safe()
            except Exception:
                positions = None
            flatten_all(positions_cache=positions)
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error in main loop: {e}")
            traceback.print_exc()
            time.sleep(5)

def main():
    print("Starting pairs trading main loop (with cointegration filtering = {})".format(HAVE_COINTEGRATION))

    # --- Run preprocessing before market open ---
    market_open = dt_time(9, 30)
    prep_start = dt_time(9, 5)
    now = now_est().time()

    if now < prep_start:
        eastern = timezone("US/Eastern")
        prep_start_dt = eastern.localize(datetime.combine(datetime.today(), prep_start))
        current_time = now_est()
        wait_secs = (prep_start_dt - current_time).total_seconds()

        prep_time_str = prep_start.strftime('%I:%M%p').lstrip('0')

        if wait_secs > 0:
            print(f"Waiting {wait_secs/60:.1f} minutes until preprocessing at {prep_time_str} EST...")
            time.sleep(wait_secs)
        else:
            print(f"Prep start time ({prep_time_str} EST) already passed. Skipping wait...")


    print("Running preprocessing to build initial pair list...")
    try:
        trade_loop() 
    except Exception as e:
        print("Preprocessing failed:", e)
        return

    now = now_est().time()
    if now < market_open:
        wait_secs = (datetime.combine(datetime.today(), market_open) - now_est()).total_seconds()
        print(f"Preprocessing done. Waiting {wait_secs/60:.1f} minutes for market to open...")
        time.sleep(wait_secs)

    trade_loop()


if __name__ == "__main__":
    main()
