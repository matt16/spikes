import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

# Optional: Stooq via pandas-datareader (falls installiert)
try:
    from pandas_datareader import data as pdr
    HAS_PDR = True
except Exception:
    HAS_PDR = False

INDEX_ORDER = ["S&P500", "EuroStoxx50", "Nikkei225", "FTSE100"]

YF_TICKERS = {
    "S&P500": "^GSPC",
    "EuroStoxx50": "^STOXX50E",
    "Nikkei225": "^N225",
    "FTSE100": "^FTSE",
}

# Kandidaten für Stooq (wir probieren der Reihe nach)
STOOQ_TICKER_CANDIDATES = {
    "S&P500": ["spx", "^spx", "us_spx"],
    "EuroStoxx50": ["stoxx50e", "^stoxx50e", "sx5e", "^sx5e"],
    "Nikkei225": ["nikkei", "^nikkei", "nkx", "^nkx"],
    "FTSE100": ["ftse100", "^ftse100", "ukx", "^ukx"],
}

def _print_src_used(name: str, src: str, ok: bool):
    status = "OK" if ok else "…keine Daten"
    print(f"[{name:<12}] Quelle {src:<12}: {status}")

def _ensure_series_close(df_or_series) -> pd.Series:
    """Extrahiere eine 1D-Series 'Close' (float), egal wie yfinance/pdr das Objekt liefert."""
    if isinstance(df_or_series, pd.Series):
        s = df_or_series
    else:
        df = df_or_series
        # zuerst versuchen, 'Close' als Spalte zu nehmen
        if isinstance(df.columns, pd.MultiIndex):
            # Bei MultiIndex (kommt bei Multi-Ticker vor, hier eigentlich nicht): ('Close', ticker)
            if "Close" in df:
                s = df["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.squeeze(axis=1)
            else:
                # Fallback: nimm letzte Spalte
                s = df.iloc[:, -1]
        else:
            if "Close" in df.columns:
                s = df["Close"]
            else:
                # wenn nur eine Spalte vorhanden ist
                if df.shape[1] == 1:
                    s = df.iloc[:, 0]
                else:
                    # Fallback: suche case-insensitive nach 'close'
                    close_cols = [c for c in df.columns if str(c).lower() == "close"]
                    s = df[close_cols[0]] if close_cols else df.iloc[:, 0]

    # Wenn wider Erwarten noch 2D: auf 1 Spalte reduzieren
    if isinstance(s, pd.DataFrame):
        s = s.squeeze(axis=1)
    s = s.astype("float64").sort_index()
    return s

def fetch_yahoo_daily(name: str, start: str, end: str) -> pd.Series:
    ticker = YF_TICKERS[name]
    df = yf.download(
        ticker, start=start, end=end, interval="1d",
        auto_adjust=True, group_by="column", progress=False, threads=True
    )
    if df is None or df.empty:
        _print_src_used(name, "Yahoo", False)
        return pd.Series(dtype=float)

    s = _ensure_series_close(df)
    _print_src_used(name, "Yahoo", True)
    return s

def fetch_stooq_daily(name: str, start: str, end: str) -> pd.Series:
    if not HAS_PDR:
        _print_src_used(name, "Stooq(N/A)", False)
        return pd.Series(dtype=float)

    for sym in STOOQ_TICKER_CANDIDATES.get(name, []):
        try:
            df = pdr.DataReader(sym, "stooq", start=start, end=end)
            if df is not None and not df.empty:
                df = df.sort_index()
                s = _ensure_series_close(df)
                _print_src_used(name, f"Stooq:{sym}", True)
                return s
        except Exception:
            continue
    _print_src_used(name, "Stooq", False)
    return pd.Series(dtype=float)

def merged_daily(name: str, start_ext: str, end: str) -> pd.Series:
    """Yahoo bevorzugt, Stooq füllt nur Lücken."""
    s_yf = fetch_yahoo_daily(name, start_ext, end)
    s_sq = fetch_stooq_daily(name, start_ext, end)

    if s_yf.empty and s_sq.empty:
        return pd.Series(dtype=float)

    if s_yf.empty:
        return s_sq
    if s_sq.empty:
        return s_yf

    # Robustes Zusammenführen mit Schlüssel-Labels als Spalten
    df = pd.concat({"yf": s_yf, "stooq": s_sq}, axis=1)

    col_yf = df.get("yf")
    col_sq = df.get("stooq")
    if col_yf is None and col_sq is None:
        return pd.Series(dtype=float)
    if col_yf is None:
        merged = col_sq
    elif col_sq is None:
        merged = col_yf
    else:
        merged = col_yf.combine_first(col_sq)

    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged

def compute_weekly_returns_table(start: str, end: str, week_anchor: str = "W-FRI") -> pd.DataFrame:
    """
    - lädt tägliche Preise (Yahoo→Stooq) mit Startpuffer (start−14d),
    - resampled auf W-FRI, berechnet Weekly Returns in %,
    - filtert auf Wochen >= start,
    - liefert 4 Zeilen (Indizes) × Spalten = Wochen (ISO-String).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ext = (start_ts - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end_str = end_ts.strftime("%Y-%m-%d")

    # tägliche Preise aller Indizes
    daily_dict = {}
    for name in INDEX_ORDER:
        s = merged_daily(name, start_ext, end_str)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        daily_dict[name] = s

    daily = pd.DataFrame(daily_dict)

    # gemeinsame Wochen (Freitag als Wochenanker)
    weekly_prices = daily.resample(week_anchor).last()

    # Weekly Returns in % (ohne %-Zeichen)
    weekly_rets_pct = weekly_prices.pct_change() * 100.0

    # nur Wochen behalten, deren Wochenende >= start liegt
    weekly_rets_pct = weekly_rets_pct.loc[weekly_rets_pct.index >= start_ts]

    # gewünschte Form: Zeilen = Indizes, Spalten = Wochen
    table = weekly_rets_pct.T
    table.columns = [d.strftime("%Y-%m-%d") for d in table.columns]
    table.index.name = "Index"

    # Reihenfolge fixieren
    table = table.reindex(INDEX_ORDER)
    return table

def main(start: str, end: str, out_prefix: str = "weekly_returns"):
    table = compute_weekly_returns_table(start, end)
    filename = f"{out_prefix}.csv"
    table.to_csv(filename, na_rep="NaN")
    print("\n--- Ergebnis ---")
    print(f"Gespeichert: {Path().resolve() / filename}")
    print(f"Form: {table.shape} (Zeilen=4, Spalten=Wochen)")
    with pd.option_context("display.max_columns", 8):
        print(table.iloc[:, :5])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly Returns mit Yahoo→Stooq-Fallback, 4 Zeilen × Wochen-Spalten.")
    parser.add_argument("--start", type=str, default="2007-06-01")
    parser.add_argument("--end", type=str, default="2025-06-01")
    args = parser.parse_args()
    main(start=args.start, end=args.end)
