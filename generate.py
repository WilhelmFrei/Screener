#!/usr/bin/env python3
"""
Stock Screener — génère docs/index.html à partir de tickers.csv
Les tickers sont dans tickers.csv (format: ticker,groupe,pea)
Modifiez tickers.csv pour ajouter/retirer des actions sans toucher ce script.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
import csv
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
SCORE_MIN        = 35
TOP_N            = 300
JOURS_HISTORIQUE = 260
POIDS_VALUE      = 30
POIDS_GROWTH     = 25
POIDS_DIVIDENDE  = 15
POIDS_TECHNIQUE  = 30
RSI_SURVENTE     = 45
CCI_SURVENTE     = -100
STOCH_SURVENTE   = 30
ADX_MIN          = 20
PER_MAX          = 30
PBV_MAX          = 8
ROE_MIN          = 8

# ══════════════════════════════════════════════════════════════
# LECTURE DES TICKERS DEPUIS tickers.csv
# ══════════════════════════════════════════════════════════════
tickers_data = []
PEA_SET = set()
csv_path = Path('tickers.csv')

if not csv_path.exists():
    print('ERREUR: tickers.csv introuvable. Placez-le a la racine du repo.')
    exit(1)

with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        tk = row['ticker'].strip()
        if tk:
            tickers_data.append({'ticker': tk, 'groupe': row.get('groupe',''), 'pea': row.get('pea','0')})
            if row.get('pea','0') == '1':
                PEA_SET.add(tk)

tickers = list(dict.fromkeys([r['ticker'] for r in tickers_data]))
print('Tickers charges depuis tickers.csv: ' + str(len(tickers)))
print('Dont PEA eligibles: ' + str(len(PEA_SET)))

# ══════════════════════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ══════════════════════════════════════════════════════════════
def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def calc_macd(s, f=12, sl=26, sig=9):
    m = s.ewm(span=f, adjust=False).mean() - s.ewm(span=sl, adjust=False).mean()
    sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg

def calc_cci(h, l, c, p=20):
    tp = (h + l + c) / 3
    sma = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def calc_stoch(h, l, c, k=14, d=3):
    lm = l.rolling(k).min()
    hm = h.rolling(k).max()
    sk = 100 * (c - lm) / (hm - lm).replace(0, np.nan)
    return sk, sk.rolling(d).mean()

def calc_adx(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    dp = (h - h.shift()).clip(lower=0)
    dm = (l.shift() - l).clip(lower=0)
    dp = dp.where(dp > dm, 0)
    dm = dm.where(dm > dp, 0)
    dip = 100 * dp.rolling(p).mean() / atr.replace(0, np.nan)
    dim = 100 * dm.rolling(p).mean() / atr.replace(0, np.nan)
    dx = 100 * (dip - dim).abs() / (dip + dim).replace(0, np.nan)
    return dx.rolling(p).mean()

def calc_boll(s, p=20, n=2):
    m = s.rolling(p).mean()
    sd = s.rolling(p).std()
    lower = m - n * sd
    px = float(s.iloc[-1])
    rng = float((m + n * sd - lower).iloc[-1])
    if rng == 0:
        return pd.Series([np.nan])
    return pd.Series([(px - float(lower.iloc[-1])) / rng * 100])

def calc_roc(s, p=12):
    """Rate of Change"""
    return ((s - s.shift(p)) / s.shift(p).replace(0, np.nan)) * 100

def calc_williams_r(h, l, c, p=14):
    """Williams %R"""
    hh = h.rolling(p).max()
    ll = l.rolling(p).min()
    return -100 * (hh - c) / (hh - ll).replace(0, np.nan)

# ══════════════════════════════════════════════════════════════
# FONCTIONS DE SCORE
# ══════════════════════════════════════════════════════════════
def score_value(info):
    p = 0
    per = info.get('trailingPE') or info.get('forwardPE')
    pbv = info.get('priceToBook')
    ev  = info.get('enterpriseToEbitda')
    ps  = info.get('priceToSalesTrailing12Months')
    # P/FCF approximé via marketCap / freeCashflow
    mcap = info.get('marketCap')
    fcf  = info.get('freeCashflow')
    pfcf = (mcap / fcf) if (mcap and fcf and fcf > 0) else None
    # Graham Number: sqrt(22.5 * BPA * VNC)
    bpa_val = info.get('trailingEps')
    bvps    = info.get('bookValue')
    px_cur  = info.get('currentPrice') or info.get('regularMarketPrice')
    graham  = None
    if bpa_val and bvps and bpa_val > 0 and bvps > 0:
        graham_num = (22.5 * bpa_val * bvps) ** 0.5
        if px_cur and px_cur > 0:
            graham = round((graham_num / px_cur - 1) * 100, 1)  # % au-dessus/dessous

    if per and per > 0: p += 40 if per <= 10 else 28 if per <= 15 else 15 if per <= PER_MAX else 0
    if pbv and pbv > 0: p += 30 if pbv <= 1  else 20 if pbv <= 2  else 10 if pbv <= PBV_MAX else 0
    if ev  and ev  > 0: p += 30 if ev  <= 8  else 20 if ev  <= 12 else 10 if ev  <= 15 else 0
    # Bonus P/S
    if ps and ps > 0:   p += 10 if ps <= 1 else 6 if ps <= 2 else 3 if ps <= 4 else 0
    # Bonus P/FCF
    if pfcf and pfcf > 0: p += 10 if pfcf <= 10 else 6 if pfcf <= 20 else 3 if pfcf <= 30 else 0
    # Bonus Graham
    if graham is not None: p += 15 if graham >= 20 else 10 if graham >= 0 else 0
    return min(100, p)

def score_growth(info):
    p = 0
    roe = info.get('returnOnEquity')
    eg  = info.get('earningsGrowth')
    peg = info.get('pegRatio')
    # Nouveaux: qualité bilan
    cr  = info.get('currentRatio')
    de  = info.get('debtToEquity')
    fcf = info.get('freeCashflow')
    mcap = info.get('marketCap')
    fcf_yield = (fcf / mcap * 100) if (fcf and mcap and mcap > 0) else None

    if roe: rp = roe * 100; p += 35 if rp >= 25 else 25 if rp >= 15 else 12 if rp >= ROE_MIN else 0
    if eg:  ep = eg  * 100; p += 40 if ep >= 25 else 28 if ep >= 15 else 15 if ep > 0 else 0
    if peg and peg > 0: p += 25 if peg <= 1 else 15 if peg <= 1.5 else 8 if peg <= 2.5 else 0
    # Current Ratio (liquidité)
    if cr: p += 8 if cr >= 2 else 5 if cr >= 1.5 else 2 if cr >= 1 else 0
    # Debt/Equity (levier)
    if de and de >= 0: p += 10 if de <= 30 else 6 if de <= 80 else 3 if de <= 150 else 0
    # FCF Yield
    if fcf_yield: p += 10 if fcf_yield >= 8 else 6 if fcf_yield >= 5 else 3 if fcf_yield >= 2 else 0
    return min(100, p)

def score_div(info):
    p = 0
    yld = info.get('dividendYield')
    po  = info.get('payoutRatio')
    if yld: yp = yld * 100; p += 60 if yp >= 6 else 45 if yp >= 4 else 30 if yp >= 3 else 15 if yp > 0 else 0
    if po and 0 < po < 0.8: p += 20
    elif po and po <= 1:     p += 10
    return min(100, p)

def score_tech(hist):
    if hist is None or len(hist) < 30:
        return 0, {}
    try:
        c = hist['Close'].dropna()
        h = hist['High'].dropna()
        l = hist['Low'].dropna()
        if len(c) < 30:
            return 0, {}
        r     = calc_rsi(c).iloc[-1]
        ml, ms, mh = calc_macd(c)
        mh_v  = mh.iloc[-1]; ml_v = ml.iloc[-1]; ms_v = ms.iloc[-1]
        cc    = calc_cci(h, l, c).iloc[-1]
        sk, _ = calc_stoch(h, l, c); st = sk.iloc[-1]
        ax    = calc_adx(h, l, c).iloc[-1]
        bb    = calc_boll(c).iloc[-1]
        px    = float(c.iloc[-1])
        # Nouveaux indicateurs momentum
        roc_v = calc_roc(c).iloc[-1] if len(c) >= 13 else np.nan
        wr_v  = calc_williams_r(h, l, c).iloc[-1] if len(c) >= 14 else np.nan
        mms   = {}
        for per in [20, 50, 200]:
            if len(c) >= per:
                mms[per] = float(c.rolling(per).mean().iloc[-1])
        pts = 0; sigs = []
        if not np.isnan(r):
            if   r <= 30:           pts += 28; sigs.append('RSI ' + str(round(r)) + ' GRN')
            elif r <= RSI_SURVENTE: pts += 18; sigs.append('RSI ' + str(round(r)))
            elif r <= 50:           pts += 8
        if not np.isnan(mh_v):
            if   mh_v > 0 and ml_v > ms_v: pts += 20; sigs.append('MACD GRN')
            elif mh_v > 0:                  pts += 12; sigs.append('MACD')
            elif mh_v > -0.1:               pts += 5
        if not np.isnan(cc):
            if   cc < CCI_SURVENTE: pts += 18; sigs.append('CCI ' + str(round(cc)) + ' GRN')
            elif 50 <= cc <= 150:   pts += 12; sigs.append('CCI ' + str(round(cc)))
            elif 0 < cc < 50:       pts += 6
        if not np.isnan(st):
            if   st <= 20:               pts += 14; sigs.append('Stoch ' + str(round(st)) + ' GRN')
            elif st <= STOCH_SURVENTE:   pts += 9;  sigs.append('Stoch ' + str(round(st)))
        above = sum(1 for v in mms.values() if px > v)
        if above == len(mms) > 0: pts += 15; sigs.append('MM OK')
        elif above >= 2:           pts += 8
        if not np.isnan(ax) and ax >= ADX_MIN:
            pts += 5; sigs.append('ADX ' + str(round(ax)))
        # Bonus Williams %R
        if not np.isnan(wr_v):
            if wr_v <= -80: pts += 8; sigs.append('W%R ' + str(round(wr_v)))
            elif wr_v <= -60: pts += 4
        # Bonus RoC positif
        if not np.isnan(roc_v) and roc_v > 5:
            pts += 5; sigs.append('RoC+')
        det = {
            'rsi':   round(float(r), 1)    if not np.isnan(r)    else None,
            'macd':  round(float(mh_v), 3) if not np.isnan(mh_v) else None,
            'cci':   round(float(cc), 1)   if not np.isnan(cc)   else None,
            'stoch': round(float(st), 1)   if not np.isnan(st)   else None,
            'adx':   round(float(ax), 1)   if not np.isnan(ax)   else None,
            'bb':    round(float(bb), 2)   if not np.isnan(bb)   else None,
            'roc':   round(float(roc_v), 1) if not np.isnan(roc_v) else None,
            'willr': round(float(wr_v), 1)  if not np.isnan(wr_v)  else None,
            'signaux': ' | '.join(sigs),
        }
        for per, val in mms.items():
            det['mm' + str(per)] = round(((px / val) - 1) * 100, 1)
        return min(100, pts), det
    except Exception:
        return 0, {}

# ══════════════════════════════════════════════════════════════
# SCAN PRINCIPAL
# ══════════════════════════════════════════════════════════════
d_fin = datetime.today()
d_deb = d_fin - timedelta(days=JOURS_HISTORIQUE)
print('Scan de ' + str(len(tickers)) + ' actions...')

BATCH = 50; all_hist = {}
scan_errors = []
for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i+BATCH]
    try:
        data = yf.download(batch, start=d_deb.strftime('%Y-%m-%d'),
                           end=d_fin.strftime('%Y-%m-%d'),
                           group_by='ticker', auto_adjust=True,
                           progress=False, threads=True)
        for tk in batch:
            try:
                all_hist[tk] = data[tk] if len(batch) > 1 and tk in data.columns.get_level_values(0) else (data if len(batch) == 1 else None)
            except Exception:
                all_hist[tk] = None
    except Exception:
        for tk in batch:
            all_hist[tk] = None
    print('  Prix: ' + str(min(i+BATCH, len(tickers))) + '/' + str(len(tickers)))

resultats = []
for idx, ticker in enumerate(tickers):
    try:
        hist  = all_hist.get(ticker)
        if idx > 0 and idx % 50 == 0:
            time.sleep(2)
        tk_obj = yf.Ticker(ticker)
        try:
            fi = tk_obj.fast_info
            px_fast = getattr(fi, 'lastPrice', None) or getattr(fi, 'last_price', None)
            dev_fast = getattr(fi, 'currency', None)
        except Exception:
            fi = None; px_fast = None; dev_fast = None
        info = {}
        info_error = None
        for attempt in range(2):
            try:
                info = tk_obj.info or {}
                if info: break
            except Exception as e:
                info_error = str(e)
                if attempt == 0: time.sleep(3)
        if not info:
            scan_errors.append({'ticker': ticker, 'raison': 'info vide', 'detail': info_error or 'N/A'})
        nom   = info.get('longName') or info.get('shortName') or ticker
        sect  = info.get('sector') or 'N/A'
        pays  = (info.get('country') or '')[:2].upper() or '?'
        dev   = dev_fast or info.get('currency') or ('EUR' if any(x in ticker for x in ['.PA','.DE','.AS','.BR','.MC','.MI','.SW','.CO','.ST','.OL','.HE','.LS','.L']) else 'USD')
        px_h  = float(hist['Close'].iloc[-1]) if hist is not None and len(hist) > 0 else None
        prix  = px_fast or info.get('currentPrice') or info.get('regularMarketPrice') or px_h
        per   = info.get('trailingPE') or info.get('forwardPE')
        pbv   = info.get('priceToBook')
        roe_r = info.get('returnOnEquity'); roe = round(roe_r * 100, 1) if roe_r else None
        mg_r  = info.get('profitMargins');  mg  = round(mg_r  * 100, 1) if mg_r  else None
        yld_r = info.get('dividendYield');  yld = round(yld_r * 100, 2) if yld_r else None
        peg   = info.get('pegRatio')
        eg_r  = info.get('earningsGrowth'); eg  = round(eg_r  * 100, 1) if eg_r  else None
        ev    = info.get('enterpriseToEbitda')
        ps_r  = info.get('priceToSalesTrailing12Months'); ps = round(float(ps_r), 2) if ps_r else None
        cr_r  = info.get('currentRatio');  cr  = round(float(cr_r), 2)  if cr_r else None
        de_r  = info.get('debtToEquity');  de  = round(float(de_r), 1)  if de_r else None
        fcf_r = info.get('freeCashflow');  mcap_r = info.get('marketCap')
        fcfy  = round(fcf_r / mcap_r * 100, 1) if (fcf_r and mcap_r and mcap_r > 0) else None
        pfcf_v = round(mcap_r / fcf_r, 1) if (fcf_r and mcap_r and fcf_r > 0) else None
        bpa_v  = info.get('trailingEps'); bvps_v = info.get('bookValue')
        px_cur_v = info.get('currentPrice') or info.get('regularMarketPrice') or px_h
        graham_v = None
        if bpa_v and bvps_v and bpa_v > 0 and bvps_v > 0 and px_cur_v and px_cur_v > 0:
            gn = (22.5 * bpa_v * bvps_v) ** 0.5
            graham_v = round((gn / px_cur_v - 1) * 100, 1)

        sv = score_value(info)
        sg = score_growth(info)
        sd = score_div(info)
        st, td = score_tech(hist)

        # Score momentum global: RoC + Williams %R + MACD histo
        roc_v2  = td.get('roc')
        willr_v2 = td.get('willr')
        macd_v2  = td.get('macd')
        smom = 0
        if roc_v2 is not None:  smom += 40 if roc_v2 > 10 else 25 if roc_v2 > 5 else 10 if roc_v2 > 0 else 0
        if willr_v2 is not None: smom += 35 if willr_v2 <= -80 else 20 if willr_v2 <= -60 else 8 if willr_v2 <= -40 else 0
        if macd_v2 is not None:  smom += 25 if macd_v2 > 0 else 10 if macd_v2 > -0.1 else 0
        smom = min(100, smom)

        # Alerte divergence: fond fort + tech faible OU fond faible + tech fort
        fond_score = round(sv * POIDS_VALUE / (POIDS_VALUE+POIDS_GROWTH+POIDS_DIVIDENDE)
                         + sg * POIDS_GROWTH / (POIDS_VALUE+POIDS_GROWTH+POIDS_DIVIDENDE)
                         + sd * POIDS_DIVIDENDE / (POIDS_VALUE+POIDS_GROWTH+POIDS_DIVIDENDE))
        divergence = None
        if fond_score >= 65 and st <= 35:
            divergence = 'FOND+'
        elif fond_score <= 35 and st >= 65:
            divergence = 'TECH+'

        total_f = POIDS_VALUE + POIDS_GROWTH + POIDS_DIVIDENDE
        sf = round(sv * POIDS_VALUE / total_f + sg * POIDS_GROWTH / total_f + sd * POIDS_DIVIDENDE / total_f)
        score = round(sf * (1 - POIDS_TECHNIQUE / 100) + st * (POIDS_TECHNIQUE / 100))
        if score < SCORE_MIN:
            continue

        # Sparkline: 60 derniers jours normalisés 0-100
        sparkline = []
        if hist is not None and len(hist) >= 10:
            try:
                closes = hist['Close'].dropna().tail(60).tolist()
                mn, mx = min(closes), max(closes)
                rng = mx - mn
                if rng > 0:
                    sparkline = [round((v - mn) / rng * 100, 1) for v in closes]
            except Exception:
                sparkline = []

        row = {
            'ticker': ticker, 'nom': nom[:35], 'secteur': sect,
            'pays': pays, 'pea': 1 if ticker in PEA_SET else 0,
            'devise': dev,
            'prix':  round(float(prix), 2) if prix else None,
            'per':   round(float(per),  1) if per and per > 0 else None,
            'pbv':   round(float(pbv),  2) if pbv else None,
            'roe': roe, 'marge': mg, 'div': yld, 'bpa': eg,
            'ev':    round(float(ev), 1) if ev else None,
            'peg':   round(float(peg), 2) if peg and peg > 0 else None,
            'ps': ps, 'cr': cr, 'de': de, 'fcfy': fcfy, 'pfcf': pfcf_v, 'graham': graham_v,
            'sv': sv, 'sg': sg, 'sd': sd, 'st': st, 'smom': smom,
            'score': score, 'divergence': divergence,
            'rsi':   td.get('rsi'),   'macd':  td.get('macd'),
            'cci':   td.get('cci'),   'stoch': td.get('stoch'),
            'adx':   td.get('adx'),   'bb':    td.get('bb'),
            'mm20':  td.get('mm20'),  'mm50':  td.get('mm50'), 'mm200': td.get('mm200'),
            'roc':   td.get('roc'),   'willr': td.get('willr'),
            'signaux': td.get('signaux', ''),
            'sparkline': sparkline,
        }
        resultats.append(row)
    except Exception as e:
        scan_errors.append({'ticker': ticker, 'raison': 'exception scan', 'detail': str(e)})

resultats.sort(key=lambda x: x['score'], reverse=True)
resultats = resultats[:TOP_N]
print('OK: ' + str(len(resultats)) + ' opportunites sur ' + str(len(tickers)) + ' actions')
if scan_errors:
    print('Erreurs: ' + str(len(scan_errors)) + ' tickers')
    err_path = Path('docs/scan_errors.json')
    err_path.write_text(json.dumps(scan_errors, ensure_ascii=False, indent=2), encoding='utf-8')

# ══════════════════════════════════════════════════════════════
# GENERATION HTML
# ══════════════════════════════════════════════════════════════
DATA  = json.dumps(resultats, ensure_ascii=False, default=str)
DATE  = d_fin.strftime('%d/%m/%Y %H:%M')
NSCAN = str(len(tickers))
SMIN  = str(SCORE_MIN)
CFG   = json.dumps({
    'perMax': PER_MAX, 'pbvMax': PBV_MAX, 'roeMin': ROE_MIN,
    'rsiSurvente': RSI_SURVENTE, 'cciSurvente': CCI_SURVENTE,
    'stochSurvente': STOCH_SURVENTE, 'adxMin': ADX_MIN,
    'poidsValue': POIDS_VALUE, 'poidsGrowth': POIDS_GROWTH,
    'poidsDividende': POIDS_DIVIDENDE, 'poidsTechnique': POIDS_TECHNIQUE,
    'scoreMin': SCORE_MIN,
    # Nouveaux seuils
    'psMax': 4, 'pfcfMax': 30, 'grahamMin': 0,
    'crMin': 1.0, 'deMax': 150, 'fcfyMin': 2,
    'willrSurvente': -60, 'rocMin': 5,
    # Toggles (1=actif, 0=inactif)
    'tPer':1,'tPbv':1,'tEv':1,'tPs':1,'tPfcf':1,'tGraham':1,
    'tRoe':1,'tBpa':1,'tPeg':1,'tCr':1,'tDe':1,'tFcfy':1,
    'tRsi':1,'tMacd':1,'tCci':1,'tStoch':1,'tMm':1,'tAdx':1,'tWillr':1,'tRoc':1,
})

page = []
page.append('<!DOCTYPE html>')
page.append('<html lang="fr"><head>')
page.append('<meta charset="UTF-8">')
page.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
page.append('<title>Stock Screener</title>')
page.append('<style>')
page.append('*{box-sizing:border-box;margin:0;padding:0}')
page.append('body{background:#0a0e1a;color:#e2e8f0;font-family:system-ui,sans-serif}')
page.append('.hdr{background:#111827;border-bottom:1px solid #1e2d45;padding:12px 16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;position:sticky;top:0;z-index:10}')
page.append('.htitle{font-family:monospace;font-size:17px;font-weight:900;color:#00d4aa}')
page.append('.hsub{font-size:10px;color:#64748b;margin-top:2px}')
page.append('.hbtns{display:flex;gap:6px;align-items:center;flex-wrap:wrap}')
page.append('.hbtn{font-size:11px;font-weight:700;padding:7px 13px;border-radius:8px;border:none;cursor:pointer;font-family:monospace;white-space:nowrap}')
page.append('.hbtn-scan{background:linear-gradient(135deg,#00d4aa,#00a87f);color:#000}')
page.append('.hbtn-cfg{background:#1e2d45;color:#00d4aa;border:1px solid #00d4aa44}')
page.append('.hbtn-imp{background:#1e2d45;color:#f59e0b;border:1px solid #f59e0b44}')
page.append('.hbtn-guide{background:#1e2d45;color:#8b5cf6;border:1px solid #8b5cf644}')
page.append('.wrap{max-width:1500px;margin:0 auto;padding:12px}')
page.append('.modal-bg{display:none;position:fixed;inset:0;background:#000000bb;z-index:50;align-items:center;justify-content:center}')
page.append('.modal-bg.open{display:flex}')
page.append('.modal{background:#111827;border:1px solid #1e2d45;border-radius:14px;padding:20px;width:90%;max-width:540px;max-height:90vh;overflow-y:auto}')
page.append('.modal h3{font-family:monospace;font-size:14px;color:#00d4aa;margin-bottom:14px}')
page.append('.modal-close{float:right;background:none;border:none;color:#64748b;font-size:18px;cursor:pointer;margin-top:-4px}')
page.append('.cfg-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px}')
page.append('.cfg-item{background:#0d1520;border-radius:8px;padding:10px}')
page.append('.cfg-item label{display:block;font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.7px;margin-bottom:5px}')
page.append('.cfg-item input{width:100%;background:#111827;border:1px solid #1e2d45;color:#00d4aa;font-size:13px;padding:6px 8px;border-radius:6px;outline:none;font-family:monospace;text-align:right}')
page.append('.cfg-section{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.8px;margin:12px 0 6px;border-top:1px solid #1e2d45;padding-top:10px}')
page.append('.cfg-note{font-size:10px;color:#f59e0b;margin-bottom:10px;line-height:1.5}')
page.append('.modal-btn{width:100%;padding:10px;border-radius:8px;border:none;cursor:pointer;font-family:monospace;font-weight:700;font-size:13px}')
page.append('.modal-btn-ok{background:linear-gradient(135deg,#00d4aa,#00a87f);color:#000;margin-bottom:8px}')
page.append('.modal-btn-cancel{background:#1e2d45;color:#64748b}')
page.append('.imp-area{width:100%;min-height:100px;background:#0d1520;border:1px solid #1e2d45;color:#e2e8f0;font-family:monospace;font-size:12px;padding:10px;border-radius:8px;margin-bottom:10px;resize:vertical}')
page.append('.imp-note{font-size:10px;color:#64748b;margin-bottom:12px;line-height:1.6}')
page.append('.mrow{display:flex;gap:6px;margin-bottom:10px}')
page.append('.mbtn{flex:1;padding:10px 4px;border-radius:10px;border:2px solid #1e2d45;background:#0d1520;cursor:pointer;text-align:center}')
page.append('.mbtn.on{border-color:#00d4aa;background:#00d4aa18}')
page.append('.mico{font-size:16px;margin-bottom:2px}')
page.append('.mlbl{font-size:10px;font-weight:700;color:#64748b;font-family:monospace}')
page.append('.mbtn.on .mlbl{color:#00d4aa}')
page.append('.mdesc{background:#0d1520;border:1px solid #1e2d45;border-radius:8px;padding:8px 12px;font-size:11px;color:#64748b;margin-bottom:10px}')
page.append('.filters{background:#111827;border:1px solid #1e2d45;border-radius:10px;padding:10px 12px;display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end;margin-bottom:10px}')
page.append('.fg{display:flex;flex-direction:column;gap:3px}')
page.append('.fg label{font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.7px}')
page.append('select,input[type=text],input[type=number]{background:#0d1520;border:1px solid #1e2d45;color:#e2e8f0;font-size:12px;padding:7px 8px;border-radius:7px;outline:none;font-family:inherit}')
page.append('.tabs{display:flex;gap:5px;margin-bottom:10px}')
page.append('.tab{padding:7px 14px;border-radius:8px;border:1px solid #1e2d45;background:#0d1520;color:#64748b;cursor:pointer;font-size:11px;font-weight:700;font-family:monospace;display:flex;align-items:center;gap:5px}')
page.append('.tab.on{border-color:#00d4aa;background:#111827;color:#e2e8f0}')
page.append('.bdg{background:#1e2d45;color:#64748b;font-size:9px;font-weight:700;padding:1px 6px;border-radius:20px}')
page.append('.tab.on .bdg{background:#00d4aa;color:#000}')
page.append('.stats{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}')
page.append('.stat{background:#111827;border:1px solid #1e2d45;border-radius:9px;padding:8px 12px;flex:1;min-width:75px}')
page.append('.sl{font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.6px}')
page.append('.sv{font-family:monospace;font-size:17px;font-weight:900;margin-top:2px}')
page.append('.tw{background:#111827;border:1px solid #1e2d45;border-radius:12px;overflow:hidden}')
page.append('.ts{overflow-x:auto;max-height:60vh;overflow-y:auto}')
page.append('table{width:100%;border-collapse:collapse;font-size:11px}')
page.append('thead th{background:#0d1520;padding:8px 8px;font-size:8px;font-family:monospace;text-transform:uppercase;letter-spacing:.7px;color:#64748b;cursor:pointer;border-bottom:1px solid #1e2d45;white-space:nowrap;position:sticky;top:0;z-index:2;text-align:left;user-select:none}')
page.append('thead th:hover{color:#00d4aa}')
page.append('thead th.sorted{color:#00d4aa}')
page.append('tbody td{padding:7px 8px;border-bottom:1px solid #0d1520;vertical-align:middle;white-space:nowrap}')
page.append('tbody tr:nth-child(even) td{background:#0a1020}')
page.append('tbody tr:hover td{background:#152030!important}')
page.append('.tk{font-family:monospace;font-weight:700;color:#00d4aa;font-size:12px}')
page.append('.ptag{font-size:8px;padding:1px 4px;border-radius:3px;background:#00d4aa18;color:#00d4aa;border:1px solid #00d4aa33;margin-left:3px}')
page.append('.imptag{font-size:8px;padding:1px 4px;border-radius:3px;background:#f59e0b18;color:#f59e0b;border:1px solid #f59e0b33;margin-left:3px}')
page.append('.nm{font-size:10px;color:#64748b;max-width:130px;overflow:hidden;text-overflow:ellipsis}')
page.append('.ct{font-size:9px;color:#2a4060}')
page.append('.bw{display:flex;align-items:center;gap:4px}')
page.append('.br{width:38px;height:4px;border-radius:2px;background:#1e2d45;overflow:hidden}')
page.append('.bf{height:100%;border-radius:2px}')
page.append('.chip{display:inline-block;font-size:9px;font-weight:700;padding:1px 5px;border-radius:20px;margin-right:2px;font-family:monospace;background:#00d4aa18;color:#00d4aa;border:1px solid #00d4aa33}')
page.append('.empty{text-align:center;padding:40px;color:#64748b}')
page.append('.disc{margin-top:10px;padding:8px 12px;border:1px solid #f59e0b33;border-radius:8px;background:#f59e0b06;font-size:10px;color:#64748b}')
# Freshness banner
page.append('.stale-banner{background:#f59e0b18;border:1px solid #f59e0b44;border-radius:8px;padding:8px 14px;font-size:11px;color:#f59e0b;margin-bottom:10px;display:none;align-items:center;gap:8px}')
# Tooltip
page.append('.th-wrap{display:inline-flex;align-items:center;gap:3px;position:relative}')
page.append('.tip{display:none;position:absolute;top:calc(100% + 6px);left:50%;transform:translateX(-50%);background:#1e2d45;border:1px solid #2a3f5f;border-radius:8px;padding:8px 10px;font-size:10px;color:#e2e8f0;white-space:normal;width:200px;z-index:20;line-height:1.5;font-family:system-ui;text-transform:none;letter-spacing:0;font-weight:400;text-align:left;pointer-events:none}')
page.append('.th-wrap:hover .tip{display:block}')
page.append('.tip-ico{font-size:8px;color:#2a4060;cursor:help}')
# Mobile cards
page.append('.card-list{display:none;flex-direction:column;gap:8px}')
page.append('.card{background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:12px 14px}')
page.append('.card-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}')
page.append('.card-tk{font-family:monospace;font-weight:900;color:#00d4aa;font-size:14px}')
page.append('.card-score{font-family:monospace;font-weight:900;font-size:20px}')
page.append('.card-nom{font-size:11px;color:#64748b;margin-bottom:6px}')
page.append('.card-tags{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:8px}')
page.append('.card-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}')
page.append('.card-kv{background:#0d1520;border-radius:6px;padding:5px 7px}')
page.append('.card-k{font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}')
page.append('.card-v{font-family:monospace;font-size:11px;font-weight:700;margin-top:2px}')
page.append('.card-sigs{font-size:9px;color:#64748b;margin-top:6px}')
page.append('.cmp-btn{font-size:9px;padding:2px 6px;border-radius:4px;border:1px solid #1e2d45;background:#0d1520;color:#64748b;cursor:pointer;margin-left:4px;font-family:monospace}')
page.append('.cmp-btn.on{border-color:#f59e0b;color:#f59e0b;background:#f59e0b10}')
# Comparator modal
page.append('.cmp-grid{display:grid;gap:12px}')
page.append('.cmp-row{display:grid;gap:4px;align-items:center;padding:6px 0;border-bottom:1px solid #0d1520}')
page.append('.cmp-row:last-child{border:none}')
page.append('.cmp-lbl{font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;min-width:80px}')
page.append('.cmp-vals{display:flex;gap:6px;flex-wrap:wrap}')
page.append('.cmp-val{font-family:monospace;font-size:11px;font-weight:700;background:#0d1520;border-radius:5px;padding:3px 7px;min-width:60px;text-align:center}')
page.append('.cmp-header{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}')
page.append('.cmp-hcol{flex:1;background:#0d1520;border-radius:8px;padding:8px 10px;text-align:center;min-width:80px}')
page.append('.cmp-htk{font-family:monospace;font-weight:900;color:#00d4aa;font-size:13px}')
page.append('.cmp-hnom{font-size:9px;color:#64748b;margin-top:2px}')
# Sparkline
page.append('.spark{display:inline-block;vertical-align:middle}')
# Divergence badge
page.append('.div-badge{font-size:8px;font-weight:700;padding:2px 5px;border-radius:4px;font-family:monospace;white-space:nowrap}')
page.append('.div-fond{background:#10b98118;color:#10b981;border:1px solid #10b98133}')
page.append('.div-tech{background:#f59e0b18;color:#f59e0b;border:1px solid #f59e0b33}')
# Column picker modal
page.append('.col-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:12px}')
page.append('.col-item{display:flex;align-items:center;gap:5px;background:#0d1520;border-radius:6px;padding:6px 8px;cursor:pointer;border:1px solid transparent}')
page.append('.col-item.pinned{border-color:#00d4aa44}')
page.append('.col-item input{accent-color:#00d4aa;cursor:pointer}')
page.append('.col-item label{font-size:10px;color:#94a3b8;cursor:pointer;font-family:monospace}')
# Sub-score filter bar
page.append('.ssf{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;align-items:center}')
page.append('.ssf-lbl{font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.6px}')
page.append('.ssf-item{display:flex;align-items:center;gap:4px;background:#0d1520;border:1px solid #1e2d45;border-radius:6px;padding:4px 8px}')
page.append('.ssf-item label{font-size:10px;color:#64748b;font-family:monospace}')
page.append('.ssf-item input{width:42px;background:#0a0e1a;border:none;color:#00d4aa;font-size:11px;font-family:monospace;text-align:center;outline:none}')
# Theme toggle
page.append('.theme-btn{background:#1e2d45;color:#f59e0b;border:1px solid #f59e0b33;font-size:13px;padding:6px 10px;border-radius:8px;cursor:pointer;border-width:1px}')
# Sort multi indicator
page.append('.sort-badge{font-size:8px;background:#00d4aa;color:#000;border-radius:3px;padding:0 3px;margin-left:2px;font-weight:900}')
# Light theme overrides
page.append('body.light{background:#f0f4f8;color:#1e293b}')
page.append('body.light .hdr,body.light .modal,body.light .tw,body.light .stat,body.light .filters,body.light .mdesc,body.light .mrow .mbtn,body.light .guide-page,body.light .guide-header,body.light .guide-card,body.light .login-box{background:#fff;border-color:#cbd5e1}')
page.append('body.light .cfg-item,body.light .col-item,body.light .ssf-item,body.light .card{background:#f8fafc;border-color:#e2e8f0}')
page.append('body.light tbody td{border-bottom-color:#e2e8f0}')
page.append('body.light tbody tr:nth-child(even) td{background:#f8fafc}')
page.append('body.light tbody tr:hover td{background:#eff6ff!important}')
page.append('body.light thead th{background:#f1f5f9;color:#64748b}')
page.append('body.light .tk{color:#0284c7}')
page.append('body.light .ct{color:#94a3b8}')
page.append('body.light select,body.light input[type=text],body.light input[type=number]{background:#fff;border-color:#cbd5e1;color:#1e293b}')
page.append('.login-screen{display:flex;position:fixed;inset:0;background:#0a0e1a;z-index:100;align-items:center;justify-content:center}')
page.append('.login-box{background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:36px 32px;width:90%;max-width:360px;text-align:center}')
page.append('.login-logo{font-family:monospace;font-size:22px;font-weight:900;color:#00d4aa;margin-bottom:6px}')
page.append('.login-sub{font-size:11px;color:#64748b;margin-bottom:28px}')
page.append('.login-input{width:100%;background:#0d1520;border:1px solid #1e2d45;color:#e2e8f0;font-size:14px;padding:12px 14px;border-radius:9px;outline:none;font-family:monospace;text-align:center;letter-spacing:3px;margin-bottom:10px}')
page.append('.login-input:focus{border-color:#00d4aa44}')
page.append('.login-btn{width:100%;padding:12px;border-radius:9px;border:none;cursor:pointer;font-family:monospace;font-weight:700;font-size:13px;background:linear-gradient(135deg,#00d4aa,#00a87f);color:#000;margin-bottom:8px}')
page.append('.login-err{font-size:11px;color:#ef4444;min-height:16px;margin-bottom:6px}')
# Toggle on/off styles
page.append('.tog-row{display:flex;align-items:center;justify-content:space-between;padding:6px 0;border-bottom:1px solid #0d1520}')
page.append('.tog-row:last-child{border-bottom:none}')
page.append('.tog-label{font-size:11px;color:#94a3b8}')
page.append('.tog{position:relative;width:32px;height:18px;flex-shrink:0}')
page.append('.tog input{opacity:0;width:0;height:0}')
page.append('.tog-sl{position:absolute;inset:0;background:#1e2d45;border-radius:18px;cursor:pointer;transition:.2s}')
page.append('.tog-sl:before{content:"";position:absolute;width:12px;height:12px;left:3px;bottom:3px;background:#64748b;border-radius:50%;transition:.2s}')
page.append('.tog input:checked+.tog-sl{background:#00d4aa22;border:1px solid #00d4aa55}')
page.append('.tog input:checked+.tog-sl:before{transform:translateX(14px);background:#00d4aa}')
# Styles onglet Guide
page.append('.guide-page{display:none;position:fixed;inset:0;background:#0a0e1a;z-index:40;overflow-y:auto}')
page.append('.guide-page.open{display:block}')
page.append('.guide-header{background:#111827;border-bottom:1px solid #1e2d45;padding:14px 20px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:5}')
page.append('.guide-title{font-family:monospace;font-size:16px;font-weight:900;color:#8b5cf6}')
page.append('.guide-wrap{max-width:1100px;margin:0 auto;padding:24px 16px}')
page.append('.guide-cat{font-family:monospace;font-size:10px;font-weight:900;letter-spacing:1.5px;text-transform:uppercase;color:#64748b;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid #1e2d45}')
page.append('.guide-cat:first-child{margin-top:0}')
page.append('.guide-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}')
page.append('.guide-card{background:#111827;border:1px solid #1e2d45;border-radius:12px;padding:18px;transition:border-color .2s}')
page.append('.guide-card:hover{border-color:#8b5cf644}')
page.append('.gc-title{font-family:monospace;font-size:13px;font-weight:900;color:#e2e8f0;margin-bottom:4px;display:flex;align-items:center;gap:8px}')
page.append('.gc-badge{font-size:8px;padding:2px 7px;border-radius:20px;font-weight:700}')
page.append('.badge-tech{background:#3b82f618;color:#3b82f6;border:1px solid #3b82f633}')
page.append('.badge-value{background:#10b98118;color:#10b981;border:1px solid #10b98133}')
page.append('.badge-growth{background:#f59e0b18;color:#f59e0b;border:1px solid #f59e0b33}')
page.append('.badge-div{background:#8b5cf618;color:#8b5cf6;border:1px solid #8b5cf633}')
page.append('.badge-qual{background:#ec489918;color:#ec4899;border:1px solid #ec489933}')
page.append('.gc-desc{font-size:11px;color:#64748b;line-height:1.6;margin-bottom:12px}')
page.append('.gc-zones{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}')
page.append('.zone{font-size:10px;padding:3px 9px;border-radius:20px;font-family:monospace;font-weight:700}')
page.append('.zone-g{background:#10b98118;color:#10b981;border:1px solid #10b98133}')
page.append('.zone-y{background:#f59e0b18;color:#f59e0b;border:1px solid #f59e0b33}')
page.append('.zone-r{background:#ef444418;color:#ef4444;border:1px solid #ef444433}')
page.append('.gc-svg{width:100%;overflow:visible}')
page.append('@media(max-width:640px){.ts{display:none}.card-list{display:flex}.hbtn-scan span{display:none}.filters{gap:6px}.fg label{font-size:8px}.stat{min-width:60px}.hbtns{gap:4px}.hbtn{padding:6px 9px;font-size:10px}}')
page.append('</style></head><body>')

# ══ HEADER ══
page.append('<div class="hdr">')
page.append('<div><div class="htitle">&#128225; Stock Screener</div>')
page.append('<div class="hsub">' + NSCAN + ' actions scannees - Yahoo Finance - ' + DATE + '</div></div>')
page.append('<div class="hbtns">')
page.append('<span style="font-family:monospace;font-size:10px;color:#64748b" id="nxt"></span>')
page.append('<button class="theme-btn" onclick="toggleTheme()" id="theme-btn" title="Mode clair/sombre">&#9788;</button>')
page.append('<button class="hbtn hbtn-guide" onclick="openGuide()">&#128218; Guide</button>')
page.append('<button class="hbtn" onclick="openCols()" style="background:#1e2d45;color:#00d4aa;border:1px solid #00d4aa44">&#9638; Colonnes</button>')
page.append('<button class="hbtn" onclick="exportCSV()" style="background:#1e2d45;color:#10b981;border:1px solid #10b98144">&#8659; CSV</button>')
page.append('<button class="hbtn" id="cmp-open-btn" onclick="openComparator()" style="background:#1e2d45;color:#f59e0b;border:1px solid #f59e0b44;display:none">&#9878; Comparer (<span id="cmp-count">0</span>)</button>')
page.append('<button class="hbtn hbtn-cfg" onclick="openCfg()">&#9881; Criteres</button>')
page.append('<button class="hbtn hbtn-imp" onclick="openImp()">&#128229; Tickers</button>')
page.append('<button class="hbtn hbtn-scan" onclick="triggerScan()">&#9889; Nouveau scan</button>')
page.append('</div></div>')

# ══ MODAL CRITERES ══
page.append('<div class="modal-bg" id="modal-cfg"><div class="modal" style="max-width:620px">')
page.append('<button class="modal-close" onclick="closeCfg()">&#10005;</button>')
page.append('<h3>&#9881; Criteres de scoring</h3>')
page.append('<div class="cfg-note">Modifications instantanees. Nouveau scan pour recharger Yahoo Finance.</div>')
# Poids
page.append('<div class="cfg-section">Poids du score (total = 100)</div><div class="cfg-grid">')
page.append('<div class="cfg-item"><label>Poids Value (%)</label><input type="number" id="c-pv" step="5"></div>')
page.append('<div class="cfg-item"><label>Poids Growth (%)</label><input type="number" id="c-pg" step="5"></div>')
page.append('<div class="cfg-item"><label>Poids Dividende (%)</label><input type="number" id="c-pd" step="5"></div>')
page.append('<div class="cfg-item"><label>Poids Technique (%)</label><input type="number" id="c-pt" step="5"></div>')
page.append('</div>')
# Valorisation seuils + toggles
page.append('<div class="cfg-section">Valorisation — Seuils & Activation</div>')
page.append('<div class="cfg-grid">')
page.append('<div class="cfg-item"><label>PER max</label><input type="number" id="c-per" step="1"></div>')
page.append('<div class="cfg-item"><label>P/Book max</label><input type="number" id="c-pbv" step="0.5"></div>')
page.append('<div class="cfg-item"><label>EV/EBITDA max</label><input type="number" id="c-ev" step="1"></div>')
page.append('<div class="cfg-item"><label>P/S max</label><input type="number" id="c-ps" step="0.5"></div>')
page.append('<div class="cfg-item"><label>P/FCF max</label><input type="number" id="c-pfcf" step="2"></div>')
page.append('<div class="cfg-item"><label>Graham min (%)</label><input type="number" id="c-graham" step="5"></div>')
page.append('</div>')
page.append('<div style="background:#0d1520;border-radius:8px;padding:10px 12px;margin-bottom:10px">')
page.append('<div class="tog-row"><span class="tog-label">PER actif</span><label class="tog"><input type="checkbox" id="t-per" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">P/Book actif</span><label class="tog"><input type="checkbox" id="t-pbv" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">EV/EBITDA actif</span><label class="tog"><input type="checkbox" id="t-ev" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">P/S actif</span><label class="tog"><input type="checkbox" id="t-ps" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">P/FCF actif</span><label class="tog"><input type="checkbox" id="t-pfcf" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">Graham actif</span><label class="tog"><input type="checkbox" id="t-graham" checked><span class="tog-sl"></span></label></div>')
page.append('</div>')
# Croissance & Qualité seuils + toggles
page.append('<div class="cfg-section">Croissance & Qualite — Seuils & Activation</div>')
page.append('<div class="cfg-grid">')
page.append('<div class="cfg-item"><label>ROE min (%)</label><input type="number" id="c-roe" step="1"></div>')
page.append('<div class="cfg-item"><label>Current Ratio min</label><input type="number" id="c-cr" step="0.1"></div>')
page.append('<div class="cfg-item"><label>D/E max (%)</label><input type="number" id="c-de" step="10"></div>')
page.append('<div class="cfg-item"><label>FCF Yield min (%)</label><input type="number" id="c-fcfy" step="1"></div>')
page.append('</div>')
page.append('<div style="background:#0d1520;border-radius:8px;padding:10px 12px;margin-bottom:10px">')
page.append('<div class="tog-row"><span class="tog-label">ROE actif</span><label class="tog"><input type="checkbox" id="t-roe" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">BPA Croissance actif</span><label class="tog"><input type="checkbox" id="t-bpa" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">PEG actif</span><label class="tog"><input type="checkbox" id="t-peg" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">Current Ratio actif</span><label class="tog"><input type="checkbox" id="t-cr" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">Debt/Equity actif</span><label class="tog"><input type="checkbox" id="t-de" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">FCF Yield actif</span><label class="tog"><input type="checkbox" id="t-fcfy" checked><span class="tog-sl"></span></label></div>')
page.append('</div>')
# Technique seuils + toggles
page.append('<div class="cfg-section">Technique — Seuils & Activation</div>')
page.append('<div class="cfg-grid">')
page.append('<div class="cfg-item"><label>RSI survente max</label><input type="number" id="c-rsi" step="1"></div>')
page.append('<div class="cfg-item"><label>Stoch. survente max</label><input type="number" id="c-stoch" step="1"></div>')
page.append('<div class="cfg-item"><label>CCI survente (negatif)</label><input type="number" id="c-cci" step="10"></div>')
page.append('<div class="cfg-item"><label>ADX min</label><input type="number" id="c-adx" step="1"></div>')
page.append('<div class="cfg-item"><label>Williams %R survente</label><input type="number" id="c-willr" step="5"></div>')
page.append('<div class="cfg-item"><label>RoC min (%)</label><input type="number" id="c-roc" step="1"></div>')
page.append('</div>')
page.append('<div style="background:#0d1520;border-radius:8px;padding:10px 12px;margin-bottom:10px">')
page.append('<div class="tog-row"><span class="tog-label">RSI actif</span><label class="tog"><input type="checkbox" id="t-rsi" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">MACD actif</span><label class="tog"><input type="checkbox" id="t-macd" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">CCI actif</span><label class="tog"><input type="checkbox" id="t-cci" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">Stochastique actif</span><label class="tog"><input type="checkbox" id="t-stoch" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">Moyennes Mobiles actif</span><label class="tog"><input type="checkbox" id="t-mm" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">ADX actif</span><label class="tog"><input type="checkbox" id="t-adx" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">Williams %R actif</span><label class="tog"><input type="checkbox" id="t-willr" checked><span class="tog-sl"></span></label></div>')
page.append('<div class="tog-row"><span class="tog-label">RoC actif</span><label class="tog"><input type="checkbox" id="t-roc" checked><span class="tog-sl"></span></label></div>')
page.append('</div>')
# Filtrage
page.append('<div class="cfg-section">Filtrage</div><div class="cfg-grid">')
page.append('<div class="cfg-item"><label>Score minimum /100</label><input type="number" id="c-smin" step="5"></div>')
page.append('</div>')
page.append('<button class="modal-btn modal-btn-ok" onclick="applyCfg()">&#10003; Appliquer</button>')
page.append('<button class="modal-btn modal-btn-cancel" onclick="closeCfg()">Annuler</button>')
page.append('</div></div>')

# ══ MODAL IMPORT ══
page.append('<div class="modal-bg" id="modal-imp"><div class="modal">')
page.append('<button class="modal-close" onclick="closeImp()">&#10005;</button>')
page.append('<h3>&#128229; Importer des tickers</h3>')
page.append('<div class="imp-note">Un ticker par ligne ou separes par virgule.<br>Ex: AAPL, MC.PA, AIR.PA<br>Filtre les donnees du dernier scan. Au prochain scan automatique la liste reprend normalement.</div>')
page.append('<textarea class="imp-area" id="imp-text" placeholder="AAPL&#10;MSFT&#10;MC.PA"></textarea>')
page.append('<button class="modal-btn modal-btn-ok" onclick="importTickers()">&#10003; Appliquer</button>')
page.append('<button class="modal-btn modal-btn-cancel" onclick="closeImp()" style="margin-top:8px">Annuler</button>')
page.append('</div></div>')

# ══ MODAL SCAN ══
page.append('<div class="modal-bg" id="modal-scan"><div class="modal" style="text-align:center">')
page.append('<h3>&#9889; Scan lance sur GitHub Actions</h3>')
page.append('<p style="color:#64748b;font-size:12px;margin:12px 0 16px;line-height:1.7">' + NSCAN + ' actions a analyser.<br>Duree estimee: 30-60 min.<br>La page se rechargera automatiquement.</p>')
page.append('<div style="font-family:monospace;font-size:13px;color:#00d4aa;background:#0d1520;padding:12px;border-radius:8px;margin-bottom:14px" id="scan-countdown"></div>')
page.append('<button class="modal-btn modal-btn-cancel" onclick="document.getElementById(\'modal-scan\').className=\'modal-bg\'">Fermer</button>')
page.append('</div></div>')

# ══ PAGE GUIDE ══
page.append('<div class="guide-page" id="guide-page">')
page.append('<div class="guide-header">')
page.append('<div class="guide-title">&#128218; Guide des indicateurs</div>')
page.append('<button class="hbtn hbtn-cfg" onclick="closeGuide()" style="color:#64748b;border-color:#64748b44">&#10005; Fermer</button>')
page.append('</div>')
page.append('<div class="guide-wrap">')

# ─── CATEGORIE: INDICATEURS TECHNIQUES ───
page.append('<div class="guide-cat">&#128200; Indicateurs Techniques — Momentum & Tendance</div>')
page.append('<div class="guide-grid">')

# RSI
page.append('''<div class="guide-card">
<div class="gc-title">RSI <span class="gc-badge badge-tech">Technique</span></div>
<div class="gc-desc">Relative Strength Index (14 jours). Mesure la vitesse et l\'amplitude des variations de prix. Oscille entre 0 et 100.</div>
<div class="gc-zones">
  <span class="zone zone-g">Survente &le;30 (+28pts)</span>
  <span class="zone zone-y">Zone &le;45 (+18pts)</span>
  <span class="zone zone-r">Surachat &ge;70</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="280" height="80" fill="none"/>
  <rect x="0" y="60" width="280" height="20" fill="#10b98110" rx="2"/>
  <rect x="0" y="0" width="280" height="14" fill="#ef444410" rx="2"/>
  <text x="2" y="10" fill="#ef4444" font-size="7" font-family="monospace">Surachat 70+</text>
  <text x="2" y="74" fill="#10b981" font-size="7" font-family="monospace">Survente -30</text>
  <line x1="0" y1="14" x2="280" y2="14" stroke="#ef444430" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="0" y1="60" x2="280" y2="60" stroke="#10b98130" stroke-width="1" stroke-dasharray="3,3"/>
  <polyline points="0,40 30,38 55,50 80,65 110,68 140,55 165,42 190,28 210,10 235,18 260,35 280,40"
    fill="none" stroke="#3b82f6" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="110" cy="68" r="4" fill="#10b981"/>
  <text x="95" y="58" fill="#10b981" font-size="8" font-family="monospace">Signal !</text>
  <circle cx="210" cy="10" r="4" fill="#ef4444"/>
</svg>
</div>''')

# MACD
page.append('''<div class="guide-card">
<div class="gc-title">MACD <span class="gc-badge badge-tech">Technique</span></div>
<div class="gc-desc">Moving Average Convergence Divergence (12/26/9). Croisement ligne MACD au-dessus du signal = momentum haussier.</div>
<div class="gc-zones">
  <span class="zone zone-g">Histo &gt;0 + croisement (+20pts)</span>
  <span class="zone zone-y">Histo &gt;0 (+12pts)</span>
  <span class="zone zone-r">Histo negatif</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <line x1="0" y1="40" x2="280" y2="40" stroke="#1e2d45" stroke-width="1"/>
  <rect x="10" y="44" width="12" height="10" fill="#ef444460" rx="1"/>
  <rect x="26" y="46" width="12" height="8" fill="#ef444460" rx="1"/>
  <rect x="42" y="43" width="12" height="11" fill="#ef444460" rx="1"/>
  <rect x="58" y="38" width="12" height="4" fill="#10b98160" rx="1"/>
  <rect x="74" y="32" width="12" height="8" fill="#10b98160" rx="1"/>
  <rect x="90" y="28" width="12" height="12" fill="#10b98160" rx="1"/>
  <rect x="106" y="25" width="12" height="15" fill="#10b98160" rx="1"/>
  <rect x="122" y="30" width="12" height="10" fill="#10b98160" rx="1"/>
  <rect x="138" y="36" width="12" height="4" fill="#10b98160" rx="1"/>
  <rect x="154" y="41" width="12" height="6" fill="#ef444460" rx="1"/>
  <rect x="170" y="43" width="12" height="9" fill="#ef444460" rx="1"/>
  <polyline points="16,52 32,54 48,50 64,40 80,34 96,32 112,29 128,33 144,38 160,45 176,50"
    fill="none" stroke="#00d4aa" stroke-width="1.5"/>
  <polyline points="16,54 32,55 48,53 64,44 80,37 96,34 112,31 128,35 144,40 160,46 176,51"
    fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="3,2"/>
  <text x="200" y="36" fill="#00d4aa" font-size="8" font-family="monospace">MACD</text>
  <text x="200" y="47" fill="#f59e0b" font-size="8" font-family="monospace">Signal</text>
  <circle cx="64" cy="40" r="4" fill="#10b981"/>
  <text x="54" y="30" fill="#10b981" font-size="8" font-family="monospace">Cross!</text>
</svg>
</div>''')

# Stochastique
page.append('''<div class="guide-card">
<div class="gc-title">Stochastique %K <span class="gc-badge badge-tech">Technique</span></div>
<div class="gc-desc">Oscillateur (14,3). Compare le cours de cloture a la fourchette High/Low. Tres efficace pour les points de retournement.</div>
<div class="gc-zones">
  <span class="zone zone-g">Survente &le;20 (+14pts)</span>
  <span class="zone zone-y">Zone &le;30 (+9pts)</span>
  <span class="zone zone-r">Surachat &ge;80</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="280" height="12" fill="#ef444410" rx="2"/>
  <rect x="0" y="66" width="280" height="14" fill="#10b98110" rx="2"/>
  <text x="2" y="9" fill="#ef4444" font-size="7" font-family="monospace">Surachat 80+</text>
  <text x="2" y="76" fill="#10b981" font-size="7" font-family="monospace">Survente -20</text>
  <line x1="0" y1="12" x2="280" y2="12" stroke="#ef444430" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="0" y1="66" x2="280" y2="66" stroke="#10b98130" stroke-width="1" stroke-dasharray="3,3"/>
  <polyline points="0,30 25,20 50,8 80,5 110,15 140,42 165,65 190,72 215,68 240,50 265,35 280,28"
    fill="none" stroke="#8b5cf6" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="80" cy="5" r="4" fill="#ef4444"/>
  <circle cx="190" cy="72" r="4" fill="#10b981"/>
  <text x="178" y="63" fill="#10b981" font-size="8" font-family="monospace">Achat</text>
</svg>
</div>''')

# CCI
page.append('''<div class="guide-card">
<div class="gc-title">CCI <span class="gc-badge badge-tech">Technique</span></div>
<div class="gc-desc">Commodity Channel Index (20 jours). Mesure l\'ecart du prix par rapport a sa moyenne statistique. Valeur negative = survente.</div>
<div class="gc-zones">
  <span class="zone zone-g">Survente &lt;-100 (+18pts)</span>
  <span class="zone zone-y">Tendance 50-150 (+12pts)</span>
  <span class="zone zone-r">Surachat &gt;200</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <line x1="0" y1="40" x2="280" y2="40" stroke="#1e2d45" stroke-width="1"/>
  <line x1="0" y1="20" x2="280" y2="20" stroke="#f59e0b20" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="0" y1="60" x2="280" y2="60" stroke="#10b98120" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="2" y="18" fill="#f59e0b" font-size="7" font-family="monospace">+100</text>
  <text x="2" y="58" fill="#10b981" font-size="7" font-family="monospace">-100</text>
  <polyline points="0,35 30,30 55,22 80,18 105,25 130,38 155,52 180,65 205,68 230,55 255,42 280,36"
    fill="none" stroke="#00d4aa" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="205" cy="68" r="4" fill="#10b981"/>
  <text x="192" y="58" fill="#10b981" font-size="8" font-family="monospace">Signal</text>
</svg>
</div>''')

# ADX
page.append('''<div class="guide-card">
<div class="gc-title">ADX <span class="gc-badge badge-tech">Technique</span></div>
<div class="gc-desc">Average Directional Index (14j). Mesure la force de la tendance (sans sa direction). ADX &gt;20 = tendance etablie, &gt;40 = forte tendance.</div>
<div class="gc-zones">
  <span class="zone zone-g">Tendance &ge;20 (+5pts bonus)</span>
  <span class="zone zone-y">Faible &lt;20</span>
  <span class="zone zone-r">Pas de tendance &lt;10</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <line x1="0" y1="50" x2="280" y2="50" stroke="#f59e0b30" stroke-width="1" stroke-dasharray="4,4"/>
  <text x="2" y="47" fill="#f59e0b" font-size="7" font-family="monospace">ADX 20 — seuil tendance</text>
  <polyline points="0,68 30,65 60,60 90,55 120,48 150,40 180,32 210,28 240,26 270,25 280,25"
    fill="none" stroke="#00d4aa" stroke-width="2.5" stroke-linejoin="round"/>
  <circle cx="120" cy="48" r="4" fill="#f59e0b"/>
  <text x="108" y="40" fill="#f59e0b" font-size="8" font-family="monospace">Seuil!</text>
  <text x="200" y="20" fill="#10b981" font-size="8" font-family="monospace">Tendance forte</text>
  <text x="10" y="78" fill="#64748b" font-size="8" font-family="monospace">Consolidation</text>
</svg>
</div>''')

# Bollinger
page.append('''<div class="guide-card">
<div class="gc-title">Bandes de Bollinger <span class="gc-badge badge-tech">Technique</span></div>
<div class="gc-desc">MM20 +/- 2 ecarts-types. Le score BB (0-100%) positionne le cours entre la bande basse (0%) et haute (100%). &lt;20% = survente potentielle.</div>
<div class="gc-zones">
  <span class="zone zone-g">Proche bande basse &lt;20%</span>
  <span class="zone zone-y">Zone centrale</span>
  <span class="zone zone-r">Proche bande haute &gt;80%</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <polyline points="0,55 40,52 80,42 120,30 160,22 200,30 240,45 280,55"
    fill="none" stroke="#1e2d45" stroke-width="1" stroke-dasharray="3,2"/>
  <polyline points="0,38 40,35 80,28 120,20 160,14 200,20 240,33 280,40"
    fill="none" stroke="#3b82f630" stroke-width="1"/>
  <polyline points="0,65 40,62 80,56 120,46 160,38 200,44 240,58 280,65"
    fill="none" stroke="#3b82f630" stroke-width="1"/>
  <polyline points="0,60 40,58 80,53 120,44 160,40 200,42 240,54 280,62"
    fill="none" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="40" cy="58" r="3.5" fill="#10b981"/>
  <text x="28" y="50" fill="#10b981" font-size="8" font-family="monospace">BB bas</text>
  <text x="5" y="34" fill="#3b82f6" font-size="7" font-family="monospace">Bande haute</text>
  <text x="5" y="70" fill="#3b82f6" font-size="7" font-family="monospace">Bande basse</text>
</svg>
</div>''')

# Moyennes mobiles
page.append('''<div class="guide-card">
<div class="gc-title">MM20 / MM50 / MM200 <span class="gc-badge badge-tech">Technique</span></div>
<div class="gc-desc">Moyennes mobiles simples sur 20, 50 et 200 jours. Cours au-dessus des 3 moyennes = tendance haussiere confirmee (+15pts).</div>
<div class="gc-zones">
  <span class="zone zone-g">Au-dessus des 3 MM (+15pts)</span>
  <span class="zone zone-y">Au-dessus de 2 MM (+8pts)</span>
  <span class="zone zone-r">En dessous</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <polyline points="0,65 30,62 60,55 90,48 120,40 150,33 180,28 210,25 240,23 270,21"
    fill="none" stroke="#64748b" stroke-width="1.5" stroke-dasharray="5,3"/>
  <polyline points="0,60 30,56 60,50 90,43 120,36 150,30 180,26 210,23 240,21 270,19"
    fill="none" stroke="#f59e0b" stroke-width="1.5"/>
  <polyline points="0,55 30,50 60,44 90,37 120,31 150,26 180,22 210,19 240,17 270,15"
    fill="none" stroke="#3b82f6" stroke-width="1.5"/>
  <polyline points="0,52 30,47 60,40 90,33 120,27 150,22 180,18 210,15 240,12 270,10"
    fill="none" stroke="#10b981" stroke-width="2"/>
  <text x="215" y="28" fill="#64748b" font-size="7" font-family="monospace">MM200</text>
  <text x="215" y="18" fill="#f59e0b" font-size="7" font-family="monospace">MM50</text>
  <text x="215" y="10" fill="#3b82f6" font-size="7" font-family="monospace">MM20</text>
  <text x="215" y="7" fill="#10b981" font-size="7" font-family="monospace">Prix</text>
</svg>
</div>''')

# Williams %R
page.append('''<div class="guide-card">
<div class="gc-title">Williams %R <span class="gc-badge badge-tech">Nouveau</span></div>
<div class="gc-desc">Oscillateur de Larry Williams (14j). Similaire au Stochastique, oscille entre -100 et 0. Tres negatif = cours proche du plus bas = opportunite.</div>
<div class="gc-zones">
  <span class="zone zone-g">Survente &le;-80 (+8pts)</span>
  <span class="zone zone-y">Zone &le;-60 (+4pts)</span>
  <span class="zone zone-r">Surachat &ge;-20</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="280" height="12" fill="#ef444410" rx="2"/>
  <rect x="0" y="66" width="280" height="14" fill="#10b98110" rx="2"/>
  <text x="2" y="9" fill="#ef4444" font-size="7" font-family="monospace">Surachat -20</text>
  <text x="2" y="76" fill="#10b981" font-size="7" font-family="monospace">Survente -80</text>
  <line x1="0" y1="12" x2="280" y2="12" stroke="#ef444430" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="0" y1="66" x2="280" y2="66" stroke="#10b98130" stroke-width="1" stroke-dasharray="3,3"/>
  <polyline points="0,28 25,18 50,8 80,6 110,18 140,38 170,60 195,72 220,68 248,50 270,33 280,26"
    fill="none" stroke="#ec4899" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="80" cy="6" r="4" fill="#ef4444"/>
  <circle cx="195" cy="72" r="4" fill="#10b981"/>
  <text x="182" y="62" fill="#10b981" font-size="8" font-family="monospace">Signal</text>
</svg>
</div>''')

# RoC
page.append('''<div class="guide-card">
<div class="gc-title">Rate of Change (RoC) <span class="gc-badge badge-tech">Nouveau</span></div>
<div class="gc-desc">Variation en % du cours sur 12 periodes. Mesure la vitesse du mouvement. RoC positif et croissant = acceleration haussiere.</div>
<div class="gc-zones">
  <span class="zone zone-g">RoC &gt;+5% (+5pts bonus)</span>
  <span class="zone zone-y">0 a +5%</span>
  <span class="zone zone-r">RoC negatif</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 80" xmlns="http://www.w3.org/2000/svg">
  <line x1="0" y1="40" x2="280" y2="40" stroke="#1e2d45" stroke-width="1"/>
  <line x1="0" y1="30" x2="280" y2="30" stroke="#10b98120" stroke-width="1" stroke-dasharray="3,3"/>
  <text x="2" y="28" fill="#10b981" font-size="7" font-family="monospace">+5%</text>
  <polyline points="0,45 30,48 60,52 90,55 110,50 130,42 155,33 175,27 200,24 225,22 255,25 280,30"
    fill="none" stroke="#f59e0b" stroke-width="2" stroke-linejoin="round"/>
  <rect x="0" y="40" width="280" height="40" fill="#ef444408"/>
  <rect x="0" y="0" width="280" height="40" fill="#10b98108"/>
  <circle cx="155" cy="33" r="3.5" fill="#10b981"/>
  <text x="143" y="25" fill="#10b981" font-size="8" font-family="monospace">Accel.</text>
</svg>
</div>''')

page.append('</div>') # fin guide-grid technique

# ─── CATEGORIE: VALORISATION ───
page.append('<div class="guide-cat">&#128202; Valorisation — Prix vs Valeur Intrinseque</div>')
page.append('<div class="guide-grid">')

# PER
page.append('''<div class="guide-card">
<div class="gc-title">PER <span class="gc-badge badge-value">Valorisation</span></div>
<div class="gc-desc">Price-to-Earnings Ratio. Combien de fois les investisseurs paient les benefices annuels. Varie fortement selon le secteur.</div>
<div class="gc-zones">
  <span class="zone zone-g">&le;10 (+40pts)</span>
  <span class="zone zone-y">&le;15 (+28pts) / &le;30 (+15pts)</span>
  <span class="zone zone-r">&gt;30 (0pt)</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="15" width="55" height="40" fill="#10b98130" rx="4" stroke="#10b98150" stroke-width="1"/>
  <rect x="70" y="20" width="55" height="35" fill="#f59e0b20" rx="4" stroke="#f59e0b40" stroke-width="1"/>
  <rect x="135" y="28" width="55" height="27" fill="#f59e0b10" rx="4" stroke="#f59e0b30" stroke-width="1"/>
  <rect x="200" y="38" width="55" height="17" fill="#ef444420" rx="4" stroke="#ef444440" stroke-width="1"/>
  <text x="22" y="60" fill="#10b981" font-size="9" font-family="monospace" text-anchor="middle">PER 10</text>
  <text x="97" y="60" fill="#f59e0b" font-size="9" font-family="monospace" text-anchor="middle">PER 15</text>
  <text x="162" y="60" fill="#f59e0b" font-size="9" font-family="monospace" text-anchor="middle">PER 30</text>
  <text x="227" y="60" fill="#ef4444" font-size="9" font-family="monospace" text-anchor="middle">PER 50+</text>
  <text x="22" y="30" fill="#10b981" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">40pts</text>
  <text x="97" y="35" fill="#f59e0b" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">28pts</text>
  <text x="162" y="42" fill="#f59e0b" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">15pts</text>
  <text x="227" y="50" fill="#ef4444" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">0pt</text>
</svg>
</div>''')

# P/Book
page.append('''<div class="guide-card">
<div class="gc-title">P/Book (PBV) <span class="gc-badge badge-value">Valorisation</span></div>
<div class="gc-desc">Price-to-Book Value. Rapport entre capitalisation boursiere et valeur comptable des actifs nets. PBV &lt;1 = decote sur actifs.</div>
<div class="gc-zones">
  <span class="zone zone-g">&le;1 (+30pts) — Decote actifs</span>
  <span class="zone zone-y">&le;2 (+20pts) / &le;8 (+10pts)</span>
  <span class="zone zone-r">&gt;8 (0pt)</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="10" width="240" height="40" fill="#1e2d45" rx="4"/>
  <rect x="20" y="10" width="120" height="40" fill="#10b98120" rx="4"/>
  <line x1="140" y1="10" x2="140" y2="50" stroke="#10b981" stroke-width="1.5" stroke-dasharray="3,2"/>
  <text x="80" y="34" fill="#10b981" font-size="9" font-family="monospace" font-weight="bold" text-anchor="middle">Valeur comptable</text>
  <text x="200" y="34" fill="#64748b" font-size="9" font-family="monospace" text-anchor="middle">Prime de marche</text>
  <text x="20" y="62" fill="#10b981" font-size="8" font-family="monospace">PBV=1</text>
  <text x="105" y="62" fill="#f59e0b" font-size="8" font-family="monospace">PBV=2</text>
  <text x="230" y="62" fill="#ef4444" font-size="8" font-family="monospace">PBV=8+</text>
</svg>
</div>''')

# EV/EBITDA
page.append('''<div class="guide-card">
<div class="gc-title">EV/EBITDA <span class="gc-badge badge-value">Valorisation</span></div>
<div class="gc-desc">Enterprise Value / Benefices avant interets, impots et amortissements. Indifferent a la structure financiere, ideal pour comparer inter-secteurs.</div>
<div class="gc-zones">
  <span class="zone zone-g">&le;8 (+30pts) — Tres bon marche</span>
  <span class="zone zone-y">&le;12 (+20pts) / &le;15 (+10pts)</span>
  <span class="zone zone-r">&gt;15 (0pt)</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="140" y="12" fill="#64748b" font-size="8" font-family="monospace" text-anchor="middle">Valeur Entreprise = Capitalisation + Dette - Cash</text>
  <rect x="5" y="20" width="60" height="36" fill="#10b98130" rx="4"/>
  <rect x="75" y="25" width="60" height="30" fill="#f59e0b20" rx="4"/>
  <rect x="145" y="30" width="60" height="25" fill="#f59e0b10" rx="4"/>
  <rect x="215" y="40" width="55" height="15" fill="#ef444420" rx="4"/>
  <text x="35" y="42" fill="#10b981" font-size="8" font-family="monospace" text-anchor="middle">EV/E &le;8</text>
  <text x="105" y="44" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">EV/E &le;12</text>
  <text x="175" y="46" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">EV/E &le;15</text>
  <text x="242" y="52" fill="#ef4444" font-size="8" font-family="monospace" text-anchor="middle">&gt;15</text>
</svg>
</div>''')

# P/S
page.append('''<div class="guide-card">
<div class="gc-title">P/S (Price-to-Sales) <span class="gc-badge badge-value">Nouveau</span></div>
<div class="gc-desc">Capitalisation / Chiffre d\'affaires. Utile quand le PER n\'est pas disponible (entreprises en croissance, pertes temporaires). Moins sensible aux manipulations comptables.</div>
<div class="gc-zones">
  <span class="zone zone-g">&le;1 (+10pts)</span>
  <span class="zone zone-y">&le;2 (+6pts) / &le;4 (+3pts)</span>
  <span class="zone zone-r">&gt;4 (0pt)</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="140" y="12" fill="#64748b" font-size="8" font-family="monospace" text-anchor="middle">Capitalisation boursiere / Chiffre d\'affaires annuel</text>
  <rect x="5" y="20" width="60" height="36" fill="#10b98130" rx="4"/>
  <rect x="75" y="28" width="60" height="28" fill="#f59e0b20" rx="4"/>
  <rect x="145" y="35" width="60" height="21" fill="#f59e0b10" rx="4"/>
  <rect x="215" y="45" width="55" height="11" fill="#ef444420" rx="4"/>
  <text x="35" y="42" fill="#10b981" font-size="8" font-family="monospace" text-anchor="middle">P/S &le;1</text>
  <text x="105" y="45" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">P/S &le;2</text>
  <text x="175" y="50" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">P/S &le;4</text>
  <text x="242" y="55" fill="#ef4444" font-size="8" font-family="monospace" text-anchor="middle">&gt;4</text>
</svg>
</div>''')

# P/FCF
page.append('''<div class="guide-card">
<div class="gc-title">P/FCF <span class="gc-badge badge-value">Nouveau</span></div>
<div class="gc-desc">Price-to-Free Cash Flow. Capitalisation / flux de tresorerie disponible. Plus fiable que le PER car difficile a manipuler comptablement. FCF = cash vraiment genere.</div>
<div class="gc-zones">
  <span class="zone zone-g">&le;10 (+10pts)</span>
  <span class="zone zone-y">&le;20 (+6pts) / &le;30 (+3pts)</span>
  <span class="zone zone-r">&gt;30 ou FCF negatif</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="5" y="12" fill="#64748b" font-size="8" font-family="monospace">Benefice net vs Free Cash Flow (FCF)</text>
  <rect x="5" y="18" width="80" height="14" fill="#f59e0b30" rx="3"/>
  <rect x="5" y="36" width="58" height="14" fill="#10b98130" rx="3"/>
  <text x="45" y="29" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">Benefice net</text>
  <text x="34" y="47" fill="#10b981" font-size="8" font-family="monospace" text-anchor="middle">FCF reel</text>
  <text x="115" y="25" fill="#64748b" font-size="8" font-family="monospace">Amortissements,</text>
  <text x="115" y="35" fill="#64748b" font-size="8" font-family="monospace">provisions, BFR...</text>
  <text x="5" y="66" fill="#10b981" font-size="8" font-family="monospace">P/FCF &le;10 = excellent</text>
  <text x="150" y="66" fill="#ef4444" font-size="8" font-family="monospace">FCF &lt;0 = attention</text>
</svg>
</div>''')

# Graham
page.append('''<div class="guide-card">
<div class="gc-title">Graham Number <span class="gc-badge badge-value">Nouveau</span></div>
<div class="gc-desc">Valeur intrinseque selon Benjamin Graham: sqrt(22.5 x BPA x VNC/action). Le score affiche l\'ecart en % entre la valeur Graham et le cours actuel. Positif = sous-evalue.</div>
<div class="gc-zones">
  <span class="zone zone-g">Cours &lt; Graham de 20%+ (+15pts)</span>
  <span class="zone zone-y">Cours &lt; Graham (+10pts)</span>
  <span class="zone zone-r">Cours &gt; Graham (surevalu)</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="140" y="11" fill="#64748b" font-size="8" font-family="monospace" text-anchor="middle">Graham = sqrt(22.5 x BPA x Valeur Comptable)</text>
  <line x1="40" y1="20" x2="240" y2="20" stroke="#00d4aa" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="243" y="23" fill="#00d4aa" font-size="8" font-family="monospace">Graham</text>
  <circle cx="90" cy="45" r="5" fill="#10b981"/>
  <text x="80" y="62" fill="#10b981" font-size="8" font-family="monospace">Sous-evalue</text>
  <circle cx="180" cy="14" r="5" fill="#ef4444"/>
  <text x="165" y="62" fill="#ef4444" font-size="8" font-family="monospace">Sur-evalue</text>
  <line x1="90" y1="20" x2="90" y2="45" stroke="#10b981" stroke-width="1" stroke-dasharray="2,2"/>
  <line x1="180" y1="14" x2="180" y2="20" stroke="#ef4444" stroke-width="1" stroke-dasharray="2,2"/>
  <text x="92" y="35" fill="#10b981" font-size="8" font-family="monospace">-25%</text>
  <text x="182" y="18" fill="#ef4444" font-size="8" font-family="monospace">+30%</text>
</svg>
</div>''')

page.append('</div>') # fin guide-grid valeur

# ─── CATEGORIE: CROISSANCE & QUALITE ───
page.append('<div class="guide-cat">&#127807; Croissance & Qualite du Bilan</div>')
page.append('<div class="guide-grid">')

# ROE
page.append('''<div class="guide-card">
<div class="gc-title">ROE <span class="gc-badge badge-growth">Croissance</span></div>
<div class="gc-desc">Return on Equity. Benefice net / Capitaux propres. Mesure l\'efficacite avec laquelle l\'entreprise genere du profit pour ses actionnaires. Warren Buffett vise &gt;20%.</div>
<div class="gc-zones">
  <span class="zone zone-g">&ge;25% (+35pts)</span>
  <span class="zone zone-y">&ge;15% (+25pts) / &ge;8% (+12pts)</span>
  <span class="zone zone-r">&lt;8% (0pt)</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="5" y="12" fill="#64748b" font-size="8" font-family="monospace">ROE = Benefice Net / Capitaux Propres x 100</text>
  <rect x="5" y="22" width="30" height="40" fill="#1e2d45" rx="3"/>
  <rect x="5" y="22" width="30" height="40" fill="none" stroke="#1e2d4580" stroke-width="1" rx="3"/>
  <rect x="5" y="38" width="30" height="24" fill="#10b98130" rx="3"/>
  <text x="20" y="60" fill="#10b981" font-size="7" font-family="monospace" text-anchor="middle">Bnf</text>
  <rect x="45" y="18" width="30" height="44" fill="#1e2d45" rx="3"/>
  <rect x="45" y="36" width="30" height="26" fill="#10b98130" rx="3"/>
  <rect x="85" y="15" width="30" height="47" fill="#1e2d45" rx="3"/>
  <rect x="85" y="40" width="30" height="22" fill="#f59e0b20" rx="3"/>
  <rect x="125" y="30" width="30" height="32" fill="#1e2d45" rx="3"/>
  <rect x="125" y="50" width="30" height="12" fill="#ef444420" rx="3"/>
  <text x="20" y="17" fill="#10b981" font-size="7" font-family="monospace" text-anchor="middle">ROE 30%</text>
  <text x="60" y="13" fill="#10b981" font-size="7" font-family="monospace" text-anchor="middle">ROE 25%</text>
  <text x="100" y="10" fill="#f59e0b" font-size="7" font-family="monospace" text-anchor="middle">ROE 15%</text>
  <text x="140" y="25" fill="#ef4444" font-size="7" font-family="monospace" text-anchor="middle">ROE 8%</text>
</svg>
</div>''')

# PEG
page.append('''<div class="guide-card">
<div class="gc-title">PEG Ratio <span class="gc-badge badge-growth">Croissance</span></div>
<div class="gc-desc">PER / Taux de croissance des benefices. Corrige le PER de la croissance. PEG &lt;1 = action sous-evaluee par rapport a sa croissance (Peter Lynch).</div>
<div class="gc-zones">
  <span class="zone zone-g">&le;1 (+25pts) — Sous-evalue</span>
  <span class="zone zone-y">&le;1.5 (+15pts) / &le;2.5 (+8pts)</span>
  <span class="zone zone-r">&gt;2.5 (0pt)</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="140" y="11" fill="#64748b" font-size="8" font-family="monospace" text-anchor="middle">PEG = PER / Croissance benefices (%)</text>
  <rect x="5" y="18" width="82" height="46" fill="#10b98110" rx="4" stroke="#10b98130"/>
  <rect x="97" y="25" width="82" height="39" fill="#f59e0b10" rx="4" stroke="#f59e0b30"/>
  <rect x="189" y="38" width="82" height="26" fill="#ef444410" rx="4" stroke="#ef444430"/>
  <text x="46" y="42" fill="#10b981" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">PEG &le;1</text>
  <text x="138" y="48" fill="#f59e0b" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">PEG 1-2.5</text>
  <text x="230" y="55" fill="#ef4444" font-size="10" font-family="monospace" font-weight="bold" text-anchor="middle">PEG &gt;2.5</text>
  <text x="46" y="55" fill="#10b981" font-size="8" font-family="monospace" text-anchor="middle">Sous-evalue</text>
  <text x="138" y="60" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">Correct</text>
  <text x="230" y="64" fill="#ef4444" font-size="8" font-family="monospace" text-anchor="middle">Cher</text>
</svg>
</div>''')

# Current Ratio
page.append('''<div class="guide-card">
<div class="gc-title">Current Ratio <span class="gc-badge badge-qual">Qualite</span></div>
<div class="gc-desc">Actifs courants / Passifs courants. Mesure la capacite de l\'entreprise a honorer ses dettes a court terme. Un ratio &lt;1 signifie un risque de liquidite.</div>
<div class="gc-zones">
  <span class="zone zone-g">&ge;2 (+8pts) — Tres solide</span>
  <span class="zone zone-y">&ge;1.5 (+5pts) / &ge;1 (+2pts)</span>
  <span class="zone zone-r">&lt;1 — Risque liquidite</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="140" y="11" fill="#64748b" font-size="8" font-family="monospace" text-anchor="middle">Actifs Courants / Passifs Courants</text>
  <rect x="20" y="20" width="80" height="40" fill="#10b98120" rx="4" stroke="#10b98140"/>
  <rect x="130" y="36" width="50" height="24" fill="#1e2d45" rx="4" stroke="#1e2d4580"/>
  <text x="60" y="42" fill="#10b981" font-size="9" font-family="monospace" text-anchor="middle">Actifs</text>
  <text x="60" y="52" fill="#10b981" font-size="9" font-family="monospace" text-anchor="middle">Courants</text>
  <text x="155" y="50" fill="#64748b" font-size="9" font-family="monospace" text-anchor="middle">Passifs</text>
  <text x="60" y="30" fill="#10b981" font-size="11" font-family="monospace" font-weight="bold" text-anchor="middle">2x</text>
  <text x="230" y="38" fill="#10b981" font-size="9" font-family="monospace">CR = 2</text>
  <text x="230" y="50" fill="#10b981" font-size="9" font-family="monospace">Solide !</text>
</svg>
</div>''')

# Debt/Equity
page.append('''<div class="guide-card">
<div class="gc-title">Debt/Equity <span class="gc-badge badge-qual">Qualite</span></div>
<div class="gc-desc">Dette totale / Capitaux propres (en %). Mesure le levier financier. Une dette elevee amplifie les gains ET les pertes. Varie beaucoup selon le secteur.</div>
<div class="gc-zones">
  <span class="zone zone-g">&le;30% (+10pts) — Peu endetee</span>
  <span class="zone zone-y">&le;80% (+6pts) / &le;150% (+3pts)</span>
  <span class="zone zone-r">&gt;150% — Fort levier</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <rect x="5" y="15" width="60" height="45" fill="#10b98120" rx="4"/>
  <rect x="5" y="15" width="60" height="12" fill="#10b98140" rx="4"/>
  <text x="35" y="25" fill="#10b981" font-size="7" font-family="monospace" text-anchor="middle">Dette</text>
  <text x="35" y="50" fill="#10b981" font-size="8" font-family="monospace" text-anchor="middle">Capitaux</text>
  <rect x="80" y="8" width="60" height="52" fill="#f59e0b15" rx="4"/>
  <rect x="80" y="8" width="60" height="26" fill="#f59e0b30" rx="4"/>
  <rect x="155" y="5" width="60" height="55" fill="#ef444415" rx="4"/>
  <rect x="155" y="5" width="60" height="42" fill="#ef444430" rx="4"/>
  <text x="110" y="44" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">D/E 80%</text>
  <text x="185" y="55" fill="#ef4444" font-size="8" font-family="monospace" text-anchor="middle">D/E 150%</text>
  <text x="35" y="68" fill="#10b981" font-size="8" font-family="monospace" text-anchor="middle">D/E 30%</text>
  <text x="235" y="40" fill="#ef4444" font-size="8" font-family="monospace">Risque</text>
  <text x="235" y="50" fill="#ef4444" font-size="8" font-family="monospace">eleve</text>
</svg>
</div>''')

# FCF Yield
page.append('''<div class="guide-card">
<div class="gc-title">FCF Yield <span class="gc-badge badge-qual">Qualite</span></div>
<div class="gc-desc">Free Cash Flow / Capitalisation boursiere (%). Rendement du flux de tresorerie libre pour l\'actionnaire. Comparable a un rendement obligataire mais pour les actions.</div>
<div class="gc-zones">
  <span class="zone zone-g">&ge;8% (+10pts)</span>
  <span class="zone zone-y">&ge;5% (+6pts) / &ge;2% (+3pts)</span>
  <span class="zone zone-r">&lt;2% ou negatif</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="140" y="11" fill="#64748b" font-size="8" font-family="monospace" text-anchor="middle">FCF Yield = Free Cash Flow / Market Cap x 100%</text>
  <rect x="5" y="18" width="55" height="45" fill="#10b98120" rx="4"/>
  <rect x="70" y="28" width="55" height="35" fill="#f59e0b20" rx="4"/>
  <rect x="135" y="38" width="55" height="25" fill="#f59e0b10" rx="4"/>
  <rect x="200" y="50" width="55" height="13" fill="#ef444420" rx="4"/>
  <text x="32" y="44" fill="#10b981" font-size="9" font-family="monospace" text-anchor="middle">8%+</text>
  <text x="97" y="50" fill="#f59e0b" font-size="9" font-family="monospace" text-anchor="middle">5%+</text>
  <text x="162" y="55" fill="#f59e0b" font-size="9" font-family="monospace" text-anchor="middle">2%+</text>
  <text x="227" y="60" fill="#ef4444" font-size="9" font-family="monospace" text-anchor="middle">&lt;2%</text>
</svg>
</div>''')

# Dividende
page.append('''<div class="guide-card">
<div class="gc-title">Rendement Dividende <span class="gc-badge badge-div">Dividende</span></div>
<div class="gc-desc">Dividende annuel / Cours de l\'action (%). Revenu passif verse aux actionnaires. Attention: un rendement trop eleve peut signaler une baisse du cours ou un dividende non soutenable.</div>
<div class="gc-zones">
  <span class="zone zone-g">&ge;6% (+60pts)</span>
  <span class="zone zone-y">&ge;4% (+45pts) / &ge;3% (+30pts)</span>
  <span class="zone zone-r">0% (0pt)</span>
</div>
<svg class="gc-svg" viewBox="0 0 280 70" xmlns="http://www.w3.org/2000/svg">
  <text x="140" y="11" fill="#64748b" font-size="8" font-family="monospace" text-anchor="middle">Dividende annuel / Cours x 100%</text>
  <circle cx="50" cy="42" r="22" fill="#8b5cf618" stroke="#8b5cf640" stroke-width="1.5"/>
  <circle cx="50" cy="42" r="12" fill="#8b5cf630"/>
  <text x="50" y="46" fill="#8b5cf6" font-size="9" font-family="monospace" font-weight="bold" text-anchor="middle">6%+</text>
  <circle cx="130" cy="42" r="16" fill="#f59e0b18" stroke="#f59e0b30" stroke-width="1.5"/>
  <text x="130" y="46" fill="#f59e0b" font-size="9" font-family="monospace" font-weight="bold" text-anchor="middle">4%+</text>
  <circle cx="195" cy="42" r="11" fill="#f59e0b10" stroke="#f59e0b20" stroke-width="1"/>
  <text x="195" y="46" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">3%</text>
  <circle cx="248" cy="42" r="6" fill="#64748b10" stroke="#64748b30" stroke-width="1"/>
  <text x="248" y="46" fill="#64748b" font-size="7" font-family="monospace" text-anchor="middle">0%</text>
  <text x="50" y="70" fill="#8b5cf6" font-size="8" font-family="monospace" text-anchor="middle">60pts</text>
  <text x="130" y="70" fill="#f59e0b" font-size="8" font-family="monospace" text-anchor="middle">45pts</text>
</svg>
</div>''')

page.append('</div>') # fin guide-grid qualite

# Légende globale du score
page.append('''<div style="background:#111827;border:1px solid #8b5cf644;border-radius:12px;padding:20px;margin-top:24px">
<div style="font-family:monospace;font-size:13px;font-weight:900;color:#8b5cf6;margin-bottom:12px">&#127919; Calcul du Score Global</div>
<div style="font-size:11px;color:#64748b;line-height:1.8;margin-bottom:14px">
Le score sur 100 combine quatre dimensions pondereables. Chaque sous-score est calcule sur 100 puis pondere.<br>
<span style="color:#e2e8f0">Score Fond = (Score Value x P.Value + Score Growth x P.Growth + Score Div x P.Div) / (P.Value + P.Growth + P.Div)</span><br>
<span style="color:#e2e8f0">Score Final = Score Fond x (1 - P.Technique/100) + Score Tech x (P.Technique/100)</span>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px">
<div style="background:#0d1520;border-radius:8px;padding:10px;text-align:center">
  <div style="font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.7px">Valorisation</div>
  <div style="font-family:monospace;font-size:18px;font-weight:900;color:#10b981">30%</div>
  <div style="font-size:9px;color:#64748b">PER, PBV, EV, P/S, P/FCF, Graham</div>
</div>
<div style="background:#0d1520;border-radius:8px;padding:10px;text-align:center">
  <div style="font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.7px">Croissance</div>
  <div style="font-family:monospace;font-size:18px;font-weight:900;color:#f59e0b">25%</div>
  <div style="font-size:9px;color:#64748b">ROE, BPA, PEG, CR, D/E, FCF Yield</div>
</div>
<div style="background:#0d1520;border-radius:8px;padding:10px;text-align:center">
  <div style="font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.7px">Dividende</div>
  <div style="font-family:monospace;font-size:18px;font-weight:900;color:#8b5cf6">15%</div>
  <div style="font-size:9px;color:#64748b">Rendement, Payout Ratio</div>
</div>
<div style="background:#0d1520;border-radius:8px;padding:10px;text-align:center">
  <div style="font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.7px">Technique</div>
  <div style="font-family:monospace;font-size:18px;font-weight:900;color:#3b82f6">30%</div>
  <div style="font-size:9px;color:#64748b">RSI, MACD, CCI, Stoch, ADX, MM, W%R, RoC</div>
</div>
</div>
</div>''')

page.append('</div></div>') # fin guide-wrap + guide-page

# ══ LOGIN SCREEN ══
page.append('<div class="login-screen" id="login-screen">')
page.append('<div class="login-box">')
page.append('<div class="login-logo">&#128225; Stock Screener</div>')
page.append('<div class="login-sub">Acces prive — entrez le code</div>')
page.append('<input class="login-input" type="password" id="login-pwd" placeholder="••••••••" onkeydown="if(event.key===\'Enter\')doLogin()" autocomplete="current-password">')
page.append('<div class="login-err" id="login-err"></div>')
page.append('<button class="login-btn" id="login-btn">&#128274; Connexion</button>')
page.append('</div></div>')
page.append('<div class="wrap">')
page.append('<div class="mrow">')
page.append('<div class="mbtn on" id="m-combine" onclick="setMode(\'combine\')"><div class="mico">&#127919;</div><div class="mlbl">Fond+Tech</div></div>')
page.append('<div class="mbtn" id="m-fond" onclick="setMode(\'fond\')"><div class="mico">&#128202;</div><div class="mlbl">Fondamental</div></div>')
page.append('<div class="mbtn" id="m-tech" onclick="setMode(\'tech\')"><div class="mico">&#128200;</div><div class="mlbl">Technique</div></div>')
page.append('</div>')
page.append('<div class="mdesc" id="mdesc">Mode Combine: score fondamentaux + technique.</div>')
page.append('<div class="filters">')
page.append('<div class="fg"><label>Recherche</label><input type="text" id="q" oninput="render()" placeholder="Ticker..." style="width:120px"></div>')
page.append('<div class="fg"><label>Marche</label><select id="mkt" onchange="render()"><option value="all">Tous</option><option value="eu">Europe</option><option value="us">USA</option><option value="ca">Canada</option><option value="other">Autres</option></select></div>')
page.append('<div class="fg"><label>Secteur</label><select id="sect" onchange="render()"></select></div>')
page.append('<div class="fg"><label>Score min</label><input type="number" id="ms" value="' + SMIN + '" min="0" max="100" step="5" onchange="render()" style="width:60px"></div>')
page.append('<div class="fg"><label>PEA seul.</label><select id="peaOnly" onchange="render()"><option value="0">Non</option><option value="1">Oui</option></select></div>')
page.append('</div>')
page.append('<div class="tabs">')
page.append('<div class="tab on" id="tab-all" onclick="setTab(\'all\')">Toutes <span class="bdg" id="b-all">0</span></div>')
page.append('<div class="tab" id="tab-pea" onclick="setTab(\'pea\')">PEA <span class="bdg" id="b-pea">0</span></div>')
page.append('<div class="tab" id="tab-div" onclick="setTab(\'div\')">&#9888; Divergences <span class="bdg" id="b-div">0</span></div>')
page.append('<div class="tab" id="tab-imp" onclick="setTab(\'imp\')" style="display:none">Import <span class="bdg" id="b-imp">0</span></div>')
page.append('</div>')
# Sub-score filters
page.append('<div class="ssf">')
page.append('<span class="ssf-lbl">Filtres scores:</span>')
page.append('<div class="ssf-item"><label>Value&ge;</label><input type="number" id="f-sv" value="0" min="0" max="100" step="5" onchange="render()"></div>')
page.append('<div class="ssf-item"><label>Growth&ge;</label><input type="number" id="f-sg" value="0" min="0" max="100" step="5" onchange="render()"></div>')
page.append('<div class="ssf-item"><label>Div&ge;</label><input type="number" id="f-sd" value="0" min="0" max="100" step="5" onchange="render()"></div>')
page.append('<div class="ssf-item"><label>Tech&ge;</label><input type="number" id="f-st" value="0" min="0" max="100" step="5" onchange="render()"></div>')
page.append('<div class="ssf-item"><label>Mom&ge;</label><input type="number" id="f-smom" value="0" min="0" max="100" step="5" onchange="render()"></div>')
page.append('</div>')
page.append('<div class="stale-banner" id="stale-banner">&#9888; Donnees potentiellement perimees — dernier scan: ' + DATE + '. Weekend ou jour ferie ?</div>')
page.append('<div class="stats" id="stats"></div>')
page.append('<div class="tw"><div class="ts"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table>')
page.append('<div class="empty" id="empty" style="display:none">Aucun resultat</div></div></div>')
page.append('<div class="card-list" id="card-list"></div>')
page.append('<div class="disc">Donnees Yahoo Finance a titre indicatif. Ne constitue pas un conseil en investissement.</div>')
page.append('</div>')

# ══ MODAL COLONNES ══
page.append('<div class="modal-bg" id="modal-cols"><div class="modal" style="max-width:560px">')
page.append('<button class="modal-close" onclick="closeCols()">&#10005;</button>')
page.append('<h3>&#9638; Colonnes affichees</h3>')
page.append('<div style="font-size:10px;color:#64748b;margin-bottom:10px">Cochez pour afficher. Decochez pour masquer.</div>')
page.append('<div class="col-grid" id="col-grid"></div>')
page.append('<button class="modal-btn modal-btn-ok" onclick="applyCols()">&#10003; Appliquer</button>')
page.append('<button class="modal-btn modal-btn-cancel" onclick="resetCols()" style="margin-top:8px">Reinitialiser par defaut</button>')
page.append('</div></div>')

# ══ MODAL COMPARATEUR ══
page.append('<div class="modal-bg" id="modal-cmp"><div class="modal" style="max-width:700px">')
page.append('<button class="modal-close" onclick="closeComparator()">&#10005;</button>')
page.append('<h3>&#9878; Comparateur</h3>')
page.append('<div id="cmp-content"></div>')
page.append('<button class="modal-btn modal-btn-cancel" onclick="closeComparator()" style="margin-top:12px">Fermer</button>')
page.append('</div></div>')

page.append('<script>')
page.append('// LOGIN ISOLE — tourne avant tout le reste')
page.append('(function(){')
page.append('  var HASH="__PWD_HASH__";')
page.append('  var screen=document.getElementById("login-screen");')
page.append('  if(!screen)return;')
page.append('  document.getElementById("login-btn").addEventListener("click",function(){tryLogin();});')
page.append('  document.getElementById("login-pwd").addEventListener("keydown",function(e){if(e.key==="Enter")tryLogin();});')
page.append('  function tryLogin(){')
page.append('    var pwd=document.getElementById("login-pwd").value;')
page.append('    var err=document.getElementById("login-err");')
page.append('    if(!pwd){err.textContent="Entrez un code.";return;}')
page.append('    err.textContent="...";')
page.append('    crypto.subtle.digest("SHA-256",new TextEncoder().encode(pwd))')
page.append('    .then(function(buf){')
page.append('      var hex=Array.from(new Uint8Array(buf)).map(function(b){return b.toString(16).padStart(2,"0");}).join("");')
page.append('      if(hex===HASH){screen.style.display="none";}')
page.append('      else{err.textContent="Code incorrect.";document.getElementById("login-pwd").value="";document.getElementById("login-pwd").focus();}')
page.append('    }).catch(function(){err.textContent="Erreur navigateur.";});')
page.append('  }')
page.append('})();')
page.append('</script>')
js = r"""
var mode="combine",curTab="all";
var sortCols=[{col:"score",dir:-1}];
var CMP_SET={};
var USER_COLS=null;

// All available columns definition
var ALL_COLS_DEF={
  fond:['ticker','spark','prix','per','pbv','roe','marge','div','bpa','ev','peg','ps','pfcf','graham','cr','de','fcfy','sv','sg','sd','smom','score','divergence'],
  tech:['ticker','spark','prix','rsi','macd','cci','stoch','adx','mm20','mm50','mm200','willr','roc','st','smom','signaux'],
  combine:['ticker','spark','prix','per','roe','div','rsi','macd','cci','mm200','graham','fcfy','sv','st','smom','score','divergence','signaux']
};
var CL={ticker:'Action',spark:'Cours 60j',prix:'Cours',per:'PER',pbv:'P/Book',roe:'ROE%',marge:'Marge%',div:'Div%',bpa:'BPA%',ev:'EV/EBITDA',peg:'PEG',ps:'P/S',pfcf:'P/FCF',graham:'Graham%',cr:'Curr.R',de:'D/E%',fcfy:'FCF Yld',sv:'Score V',sg:'Score G',sd:'Score D',st:'Score T',smom:'Momentum',score:'Score',divergence:'Diverg.',rsi:'RSI',macd:'MACD',cci:'CCI',stoch:'Stoch',adx:'ADX',mm20:'MM20%',mm50:'MM50%',mm200:'MM200%',willr:'W%R',roc:'RoC%',signaux:'Signaux'};

// localStorage helpers
var LS_KEY='screener_prefs_v3';
function savePrefs(){
  try{
    localStorage.setItem(LS_KEY,JSON.stringify({
      cfg:CFG, sortCols:sortCols, userCols:USER_COLS,
      theme:document.body.classList.contains('light')?'light':'dark',
      filters:{q:'',mkt:'all',sect:'all',ms:CFG.scoreMin,pea:'0',sv:0,sg:0,sd:0,st:0,smom:0}
    }));
  }catch(e){}
}
function loadPrefs(){
  try{
    var raw=localStorage.getItem(LS_KEY);
    if(!raw)return;
    var p=JSON.parse(raw);
    if(p.cfg&&typeof p.cfg==='object')Object.assign(CFG,p.cfg);
    if(Array.isArray(p.sortCols)&&p.sortCols.length)sortCols=p.sortCols;
    if(p.userCols&&typeof p.userCols==='object')USER_COLS=p.userCols;
    if(p.theme==='light')document.body.classList.add('light');
    var msEl=document.getElementById('ms');if(msEl)msEl.value=CFG.scoreMin;
  }catch(e){
    try{localStorage.removeItem(LS_KEY);}catch(e2){}
  }
}

// Freshness check
function checkFreshness(){
  try{
    var parts=SCAN_DATE.split(/[/ :]/);
    var d=new Date(parts[2],parts[1]-1,parts[0],parseInt(parts[3])||7,parseInt(parts[4])||0);
    var diff=(Date.now()-d)/3600000;
    var el=document.getElementById('stale-banner');
    if(diff>72&&el)el.style.display='flex';
  }catch(e){}
}
checkFreshness();

function buildSects(){
  var s=['all'],i,o;
  for(i=0;i<RAW_ORIG.length;i++){if(RAW_ORIG[i].secteur&&s.indexOf(RAW_ORIG[i].secteur)===-1)s.push(RAW_ORIG[i].secteur);}
  s.sort();
  var se=document.getElementById('sect'); se.innerHTML='';
  for(i=0;i<s.length;i++){o=document.createElement('option');o.value=s[i];o.text=s[i]==='all'?'Tous secteurs':s[i];se.appendChild(o);}
}
buildSects();
loadPrefs();

function nextUpdate(){
  var now=new Date(),nxt=new Date();
  nxt.setDate(now.getDate()+(now.getHours()>=7?1:0));nxt.setHours(7,0,0,0);
  var diff=nxt-now,h=Math.floor(diff/3600000),m=Math.floor((diff%3600000)/60000);
  var el=document.getElementById('nxt');if(el)el.textContent='Scan auto: '+h+'h'+m+'m';
}
nextUpdate();setInterval(nextUpdate,60000);

// Theme
function toggleTheme(){
  document.body.classList.toggle('light');
  document.getElementById('theme-btn').textContent=document.body.classList.contains('light')?'\u263D':'\u2600';
  savePrefs();
}

var DESCS={combine:'Mode Combine: fondamentaux + technique. Recommande.',fond:'Mode Fondamental: PER, ROE, dividende, croissance.',tech:'Mode Technique: RSI, MACD, CCI, Stochastique, MM.'};

function setMode(m){
  mode=m;
  ['combine','fond','tech'].forEach(function(x){document.getElementById('m-'+x).className='mbtn'+(x===m?' on':'');});
  document.getElementById('mdesc').textContent=DESCS[m];render();
}
function setTab(t){
  curTab=t;
  ['all','pea','div','imp'].forEach(function(x){var el=document.getElementById('tab-'+x);if(el)el.className='tab'+(x===t?' on':'');});
  render();
}
function openGuide(){document.getElementById('guide-page').className='guide-page open';}
function closeGuide(){document.getElementById('guide-page').className='guide-page';}

// Scoring functions
function lsv(r){
  var p=0;
  if(CFG.tPer&&r.per)p+=r.per<=10?40:r.per<=15?28:r.per<=CFG.perMax?15:0;
  if(CFG.tPbv&&r.pbv)p+=r.pbv<=1?30:r.pbv<=2?20:r.pbv<=CFG.pbvMax?10:0;
  if(CFG.tEv&&r.ev)p+=r.ev<=8?30:r.ev<=12?20:r.ev<=15?10:0;
  if(CFG.tPs&&r.ps)p+=r.ps<=1?10:r.ps<=2?6:r.ps<=CFG.psMax?3:0;
  if(CFG.tPfcf&&r.pfcf)p+=r.pfcf<=10?10:r.pfcf<=20?6:r.pfcf<=CFG.pfcfMax?3:0;
  if(CFG.tGraham&&r.graham!==null&&r.graham!==undefined)p+=r.graham>=(CFG.grahamMin+20)?15:r.graham>=CFG.grahamMin?10:0;
  return Math.min(100,p);
}
function lsg(r){
  var p=0;
  if(CFG.tRoe&&r.roe)p+=r.roe>=25?35:r.roe>=15?25:r.roe>=CFG.roeMin?12:0;
  if(CFG.tBpa&&r.bpa)p+=r.bpa>=25?40:r.bpa>=15?28:r.bpa>0?15:0;
  if(CFG.tPeg&&r.peg)p+=r.peg<=1?25:r.peg<=1.5?15:r.peg<=2.5?8:0;
  if(CFG.tCr&&r.cr)p+=r.cr>=(CFG.crMin+1)?8:r.cr>=(CFG.crMin+0.5)?5:r.cr>=CFG.crMin?2:0;
  if(CFG.tDe&&r.de!==null&&r.de!==undefined)p+=r.de<=30?10:r.de<=80?6:r.de<=CFG.deMax?3:0;
  if(CFG.tFcfy&&r.fcfy)p+=r.fcfy>=8?10:r.fcfy>=5?6:r.fcfy>=CFG.fcfyMin?3:0;
  return Math.min(100,p);
}
function lsd(r){var p=0;if(r.div)p+=r.div>=6?60:r.div>=4?45:r.div>=3?30:r.div>0?15:0;return Math.min(100,p);}
function lst(r){
  var p=0;
  if(CFG.tRsi&&r.rsi!==null)if(r.rsi<=30)p+=28;else if(r.rsi<=CFG.rsiSurvente)p+=18;else if(r.rsi<=50)p+=8;
  if(CFG.tMacd&&r.macd!==null)if(r.macd>0)p+=15;else if(r.macd>-0.1)p+=5;
  if(CFG.tCci&&r.cci!==null)if(r.cci<CFG.cciSurvente)p+=18;else if(r.cci>=50&&r.cci<=150)p+=12;else if(r.cci>0)p+=6;
  if(CFG.tStoch&&r.stoch!==null)if(r.stoch<=20)p+=14;else if(r.stoch<=CFG.stochSurvente)p+=9;
  if(CFG.tMm&&r.mm20!==null&&r.mm50!==null&&r.mm200!==null&&r.mm20>0&&r.mm50>0&&r.mm200>0)p+=15;
  if(CFG.tAdx&&r.adx!==null&&r.adx>=CFG.adxMin)p+=5;
  if(CFG.tWillr&&r.willr!==null)if(r.willr<=-80)p+=8;else if(r.willr<=CFG.willrSurvente)p+=4;
  if(CFG.tRoc&&r.roc!==null&&r.roc>=CFG.rocMin)p+=5;
  return Math.min(100,p);
}
function lsmom(r){
  var p=0;
  if(r.roc!==null&&r.roc!==undefined)p+=r.roc>10?40:r.roc>5?25:r.roc>0?10:0;
  if(r.willr!==null&&r.willr!==undefined)p+=r.willr<=-80?35:r.willr<=-60?20:r.willr<=-40?8:0;
  if(r.macd!==null&&r.macd!==undefined)p+=r.macd>0?25:r.macd>-0.1?10:0;
  return Math.min(100,p);
}
function ls(r){var sv=lsv(r),sg=lsg(r),sd=lsd(r),st=lst(r),tf=CFG.poidsValue+CFG.poidsGrowth+CFG.poidsDividende;return Math.round(Math.round(sv*CFG.poidsValue/tf+sg*CFG.poidsGrowth/tf+sd*CFG.poidsDividende/tf)*(1-CFG.poidsTechnique/100)+st*(CFG.poidsTechnique/100));}
function getScore(r){if(mode==='fond')return Math.round(lsv(r)*0.4+lsg(r)*0.35+lsd(r)*0.25);if(mode==='tech')return lst(r);return ls(r);}

// Multi-column sort
function doSort(col){
  if(sortCols.length>0&&sortCols[0].col===col){
    sortCols[0].dir*=-1;
  } else {
    sortCols=sortCols.filter(function(s){return s.col!==col;});
    sortCols.unshift({col:col,dir:-1});
    if(sortCols.length>3)sortCols=sortCols.slice(0,3);
  }
  savePrefs();render();
}
function compareRows(a,b){
  for(var i=0;i<sortCols.length;i++){
    var s=sortCols[i],col=s.col,dir=s.dir;
    var av,bv;
    if(col==='score'){av=getScore(a);bv=getScore(b);}
    else if(col==='smom'){av=lsmom(a);bv=lsmom(b);}
    else{av=a[col];bv=b[col];}
    if(av===null||av===undefined){if(bv===null||bv===undefined)continue;return 1;}
    if(bv===null||bv===undefined)return -1;
    var na=parseFloat(av),nb=parseFloat(bv);
    var diff=!isNaN(na)&&!isNaN(nb)?dir*(na-nb):dir*String(av).localeCompare(String(bv));
    if(diff!==0)return diff;
  }
  return 0;
}

function doFilter(){
  var q=(document.getElementById('q').value||'').toLowerCase();
  var mkt=document.getElementById('mkt').value;
  var sec=document.getElementById('sect').value;
  var ms=parseFloat(document.getElementById('ms').value)||0;
  var po=document.getElementById('peaOnly').value==='1';
  var fsv=parseFloat(document.getElementById('f-sv').value)||0;
  var fsg=parseFloat(document.getElementById('f-sg').value)||0;
  var fsd=parseFloat(document.getElementById('f-sd').value)||0;
  var fst=parseFloat(document.getElementById('f-st').value)||0;
  var fsmom=parseFloat(document.getElementById('f-smom').value)||0;
  var src=(curTab==='imp'&&IMP_DATA)?IMP_DATA:RAW_ORIG;
  var out=[],i,r;
  for(i=0;i<src.length;i++){
    r=src[i];
    if(getScore(r)<ms)continue;
    if(po&&!r.pea)continue;
    if(curTab==='pea'&&!r.pea)continue;
    if(curTab==='div'&&!r.divergence)continue;
    if(mkt==='eu'&&(r.pays==='US'||r.pays==='CA'||r.pays==='AU'))continue;
    if(mkt==='us'&&r.pays!=='US')continue;
    if(mkt==='ca'&&r.pays!=='CA')continue;
    if(mkt==='other'&&(r.pays==='US'||r.pays==='CA'||r.pays==='FR'||r.pays==='DE'||r.pays==='GB'||r.pays==='NL'||r.pays==='ES'||r.pays==='IT'||r.pays==='CH'))continue;
    if(sec!=='all'&&r.secteur!==sec)continue;
    if(q&&r.ticker.toLowerCase().indexOf(q)===-1&&r.nom.toLowerCase().indexOf(q)===-1)continue;
    if(lsv(r)<fsv||lsg(r)<fsg||lsd(r)<fsd||lst(r)<fst||lsmom(r)<fsmom)continue;
    out.push(r);
  }
  out.sort(compareRows);
  return out;
}

// Sparkline SVG renderer
function sparkSVG(data){
  if(!data||data.length<2)return '<span style="color:#2a3a52;font-size:9px">-</span>';
  var w=70,h=22,pts=data,n=pts.length;
  var xStep=w/(n-1);
  var coords=pts.map(function(v,i){return [Math.round(i*xStep),Math.round(h-(v/100*h))];});
  var path=coords.map(function(p,i){return (i===0?'M':'L')+p[0]+' '+p[1];}).join(' ');
  var last=pts[pts.length-1],first=pts[0];
  var trend=last>first+5?'#10b981':last<first-5?'#ef4444':'#64748b';
  var dot=coords[coords.length-1];
  return '<svg class="spark" width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'">'
    +'<path d="'+path+'" fill="none" stroke="'+trend+'" stroke-width="1.5" stroke-linejoin="round"/>'
    +'<circle cx="'+dot[0]+'" cy="'+dot[1]+'" r="2" fill="'+trend+'"/>'
    +'</svg>';
}

// Tooltips
var TIPS={
  ticker:'Action. Cliquez pour voir sur Yahoo Finance.',spark:'Mini-courbe des 60 derniers jours. Vert=hausse, Rouge=baisse.',
  prix:'Dernier cours au moment du scan.',per:'PER: cours/benefice. Moins c'est eleve, moins vous payez les benefices.',
  pbv:'P/Book: cours/valeur comptable. <1 = decote sur actifs.',roe:'ROE: benefice/capitaux propres. Mesure l'efficacite.',
  marge:'Marge nette: benefice net/CA.',div:'Rendement dividende annuel en % du cours.',
  bpa:'Croissance des benefices par action.',ev:'EV/EBITDA: valorisation independante de la structure financiere.',
  peg:'PER/croissance. <1 = sous-evalue vs sa croissance (Peter Lynch).',
  ps:'P/S: cours/CA. Utile quand le PER est indisponible.',
  pfcf:'P/FCF: plus fiable que le PER car difficile a manipuler.',
  graham:'Ecart vs valeur Graham sqrt(22.5xBPAxVNC). Positif=sous-evalue.',
  cr:'Current Ratio: actifs/passifs courants. >1.5 = bonne liquidite.',
  de:'D/E: dette/capitaux propres. Bas = moins endetee.',
  fcfy:'FCF Yield: free cash flow/capitalisation. Comparable a un rendement obligataire.',
  sv:'Score Valorisation: PER,PBV,EV,P/S,P/FCF,Graham.',
  sg:'Score Croissance: ROE,BPA,PEG,Current Ratio,D/E,FCF Yield.',
  sd:'Score Dividende: rendement et payout ratio.',
  st:'Score Technique: RSI,MACD,CCI,Stoch,ADX,MM,Williams,RoC.',
  smom:'Score Momentum: RoC+Williams %R+MACD. Mesure l'acceleration du cours.',
  score:'Score global = fondamentaux + technique selon les poids.',
  divergence:'FOND+: bons fondamentaux + technique faible = opportunite. TECH+: technique fort + fondamentaux faibles = prudence.',
  rsi:'RSI 14j: <30 survente, >70 surachat.',macd:'MACD 12/26/9: positif = momentum haussier.',
  cci:'CCI 20j: <-100 survente, >+100 tendance.',stoch:'Stoch %K 14j: <20 survente, >80 surachat.',
  adx:'ADX: force de tendance. >20=etablie, >40=forte.',
  mm20:'Ecart cours/MM20j.',mm50:'Ecart cours/MM50j.',mm200:'Ecart cours/MM200j (tendance long terme).',
  willr:'Williams %R 14j: <-80 survente, >-20 surachat.',
  roc:'Rate of Change 12p: vitesse du mouvement. >+5%=acceleration.',
  signaux:'Signaux techniques actifs au moment du scan.'
};

function thLabel(col,sorted,dir,rank){
  var lbl=CL[col]||col;
  var tip=TIPS[col]?'<span class="tip-ico">?</span><div class="tip">'+TIPS[col]+'</div>':'';
  var sortInd=sorted?(dir>0?' ^':' v'):'';
  var rankBadge=rank>1?'<span class="sort-badge">'+rank+'</span>':'';
  return '<div class="th-wrap">'+lbl+sortInd+rankBadge+tip+'</div>';
}

function cH(col,r){
  var v=r[col];
  if(col==='ticker'){
    var tg='';if(r.pea)tg+='<span class="ptag">PEA</span>';if(r._imported)tg+='<span class="imptag">IMP</span>';
    var cmpOn=CMP_SET[r.ticker]?'on':'';
    return '<div><span class="tk" style="cursor:pointer" onclick="window.open('https://finance.yahoo.com/quote/'+r.ticker+'','_blank')">'+r.ticker+'</span>'+tg
      +'<button class="cmp-btn '+cmpOn+'" onclick="toggleCmp(''+r.ticker+'',this)">&#9878;</button></div>'
      +'<div class="nm">'+r.nom+'</div><div class="ct">'+r.pays+' - '+r.secteur+'</div>';
  }
  if(col==='spark')return sparkSVG(r.sparkline);
  if(col==='prix')return v!==null?'<span style="font-family:monospace">'+(r.devise==='USD'?'$':r.devise==='GBP'?'&pound;':r.devise==='CAD'?'C$':r.devise==='AUD'?'A$':'')+parseFloat(v).toFixed(2)+'</span>':'-';
  if(col==='per')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v<=12?'#10b981':v<=25?'#f59e0b':'#ef4444')+'">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='pbv')return v?'<span style="font-family:monospace;font-size:11px;color:#94a3b8">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='roe')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v>=20?'#10b981':v>=10?'#f59e0b':'#94a3b8')+'">'+v+'%</span>':'-';
  if(col==='marge')return v?'<span style="font-family:monospace;font-size:11px;color:#94a3b8">'+v+'%</span>':'-';
  if(col==='div')return v?'<span style="font-family:monospace;color:'+(v>=4?'#10b981':'#94a3b8')+'">'+parseFloat(v).toFixed(1)+'%</span>':'-';
  if(col==='bpa')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v>=0?'#10b981':'#ef4444')+'">'+v+'%</span>':'-';
  if(col==='ev')return v?'<span style="font-family:monospace;font-size:11px;color:#94a3b8">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='peg')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v<=1?'#10b981':v<=2?'#f59e0b':'#ef4444')+'">'+parseFloat(v).toFixed(2)+'</span>':'-';
  if(col==='ps')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v<=1?'#10b981':v<=2?'#f59e0b':'#94a3b8')+'">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='cr')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v>=2?'#10b981':v>=1?'#f59e0b':'#ef4444')+'">'+parseFloat(v).toFixed(1)+'</span>':'-';
  if(col==='de')return v!==null&&v!==undefined?'<span style="font-family:monospace;font-size:11px;color:'+(v<=30?'#10b981':v<=80?'#f59e0b':'#ef4444')+'">'+parseFloat(v).toFixed(0)+'%</span>':'-';
  if(col==='fcfy')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v>=5?'#10b981':v>=2?'#f59e0b':'#94a3b8')+'">'+parseFloat(v).toFixed(1)+'%</span>':'-';
  if(col==='pfcf')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v<=10?'#10b981':v<=20?'#f59e0b':'#94a3b8')+'">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='graham')return v!==null&&v!==undefined?'<span style="font-family:monospace;font-size:11px;color:'+(v>=20?'#10b981':v>=0?'#f59e0b':'#ef4444')+'">'+(v>=0?'+':'')+v.toFixed(0)+'%</span>':'-';
  if(col==='sv')return bH(lsv(r));
  if(col==='sg')return bH(lsg(r));
  if(col==='sd')return bH(lsd(r));
  if(col==='st')return bH(lst(r));
  if(col==='smom')return bH(lsmom(r));
  if(col==='score')return bH(getScore(r));
  if(col==='divergence'){
    if(!v)return '-';
    var cls=v==='FOND+'?'div-fond':'div-tech';
    var desc=v==='FOND+'?'Fond fort / Tech faible':'Tech fort / Fond faible';
    return '<span class="div-badge '+cls+'" title="'+desc+'">'+v+'</span>';
  }
  if(col==='rsi')return v!==null?'<span style="font-family:monospace;font-size:11px;font-weight:'+(v<=45||v>=70?700:400)+';color:'+(v<=30?'#10b981':v<=45?'#f59e0b':v>=70?'#ef4444':'#94a3b8')+'">'+v+'</span>':'-';
  if(col==='macd')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v>0?'#10b981':v<0?'#ef4444':'#94a3b8')+'">'+parseFloat(v).toFixed(2)+'</span>':'-';
  if(col==='cci')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v<-100?'#10b981':v>150?'#ef4444':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='stoch')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v<=30?'#10b981':v>=70?'#ef4444':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='adx')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v>=25?'#00d4aa':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='willr')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v<=-80?'#10b981':v>=-20?'#ef4444':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='roc')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v>5?'#10b981':v>0?'#f59e0b':'#ef4444')+'">'+(v>0?'+':'')+parseFloat(v).toFixed(1)+'%</span>':'-';
  if(col==='mm20'||col==='mm50'||col==='mm200')return mmC(v);
  if(col==='signaux'){if(!v)return '-';var ps=v.split('|'),h2='',ii;for(ii=0;ii<ps.length;ii++){var s2=ps[ii].trim();if(s2)h2+='<span class="chip">'+s2+'</span>';}return h2||'-';}
  return v!==null&&v!==undefined?String(v):'-';
}

function bH(v){var pct=Math.min(v||0,100),c=pct>=70?'#10b981':pct>=50?'#f59e0b':'#64748b';return '<div class="bw"><div class="br"><div class="bf" style="width:'+pct+'%;background:'+c+'"></div></div><span style="font-family:monospace;font-size:11px;color:'+c+';font-weight:700">'+Math.round(pct)+'</span></div>';}
function mmC(v){if(v===null||v===undefined)return '<span style="color:#2a3a52">-</span>';var c=v>=0?'#10b981':'#ef4444';return '<span style="font-family:monospace;font-size:11px;color:'+c+'">'+(v>=0?'+':'')+v.toFixed(1)+'%</span>';}

function getActiveCols(){
  if(USER_COLS&&USER_COLS[mode])return USER_COLS[mode];
  return ALL_COLS_DEF[mode];
}

function render(){
  var data=doFilter(),i,j,r,c,sc;
  var allN=data.length,peaN=0,divN=0;
  for(i=0;i<data.length;i++){if(data[i].pea)peaN++;if(data[i].divergence)divN++;}
  document.getElementById('b-all').textContent=allN;
  document.getElementById('b-pea').textContent=peaN;
  document.getElementById('b-div').textContent=divN;
  var ie=document.getElementById('b-imp');if(ie&&IMP_DATA)ie.textContent=IMP_DATA.length;

  var s70=0,s50=0,sumD=0,cntD=0,rsiLow=0;
  for(i=0;i<data.length;i++){sc=getScore(data[i]);if(sc>=70)s70++;else if(sc>=50)s50++;if(data[i].div){sumD+=data[i].div;cntD++;}if(data[i].rsi&&data[i].rsi<=45)rsiLow++;}
  var avgD=cntD?(sumD/cntD).toFixed(1)+'%':'-';
  var sd2=[['Total',data.length,'#00d4aa'],['Score 70+',s70,'#10b981'],['Score 50-69',s50,'#f59e0b'],['Div moy',avgD,'#8b5cf6'],['RSI<=45',rsiLow,'#3b82f6']];
  var sh='';for(i=0;i<sd2.length;i++)sh+='<div class="stat"><div class="sl">'+sd2[i][0]+'</div><div class="sv" style="color:'+sd2[i][2]+'">'+sd2[i][1]+'</div></div>';
  document.getElementById('stats').innerHTML=sh;

  var cols=getActiveCols();
  var sortMap={};
  for(i=0;i<sortCols.length;i++)sortMap[sortCols[i].col]=i+1;
  var thH='<tr>';
  for(i=0;i<cols.length;i++){
    c=cols[i];
    var isSorted=sortMap[c]>0;
    var sdir=isSorted?(sortCols[sortMap[c]-1].dir):1;
    thH+='<th class="'+(isSorted?'sorted':'')+'" onclick="doSort(''+c+'')">'+thLabel(c,isSorted,sdir,sortMap[c]||0)+'</th>';
  }
  thH+='</tr>';
  document.getElementById('thead').innerHTML=thH;

  var empty=document.getElementById('empty');
  if(!data.length){document.getElementById('tbody').innerHTML='';document.getElementById('card-list').innerHTML='';empty.style.display='block';return;}
  empty.style.display='none';
  var rows='',cards='';
  for(i=0;i<data.length;i++){
    rows+='<tr>';
    for(j=0;j<cols.length;j++)rows+='<td>'+cH(cols[j],data[i])+'</td>';
    rows+='</tr>';
    cards+=renderCard(data[i]);
  }
  document.getElementById('tbody').innerHTML=rows;
  document.getElementById('card-list').innerHTML=cards;
}

// Mobile cards
function renderCard(r){
  var sc=getScore(r),mom=lsmom(r);
  var c=sc>=70?'#10b981':sc>=50?'#f59e0b':'#64748b';
  var tgs='';if(r.pea)tgs+='<span class="ptag">PEA</span>';if(r._imported)tgs+='<span class="imptag">IMP</span>';
  if(r.divergence){var dc=r.divergence==='FOND+'?'div-fond':'div-tech';tgs+='<span class="div-badge '+dc+'" style="margin-left:3px">'+r.divergence+'</span>';}
  var cmpOn=CMP_SET[r.ticker]?'on':'';
  var kv=function(k,v){return '<div class="card-kv"><div class="card-k">'+k+'</div><div class="card-v" style="color:#94a3b8">'+v+'</div></div>';};
  return '<div class="card">'
    +'<div class="card-top"><div>'
      +'<span class="card-tk" onclick="window.open('https://finance.yahoo.com/quote/'+r.ticker+'','_blank')" style="cursor:pointer">'+r.ticker+'</span>'
      +'<button class="cmp-btn '+cmpOn+'" onclick="toggleCmp(''+r.ticker+'',this)" style="margin-left:6px">&#9878;</button>'
    +'</div>'
    +'<div style="text-align:right"><span class="card-score" style="color:'+c+'">'+sc+'</span>'
      +(mom>=50?'<div style="font-size:9px;color:#3b82f6;font-family:monospace">Mom:'+mom+'</div>':'')
    +'</div></div>'
    +'<div class="card-nom">'+r.nom+'</div>'
    +'<div class="card-tags">'+tgs+'<span class="ptag" style="background:#3b82f618;color:#3b82f6;border-color:#3b82f633">'+r.secteur+'</span></div>'
    +(r.sparkline&&r.sparkline.length?'<div style="margin:6px 0">'+sparkSVG(r.sparkline)+'</div>':'')
    +'<div class="card-grid">'
      +kv('PER',r.per?parseFloat(r.per).toFixed(1)+'x':'-')
      +kv('ROE',r.roe?r.roe+'%':'-')
      +kv('Div',r.div?parseFloat(r.div).toFixed(1)+'%':'-')
      +kv('RSI',r.rsi||'-')
      +kv('MM200',r.mm200!==null&&r.mm200!==undefined?(r.mm200>=0?'+':'')+r.mm200+'%':'-')
      +kv('Graham',r.graham!==null&&r.graham!==undefined?(r.graham>=0?'+':'')+r.graham+'%':'-')
    +'</div>'
    +(r.signaux?'<div class="card-sigs">'+r.signaux+'</div>':'')
  +'</div>';
}

// Export CSV
function exportCSV(){
  var data=doFilter();
  if(!data.length){alert('Aucune donnee a exporter.');return;}
  var cols=['ticker','nom','secteur','pays','pea','devise','prix','per','pbv','roe','marge','div','bpa','ev','peg','ps','pfcf','graham','cr','de','fcfy','sv','sg','sd','st','smom','score','divergence','rsi','macd','cci','stoch','adx','mm20','mm50','mm200','willr','roc','signaux'];
  var lines=[cols.join(';')],i,j,v;
  for(i=0;i<data.length;i++){
    var row=[];
    for(j=0;j<cols.length;j++){
      if(cols[j]==='score')v=getScore(data[i]);
      else if(cols[j]==='smom')v=lsmom(data[i]);
      else v=data[i][cols[j]]!==null&&data[i][cols[j]]!==undefined?data[i][cols[j]]:'';
      row.push(String(v).replace(/;/g,','));
    }
    lines.push(row.join(';'));
  }
  var blob=new Blob(['\uFEFF'+lines.join('\r\n')],{type:'text/csv;charset=utf-8'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);
  var now=new Date();a.download='screener_'+now.getFullYear()+(now.getMonth()+1<10?'0':'')+(now.getMonth()+1)+now.getDate()+'.csv';a.click();
}

// Comparateur
function toggleCmp(ticker,btn){
  if(CMP_SET[ticker]){delete CMP_SET[ticker];btn.className='cmp-btn';}
  else{
    if(Object.keys(CMP_SET).length>=4){alert('Maximum 4 actions.');return;}
    var found=RAW_ORIG.filter(function(r){return r.ticker===ticker;});
    if(found.length)CMP_SET[ticker]=found[0];
    btn.className='cmp-btn on';
  }
  var n=Object.keys(CMP_SET).length;
  var ob=document.getElementById('cmp-open-btn');
  if(ob){ob.style.display=n>0?'':'none';document.getElementById('cmp-count').textContent=n;}
}
function openComparator(){
  var tickers=Object.keys(CMP_SET);
  if(!tickers.length){alert('Selectionnez des actions avec le bouton de comparaison.');return;}
  var ROWS=[
    {k:'Score',f:function(r){return getScore(r);}},
    {k:'Momentum',f:function(r){return lsmom(r);}},
    {k:'Divergence',f:function(r){return r.divergence||'-';}},
    {k:'Cours',f:function(r){return r.prix?parseFloat(r.prix).toFixed(2)+' '+r.devise:'-';}},
    {k:'PER',f:function(r){return r.per?parseFloat(r.per).toFixed(1)+'x':'-';}},
    {k:'P/Book',f:function(r){return r.pbv?parseFloat(r.pbv).toFixed(1)+'x':'-';}},
    {k:'EV/EBITDA',f:function(r){return r.ev?parseFloat(r.ev).toFixed(1)+'x':'-';}},
    {k:'P/S',f:function(r){return r.ps?parseFloat(r.ps).toFixed(1)+'x':'-';}},
    {k:'P/FCF',f:function(r){return r.pfcf?parseFloat(r.pfcf).toFixed(1)+'x':'-';}},
    {k:'Graham %',f:function(r){return r.graham!==null&&r.graham!==undefined?(r.graham>=0?'+':'')+r.graham.toFixed(0)+'%':'-';}},
    {k:'ROE %',f:function(r){return r.roe?r.roe+'%':'-';}},
    {k:'BPA Croiss.',f:function(r){return r.bpa?r.bpa+'%':'-';}},
    {k:'PEG',f:function(r){return r.peg?parseFloat(r.peg).toFixed(2):'-';}},
    {k:'Dividende',f:function(r){return r.div?parseFloat(r.div).toFixed(1)+'%':'-';}},
    {k:'Current Ratio',f:function(r){return r.cr?parseFloat(r.cr).toFixed(1):'-';}},
    {k:'D/E %',f:function(r){return r.de!==null&&r.de!==undefined?r.de.toFixed(0)+'%':'-';}},
    {k:'FCF Yield',f:function(r){return r.fcfy?parseFloat(r.fcfy).toFixed(1)+'%':'-';}},
    {k:'RSI',f:function(r){return r.rsi||'-';}},
    {k:'MACD',f:function(r){return r.macd!==null?parseFloat(r.macd).toFixed(2):'-';}},
    {k:'CCI',f:function(r){return r.cci!==null?Math.round(r.cci):'-';}},
    {k:'ADX',f:function(r){return r.adx!==null?Math.round(r.adx):'-';}},
    {k:'MM200 %',f:function(r){return r.mm200!==null&&r.mm200!==undefined?(r.mm200>=0?'+':'')+r.mm200.toFixed(1)+'%':'-';}},
    {k:'Williams %R',f:function(r){return r.willr!==null?Math.round(r.willr):'-';}},
    {k:'RoC %',f:function(r){return r.roc!==null?(r.roc>=0?'+':'')+parseFloat(r.roc).toFixed(1)+'%':'-';}},
    {k:'Score Value',f:function(r){return lsv(r);}},
    {k:'Score Growth',f:function(r){return lsg(r);}},
    {k:'Score Div',f:function(r){return lsd(r);}},
    {k:'Score Tech',f:function(r){return lst(r);}},
  ];
  var h='<div class="cmp-header">';
  for(var t=0;t<tickers.length;t++){var ri=CMP_SET[tickers[t]];h+='<div class="cmp-hcol"><div class="cmp-htk">'+ri.ticker+'</div><div class="cmp-hnom">'+ri.nom+'</div>'+sparkSVG(ri.sparkline)+'</div>';}
  h+='</div>';
  var body='';
  for(var ri2=0;ri2<ROWS.length;ri2++){
    var vals=[],maxV=-Infinity,minV=Infinity,nums=[];
    for(var ti=0;ti<tickers.length;ti++){vals.push(ROWS[ri2].f(CMP_SET[tickers[ti]]));}
    for(var vi=0;vi<vals.length;vi++){var n=parseFloat(vals[vi]);if(!isNaN(n)){nums.push(n);if(n>maxV)maxV=n;if(n<minV)minV=n;}}
    body+='<div class="cmp-row"><div class="cmp-lbl">'+ROWS[ri2].k+'</div><div class="cmp-vals">';
    for(var vi2=0;vi2<vals.length;vi2++){
      var n2=parseFloat(vals[vi2]),col2='#94a3b8';
      if(!isNaN(n2)&&nums.length>1)col2=n2===maxV?'#10b981':n2===minV?'#ef4444':'#94a3b8';
      body+='<div class="cmp-val" style="color:'+col2+'">'+vals[vi2]+'</div>';
    }
    body+='</div></div>';
  }
  document.getElementById('cmp-content').innerHTML=h+body;
  document.getElementById('modal-cmp').className='modal-bg open';
}
function closeComparator(){document.getElementById('modal-cmp').className='modal-bg';}

// Column picker
function openCols(){
  var cols=ALL_COLS_DEF[mode];
  var active=getActiveCols();
  var grid=document.getElementById('col-grid');grid.innerHTML='';
  var allKeys=Object.keys(CL);
  for(var i=0;i<allKeys.length;i++){
    var col=allKeys[i];
    var checked=active.indexOf(col)!==-1;
    var item=document.createElement('div');item.className='col-item'+(checked?' pinned':'');
    item.innerHTML='<input type="checkbox" id="chk-'+col+'" '+(checked?'checked':'')+'><label for="chk-'+col+'">'+(CL[col]||col)+'</label>';
    grid.appendChild(item);
  }
  document.getElementById('modal-cols').className='modal-bg open';
}
function closeCols(){document.getElementById('modal-cols').className='modal-bg';}
function applyCols(){
  var allKeys=Object.keys(CL),newCols=[];
  for(var i=0;i<allKeys.length;i++){var el=document.getElementById('chk-'+allKeys[i]);if(el&&el.checked)newCols.push(allKeys[i]);}
  if(!newCols.length){alert('Selectionnez au moins une colonne.');return;}
  if(!USER_COLS)USER_COLS={};
  USER_COLS[mode]=newCols;
  savePrefs();closeCols();render();
}
function resetCols(){
  if(!USER_COLS)USER_COLS={};
  delete USER_COLS[mode];
  savePrefs();closeCols();render();
}

// Settings modal
function openCfg(){
  document.getElementById('c-per').value=CFG.perMax;document.getElementById('c-pbv').value=CFG.pbvMax;
  document.getElementById('c-ev').value=15;document.getElementById('c-ps').value=CFG.psMax;
  document.getElementById('c-pfcf').value=CFG.pfcfMax;document.getElementById('c-graham').value=CFG.grahamMin;
  document.getElementById('c-roe').value=CFG.roeMin;document.getElementById('c-cr').value=CFG.crMin;
  document.getElementById('c-de').value=CFG.deMax;document.getElementById('c-fcfy').value=CFG.fcfyMin;
  document.getElementById('c-rsi').value=CFG.rsiSurvente;document.getElementById('c-stoch').value=CFG.stochSurvente;
  document.getElementById('c-cci').value=CFG.cciSurvente;document.getElementById('c-adx').value=CFG.adxMin;
  document.getElementById('c-willr').value=CFG.willrSurvente;document.getElementById('c-roc').value=CFG.rocMin;
  document.getElementById('c-pv').value=CFG.poidsValue;document.getElementById('c-pg').value=CFG.poidsGrowth;
  document.getElementById('c-pd').value=CFG.poidsDividende;document.getElementById('c-pt').value=CFG.poidsTechnique;
  document.getElementById('c-smin').value=CFG.scoreMin;
  var togs={'per':CFG.tPer,'pbv':CFG.tPbv,'ev':CFG.tEv,'ps':CFG.tPs,'pfcf':CFG.tPfcf,'graham':CFG.tGraham,
    'roe':CFG.tRoe,'bpa':CFG.tBpa,'peg':CFG.tPeg,'cr':CFG.tCr,'de':CFG.tDe,'fcfy':CFG.tFcfy,
    'rsi':CFG.tRsi,'macd':CFG.tMacd,'cci':CFG.tCci,'stoch':CFG.tStoch,'mm':CFG.tMm,'adx':CFG.tAdx,'willr':CFG.tWillr,'roc':CFG.tRoc};
  for(var k in togs){var el=document.getElementById('t-'+k);if(el)el.checked=!!togs[k];}
  document.getElementById('modal-cfg').className='modal-bg open';
}
function closeCfg(){document.getElementById('modal-cfg').className='modal-bg';}
function applyCfg(){
  var pv=parseFloat(document.getElementById('c-pv').value)||0,pg=parseFloat(document.getElementById('c-pg').value)||0,
      pd=parseFloat(document.getElementById('c-pd').value)||0,pt=parseFloat(document.getElementById('c-pt').value)||0;
  if(pv+pg+pd+pt!==100){alert('Les 4 poids doivent totaliser 100. Total: '+(pv+pg+pd+pt));return;}
  CFG.perMax=parseFloat(document.getElementById('c-per').value)||CFG.perMax;
  CFG.pbvMax=parseFloat(document.getElementById('c-pbv').value)||CFG.pbvMax;
  CFG.psMax=parseFloat(document.getElementById('c-ps').value)||CFG.psMax;
  CFG.pfcfMax=parseFloat(document.getElementById('c-pfcf').value)||CFG.pfcfMax;
  CFG.grahamMin=parseFloat(document.getElementById('c-graham').value)||0;
  CFG.roeMin=parseFloat(document.getElementById('c-roe').value)||CFG.roeMin;
  CFG.crMin=parseFloat(document.getElementById('c-cr').value)||CFG.crMin;
  CFG.deMax=parseFloat(document.getElementById('c-de').value)||CFG.deMax;
  CFG.fcfyMin=parseFloat(document.getElementById('c-fcfy').value)||CFG.fcfyMin;
  CFG.rsiSurvente=parseFloat(document.getElementById('c-rsi').value)||CFG.rsiSurvente;
  CFG.stochSurvente=parseFloat(document.getElementById('c-stoch').value)||CFG.stochSurvente;
  CFG.cciSurvente=parseFloat(document.getElementById('c-cci').value)||CFG.cciSurvente;
  CFG.adxMin=parseFloat(document.getElementById('c-adx').value)||CFG.adxMin;
  CFG.willrSurvente=parseFloat(document.getElementById('c-willr').value)||CFG.willrSurvente;
  CFG.rocMin=parseFloat(document.getElementById('c-roc').value)||CFG.rocMin;
  CFG.poidsValue=pv;CFG.poidsGrowth=pg;CFG.poidsDividende=pd;CFG.poidsTechnique=pt;
  CFG.scoreMin=parseFloat(document.getElementById('c-smin').value)||CFG.scoreMin;
  document.getElementById('ms').value=CFG.scoreMin;
  var tkeys=['per','pbv','ev','ps','pfcf','graham','roe','bpa','peg','cr','de','fcfy','rsi','macd','cci','stoch','mm','adx','willr','roc'];
  var tmap={'per':'tPer','pbv':'tPbv','ev':'tEv','ps':'tPs','pfcf':'tPfcf','graham':'tGraham','roe':'tRoe','bpa':'tBpa','peg':'tPeg','cr':'tCr','de':'tDe','fcfy':'tFcfy','rsi':'tRsi','macd':'tMacd','cci':'tCci','stoch':'tStoch','mm':'tMm','adx':'tAdx','willr':'tWillr','roc':'tRoc'};
  for(var i=0;i<tkeys.length;i++){var el2=document.getElementById('t-'+tkeys[i]);if(el2)CFG[tmap[tkeys[i]]]=el2.checked?1:0;}
  savePrefs();closeCfg();render();
}
function resetPrefs(){
  try{localStorage.removeItem(LS_KEY);}catch(e){}
  location.reload();
}

function openImp(){document.getElementById('modal-imp').className='modal-bg open';}
function closeImp(){document.getElementById('modal-imp').className='modal-bg';}
function importTickers(){
  var txt=document.getElementById('imp-text').value;
  var tks=txt.split(/[\n,;\s]+/).map(function(t){return t.trim().toUpperCase();}).filter(function(t){return t.length>0;});
  if(!tks.length){alert('Aucun ticker valide');return;}
  var found=[],notFound=[],i,j,hit;
  for(i=0;i<tks.length;i++){hit=null;for(j=0;j<RAW_ORIG.length;j++){if(RAW_ORIG[j].ticker.toUpperCase()===tks[i]){hit=RAW_ORIG[j];break;}}if(hit){var rr=JSON.parse(JSON.stringify(hit));rr._imported=true;found.push(rr);}else notFound.push(tks[i]);}
  IMP_DATA=found;
  var it=document.getElementById('tab-imp');if(it)it.style.display='';
  closeImp();setTab('imp');
  if(notFound.length)alert(found.length+' ticker(s) trouves.\n'+notFound.length+' absents: '+notFound.join(', '));
}

function triggerScan(){
  if(!GH_TOKEN||GH_TOKEN==='VOTRE_GH_TOKEN'){alert('Token GitHub non configure.');return;}
  var url='https://api.github.com/repos/'+GH_OWNER+'/'+GH_REPO+'/actions/workflows/scan.yml/dispatches';
  fetch(url,{method:'POST',headers:{'Authorization':'token '+GH_TOKEN,'Accept':'application/vnd.github.v3+json','Content-Type':'application/json'},body:JSON.stringify({ref:'main'})})
  .then(function(r){
    if(r.status===204||r.ok){
      document.getElementById('modal-scan').className='modal-bg open';
      var secs=3600,el=document.getElementById('scan-countdown');
      var iv=setInterval(function(){secs--;var m=Math.floor(secs/60),s=secs%60;if(el)el.textContent='Rechargement dans '+m+'min '+(s<10?'0':'')+s+'s';if(secs<=0){clearInterval(iv);location.reload();}},1000);
    }else r.text().then(function(t){alert('Erreur GitHub: '+r.status+'\n'+t);});
  }).catch(function(e){alert('Erreur: '+e.message);});
}

// LOGIN SHA-256 gere dans le script isole ci-dessus
var PWD_HASH='__PWD_HASH__'; // reference uniquement

render();
"""

page.append('<script>')
page.append('var RAW_ORIG=' + DATA + ';')
page.append('var CFG=' + CFG + ';')
page.append('var IMP_DATA=null;')
page.append('var SCAN_DATE="' + DATE + '";')
page.append('var GH_OWNER="VOTRE_USERNAME";')
page.append('var GH_REPO="screener";')
page.append('var GH_TOKEN="VOTRE_GH_TOKEN";')

page.append(js)
page.append('</script></body></html>')

out = Path('docs/index.html')
out.parent.mkdir(exist_ok=True)

# ── Injection du hash SHA-256 du mot de passe ──
import hashlib, os
raw_pwd = os.environ.get('SCREENER_PASSWORD', 'changeme2025')
pwd_hash = hashlib.sha256(raw_pwd.encode('utf-8')).hexdigest()
html_content = '\n'.join(page).replace('__PWD_HASH__', pwd_hash)
out.write_text(html_content, encoding='utf-8')
print('HTML OK: ' + str(out))
