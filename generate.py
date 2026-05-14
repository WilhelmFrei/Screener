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
    b = s.rolling(p).std()
    up = m + n * b
    lo = m - n * b
    return (s - lo) / (up - lo).replace(0, np.nan)

# ══════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════
def score_value(info):
    p = 0
    per = info.get('trailingPE') or info.get('forwardPE')
    pbv = info.get('priceToBook')
    ev  = info.get('enterpriseToEbitda')
    if per and per > 0: p += 40 if per <= 10 else 28 if per <= 15 else 15 if per <= PER_MAX else 0
    if pbv and pbv > 0: p += 30 if pbv <= 1  else 20 if pbv <= 2  else 10 if pbv <= PBV_MAX else 0
    if ev  and ev  > 0: p += 30 if ev  <= 8  else 20 if ev  <= 12 else 10 if ev  <= 15 else 0
    return min(100, p)

def score_growth(info):
    p = 0
    roe = info.get('returnOnEquity')
    eg  = info.get('earningsGrowth')
    peg = info.get('pegRatio')
    if roe: rp = roe * 100; p += 35 if rp >= 25 else 25 if rp >= 15 else 12 if rp >= ROE_MIN else 0
    if eg:  ep = eg  * 100; p += 40 if ep >= 25 else 28 if ep >= 15 else 15 if ep > 0 else 0
    if peg and peg > 0: p += 25 if peg <= 1 else 15 if peg <= 1.5 else 8 if peg <= 2.5 else 0
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
        det = {
            'rsi':   round(float(r), 1)    if not np.isnan(r)    else None,
            'macd':  round(float(mh_v), 3) if not np.isnan(mh_v) else None,
            'cci':   round(float(cc), 1)   if not np.isnan(cc)   else None,
            'stoch': round(float(st), 1)   if not np.isnan(st)   else None,
            'adx':   round(float(ax), 1)   if not np.isnan(ax)   else None,
            'bb':    round(float(bb), 2)   if not np.isnan(bb)   else None,
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
        # Pause anti rate-limit toutes les 50 actions
        if idx > 0 and idx % 50 == 0:
            time.sleep(2)
        # Utiliser fast_info d'abord (plus léger, moins de rate limit)
        tk_obj = yf.Ticker(ticker)
        try:
            fi = tk_obj.fast_info
            px_fast = getattr(fi, 'lastPrice', None) or getattr(fi, 'last_price', None)
            dev_fast = getattr(fi, 'currency', None)
        except Exception:
            fi = None; px_fast = None; dev_fast = None
        # Info complète avec retry
        info = {}
        for attempt in range(2):
            try:
                info = tk_obj.info or {}
                if info: break
            except Exception:
                if attempt == 0: time.sleep(3)
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
        sv = score_value(info)
        sg = score_growth(info)
        sd = score_div(info)
        st, td = score_tech(hist)
        total_f = POIDS_VALUE + POIDS_GROWTH + POIDS_DIVIDENDE
        sf = round(sv * POIDS_VALUE / total_f + sg * POIDS_GROWTH / total_f + sd * POIDS_DIVIDENDE / total_f)
        score = round(sf * (1 - POIDS_TECHNIQUE / 100) + st * (POIDS_TECHNIQUE / 100))
        if score < SCORE_MIN:
            continue
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
            'sv': sv, 'sg': sg, 'sd': sd, 'st': st, 'score': score,
            'rsi':   td.get('rsi'),   'macd':  td.get('macd'),
            'cci':   td.get('cci'),   'stoch': td.get('stoch'),
            'adx':   td.get('adx'),   'bb':    td.get('bb'),
            'mm20':  td.get('mm20'),  'mm50':  td.get('mm50'), 'mm200': td.get('mm200'),
            'signaux': td.get('signaux', ''),
        }
        resultats.append(row)
    except Exception:
        pass

resultats.sort(key=lambda x: x['score'], reverse=True)
resultats = resultats[:TOP_N]
print('OK: ' + str(len(resultats)) + ' opportunites sur ' + str(len(tickers)) + ' actions')

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
page.append('</style></head><body>')

page.append('<div class="hdr">')
page.append('<div><div class="htitle">&#128225; Stock Screener</div>')
page.append('<div class="hsub">' + NSCAN + ' actions scannees - Yahoo Finance - ' + DATE + '</div></div>')
page.append('<div class="hbtns">')
page.append('<span style="font-family:monospace;font-size:10px;color:#64748b" id="nxt"></span>')
page.append('<button class="hbtn hbtn-cfg" onclick="openCfg()">&#9881; Criteres</button>')
page.append('<button class="hbtn hbtn-imp" onclick="openImp()">&#128229; Tickers</button>')
page.append('<button class="hbtn hbtn-scan" onclick="triggerScan()">&#9889; Nouveau scan</button>')
page.append('</div></div>')

page.append('<div class="modal-bg" id="modal-cfg"><div class="modal">')
page.append('<button class="modal-close" onclick="closeCfg()">&#10005;</button>')
page.append('<h3>&#9881; Criteres de scoring</h3>')
page.append('<div class="cfg-note">Modifications instantanees sur les donnees en memoire. Nouveau scan pour recharger Yahoo Finance.</div>')
page.append('<div class="cfg-section">Valorisation</div><div class="cfg-grid">')
page.append('<div class="cfg-item"><label>PER max</label><input type="number" id="c-per" step="1"></div>')
page.append('<div class="cfg-item"><label>P/Book max</label><input type="number" id="c-pbv" step="0.5"></div>')
page.append('<div class="cfg-item"><label>EV/EBITDA max</label><input type="number" id="c-ev" step="1"></div>')
page.append('<div class="cfg-item"><label>ROE min (%)</label><input type="number" id="c-roe" step="1"></div>')
page.append('</div><div class="cfg-section">Technique</div><div class="cfg-grid">')
page.append('<div class="cfg-item"><label>RSI survente max</label><input type="number" id="c-rsi" step="1"></div>')
page.append('<div class="cfg-item"><label>Stoch. survente max</label><input type="number" id="c-stoch" step="1"></div>')
page.append('<div class="cfg-item"><label>CCI survente (negatif)</label><input type="number" id="c-cci" step="10"></div>')
page.append('<div class="cfg-item"><label>ADX min</label><input type="number" id="c-adx" step="1"></div>')
page.append('</div><div class="cfg-section">Poids du score (total = 100)</div><div class="cfg-grid">')
page.append('<div class="cfg-item"><label>Poids Value (%)</label><input type="number" id="c-pv" step="5"></div>')
page.append('<div class="cfg-item"><label>Poids Growth (%)</label><input type="number" id="c-pg" step="5"></div>')
page.append('<div class="cfg-item"><label>Poids Dividende (%)</label><input type="number" id="c-pd" step="5"></div>')
page.append('<div class="cfg-item"><label>Poids Technique (%)</label><input type="number" id="c-pt" step="5"></div>')
page.append('</div><div class="cfg-section">Filtrage</div><div class="cfg-grid">')
page.append('<div class="cfg-item"><label>Score minimum /100</label><input type="number" id="c-smin" step="5"></div>')
page.append('</div>')
page.append('<button class="modal-btn modal-btn-ok" onclick="applyCfg()">&#10003; Appliquer</button>')
page.append('<button class="modal-btn modal-btn-cancel" onclick="closeCfg()">Annuler</button>')
page.append('</div></div>')

page.append('<div class="modal-bg" id="modal-imp"><div class="modal">')
page.append('<button class="modal-close" onclick="closeImp()">&#10005;</button>')
page.append('<h3>&#128229; Importer des tickers</h3>')
page.append('<div class="imp-note">Un ticker par ligne ou separes par virgule.<br>Ex: AAPL, MC.PA, AIR.PA<br>Filtre les donnees du dernier scan. Au prochain scan automatique la liste reprend normalement.</div>')
page.append('<textarea class="imp-area" id="imp-text" placeholder="AAPL&#10;MSFT&#10;MC.PA"></textarea>')
page.append('<button class="modal-btn modal-btn-ok" onclick="importTickers()">&#10003; Appliquer</button>')
page.append('<button class="modal-btn modal-btn-cancel" onclick="closeImp()" style="margin-top:8px">Annuler</button>')
page.append('</div></div>')

page.append('<div class="modal-bg" id="modal-scan"><div class="modal" style="text-align:center">')
page.append('<h3>&#9889; Scan lance sur GitHub Actions</h3>')
page.append('<p style="color:#64748b;font-size:12px;margin:12px 0 16px;line-height:1.7">' + NSCAN + ' actions a analyser.<br>Duree estimee: 30-60 min.<br>La page se rechargera automatiquement.</p>')
page.append('<div style="font-family:monospace;font-size:13px;color:#00d4aa;background:#0d1520;padding:12px;border-radius:8px;margin-bottom:14px" id="scan-countdown"></div>')
page.append('<button class="modal-btn modal-btn-cancel" onclick="document.getElementById(\'modal-scan\').className=\'modal-bg\'">Fermer</button>')
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
page.append('<div class="tab" id="tab-imp" onclick="setTab(\'imp\')" style="display:none">Import <span class="bdg" id="b-imp">0</span></div>')
page.append('</div>')
page.append('<div class="stats" id="stats"></div>')
page.append('<div class="tw"><div class="ts"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table>')
page.append('<div class="empty" id="empty" style="display:none">Aucun resultat</div></div></div>')
page.append('<div class="disc">Donnees Yahoo Finance a titre indicatif. Ne constitue pas un conseil en investissement.</div>')
page.append('</div>')

page.append('<script>')
page.append('var RAW_ORIG=' + DATA + ';')
page.append('var CFG=' + CFG + ';')
page.append('var IMP_DATA=null;')
page.append('var GH_OWNER="VOTRE_USERNAME";')
page.append('var GH_REPO="screener";')
page.append('var GH_TOKEN="VOTRE_GH_TOKEN";')

js = r"""
var mode="combine",curTab="all",sortCol="score",sortDir=-1;

function buildSects(){
  var s=['all'],i,o;
  for(i=0;i<RAW_ORIG.length;i++){if(RAW_ORIG[i].secteur&&s.indexOf(RAW_ORIG[i].secteur)===-1)s.push(RAW_ORIG[i].secteur);}
  s.sort();
  var se=document.getElementById('sect'); se.innerHTML='';
  for(i=0;i<s.length;i++){o=document.createElement('option');o.value=s[i];o.text=s[i]==='all'?'Tous secteurs':s[i];se.appendChild(o);}
}
buildSects();

function nextUpdate(){
  var now=new Date(),nxt=new Date();
  nxt.setDate(now.getDate()+(now.getHours()>=7?1:0));nxt.setHours(7,0,0,0);
  var diff=nxt-now,h=Math.floor(diff/3600000),m=Math.floor((diff%3600000)/60000);
  var el=document.getElementById('nxt');if(el)el.textContent='Scan auto: '+h+'h'+m+'m';
}
nextUpdate();setInterval(nextUpdate,60000);

var DESCS={combine:'Mode Combine: fondamentaux + technique. Recommande.',fond:'Mode Fondamental: PER, ROE, dividende, croissance.',tech:'Mode Technique: RSI, MACD, CCI, Stochastique, MM.'};

function setMode(m){
  mode=m;
  ['combine','fond','tech'].forEach(function(x){document.getElementById('m-'+x).className='mbtn'+(x===m?' on':'');});
  document.getElementById('mdesc').textContent=DESCS[m];render();
}
function setTab(t){
  curTab=t;
  ['all','pea','imp'].forEach(function(x){var el=document.getElementById('tab-'+x);if(el)el.className='tab'+(x===t?' on':'');});
  render();
}

function lsv(r){var p=0;if(r.per)p+=r.per<=10?40:r.per<=15?28:r.per<=CFG.perMax?15:0;if(r.pbv)p+=r.pbv<=1?30:r.pbv<=2?20:r.pbv<=CFG.pbvMax?10:0;if(r.ev)p+=r.ev<=8?30:r.ev<=12?20:r.ev<=15?10:0;return Math.min(100,p);}
function lsg(r){var p=0;if(r.roe)p+=r.roe>=25?35:r.roe>=15?25:r.roe>=CFG.roeMin?12:0;if(r.bpa)p+=r.bpa>=25?40:r.bpa>=15?28:r.bpa>0?15:0;if(r.peg)p+=r.peg<=1?25:r.peg<=1.5?15:r.peg<=2.5?8:0;return Math.min(100,p);}
function lsd(r){var p=0;if(r.div)p+=r.div>=6?60:r.div>=4?45:r.div>=3?30:r.div>0?15:0;return Math.min(100,p);}
function lst(r){
  var p=0;
  if(r.rsi!==null&&r.rsi!==undefined){if(r.rsi<=30)p+=28;else if(r.rsi<=CFG.rsiSurvente)p+=18;else if(r.rsi<=50)p+=8;}
  if(r.macd!==null&&r.macd!==undefined){if(r.macd>0)p+=15;else if(r.macd>-0.1)p+=5;}
  if(r.cci!==null&&r.cci!==undefined){if(r.cci<CFG.cciSurvente)p+=18;else if(r.cci>=50&&r.cci<=150)p+=12;else if(r.cci>0)p+=6;}
  if(r.stoch!==null&&r.stoch!==undefined){if(r.stoch<=20)p+=14;else if(r.stoch<=CFG.stochSurvente)p+=9;}
  if(r.mm20!==null&&r.mm50!==null&&r.mm200!==null&&r.mm20>0&&r.mm50>0&&r.mm200>0)p+=15;
  if(r.adx!==null&&r.adx>=CFG.adxMin)p+=5;
  return Math.min(100,p);
}
function ls(r){var sv=lsv(r),sg=lsg(r),sd=lsd(r),st=lst(r),tf=CFG.poidsValue+CFG.poidsGrowth+CFG.poidsDividende;return Math.round(Math.round(sv*CFG.poidsValue/tf+sg*CFG.poidsGrowth/tf+sd*CFG.poidsDividende/tf)*(1-CFG.poidsTechnique/100)+st*(CFG.poidsTechnique/100));}
function getScore(r){if(mode==='fond')return Math.round(lsv(r)*0.4+lsg(r)*0.35+lsd(r)*0.25);if(mode==='tech')return lst(r);return ls(r);}

function doFilter(){
  var q=(document.getElementById('q').value||'').toLowerCase();
  var mkt=document.getElementById('mkt').value;
  var sec=document.getElementById('sect').value;
  var ms=parseFloat(document.getElementById('ms').value)||0;
  var po=document.getElementById('peaOnly').value==='1';
  var src=(curTab==='imp'&&IMP_DATA)?IMP_DATA:RAW_ORIG;
  var out=[],i,r;
  for(i=0;i<src.length;i++){
    r=src[i];
    if(getScore(r)<ms)continue;
    if(po&&!r.pea)continue;
    if(curTab==='pea'&&!r.pea)continue;
    if(mkt==='eu'&&(r.pays==='US'||r.pays==='CA'||r.pays==='AU'))continue;
    if(mkt==='us'&&r.pays!=='US')continue;
    if(mkt==='ca'&&r.pays!=='CA')continue;
    if(mkt==='other'&&(r.pays==='US'||r.pays==='CA'||r.pays==='FR'||r.pays==='DE'||r.pays==='GB'||r.pays==='NL'||r.pays==='ES'||r.pays==='IT'||r.pays==='CH'))continue;
    if(sec!=='all'&&r.secteur!==sec)continue;
    if(q&&r.ticker.toLowerCase().indexOf(q)===-1&&r.nom.toLowerCase().indexOf(q)===-1)continue;
    out.push(r);
  }
  out.sort(function(a,b){
    if(sortCol==='score')return sortDir*(getScore(b)-getScore(a));
    var av=a[sortCol],bv=b[sortCol];
    if(av===null||av===undefined)return 1;if(bv===null||bv===undefined)return -1;
    var na=parseFloat(av),nb=parseFloat(bv);
    if(!isNaN(na)&&!isNaN(nb))return sortDir*(na-nb);
    return sortDir*String(av).localeCompare(String(bv));
  });
  return out;
}

function bH(v){var pct=Math.min(v||0,100),c=pct>=70?'#10b981':pct>=50?'#f59e0b':'#64748b';return '<div class="bw"><div class="br"><div class="bf" style="width:'+pct+'%;background:'+c+'"></div></div><span style="font-family:monospace;font-size:11px;color:'+c+';font-weight:700">'+Math.round(pct)+'</span></div>';}
function mmC(v){if(v===null||v===undefined)return '<span style="color:#2a3a52">-</span>';var c=v>=0?'#10b981':'#ef4444';return '<span style="font-family:monospace;font-size:11px;color:'+c+'">'+(v>=0?'+':'')+v.toFixed(1)+'%</span>';}

function cH(col,r){
  var v=r[col];
  if(col==='ticker'){var tg='';if(r.pea)tg+='<span class="ptag">PEA</span>';if(r._imported)tg+='<span class="imptag">IMP</span>';return '<div><span class="tk">'+r.ticker+'</span>'+tg+'</div><div class="nm">'+r.nom+'</div><div class="ct">'+r.pays+' - '+r.secteur+'</div>';}
  if(col==='prix')return v!==null?'<span style="font-family:monospace">'+(r.devise==='USD'?'$':r.devise==='GBP'?'&pound;':r.devise==='CAD'?'C$':r.devise==='AUD'?'A$':'')+parseFloat(v).toFixed(2)+'</span>':'-';
  if(col==='per')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v<=12?'#10b981':v<=25?'#f59e0b':'#ef4444')+'">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='pbv')return v?'<span style="font-family:monospace;font-size:11px;color:#94a3b8">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='roe')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v>=20?'#10b981':v>=10?'#f59e0b':'#94a3b8')+'">'+v+'%</span>':'-';
  if(col==='marge')return v?'<span style="font-family:monospace;font-size:11px;color:#94a3b8">'+v+'%</span>':'-';
  if(col==='div')return v?'<span style="font-family:monospace;color:'+(v>=4?'#10b981':'#94a3b8')+'">'+parseFloat(v).toFixed(1)+'%</span>':'-';
  if(col==='bpa')return v?'<span style="font-family:monospace;font-size:11px;color:'+(v>=0?'#10b981':'#ef4444')+'">'+v+'%</span>':'-';
  if(col==='ev')return v?'<span style="font-family:monospace;font-size:11px;color:#94a3b8">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='sv'||col==='sg'||col==='sd'||col==='st')return v!==null?bH(v):'-';
  if(col==='score')return bH(getScore(r));
  if(col==='rsi')return v!==null?'<span style="font-family:monospace;font-size:11px;font-weight:'+(v<=45||v>=70?700:400)+';color:'+(v<=30?'#10b981':v<=45?'#f59e0b':v>=70?'#ef4444':'#94a3b8')+'">'+v+'</span>':'-';
  if(col==='macd')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v>0?'#10b981':v<0?'#ef4444':'#94a3b8')+'">'+parseFloat(v).toFixed(2)+'</span>':'-';
  if(col==='cci')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v<-100?'#10b981':v>150?'#ef4444':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='stoch')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v<=30?'#10b981':v>=70?'#ef4444':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='adx')return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v>=25?'#00d4aa':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='mm20'||col==='mm50'||col==='mm200')return mmC(v);
  if(col==='signaux'){if(!v)return '-';var ps=v.split('|'),h='',i;for(i=0;i<ps.length;i++){var s=ps[i].trim();if(s)h+='<span class="chip">'+s+'</span>';}return h||'-';}
  return v!==null&&v!==undefined?String(v):'-';
}

var CF=['ticker','prix','per','pbv','roe','marge','div','bpa','ev','sv','sg','sd','score'];
var CT=['ticker','prix','rsi','macd','cci','stoch','adx','mm20','mm50','mm200','st','signaux'];
var CC=['ticker','prix','per','roe','div','rsi','macd','cci','mm200','sv','st','score','signaux'];
var CL={ticker:'Action',prix:'Cours',per:'PER',pbv:'P/Book',roe:'ROE%',marge:'Marge%',div:'Div%',bpa:'BPA%',ev:'EV/EBITDA',sv:'Score V',sg:'Score G',sd:'Score D',st:'Score T',score:'Score',rsi:'RSI',macd:'MACD',cci:'CCI',stoch:'Stoch',adx:'ADX',mm20:'MM20%',mm50:'MM50%',mm200:'MM200%',signaux:'Signaux'};

function doSort(col){if(sortCol===col)sortDir*=-1;else{sortCol=col;sortDir=-1;}render();}

function render(){
  var data=doFilter(),i,j,r,c,sc;
  var allN=data.length,peaN=0;
  for(i=0;i<data.length;i++){if(data[i].pea)peaN++;}
  document.getElementById('b-all').textContent=allN;
  document.getElementById('b-pea').textContent=peaN;
  var ie=document.getElementById('b-imp');if(ie&&IMP_DATA)ie.textContent=IMP_DATA.length;

  var s70=0,s50=0,sumD=0,cntD=0,rsiLow=0;
  for(i=0;i<data.length;i++){sc=getScore(data[i]);if(sc>=70)s70++;else if(sc>=50)s50++;if(data[i].div){sumD+=data[i].div;cntD++;}if(data[i].rsi&&data[i].rsi<=45)rsiLow++;}
  var avgD=cntD?(sumD/cntD).toFixed(1)+'%':'-';
  var sd2=[['Total',data.length,'#00d4aa'],['Score 70+',s70,'#10b981'],['Score 50-69',s50,'#f59e0b'],['Div moy',avgD,'#8b5cf6'],['RSI<=45',rsiLow,'#3b82f6']];
  var sh='';for(i=0;i<sd2.length;i++)sh+='<div class="stat"><div class="sl">'+sd2[i][0]+'</div><div class="sv" style="color:'+sd2[i][2]+'">'+sd2[i][1]+'</div></div>';
  document.getElementById('stats').innerHTML=sh;

  var cols=mode==='fond'?CF:mode==='tech'?CT:CC;
  var thH='<tr>';
  for(i=0;i<cols.length;i++){c=cols[i];thH+='<th class="'+(sortCol===c?'sorted':'')+'" onclick="doSort(\''+c+'\')">'+( CL[c]||c)+(sortCol===c?(sortDir>0?' ^':' v'):'')+' </th>';}
  thH+='</tr>';
  document.getElementById('thead').innerHTML=thH;

  var empty=document.getElementById('empty');
  if(!data.length){document.getElementById('tbody').innerHTML='';empty.style.display='block';return;}
  empty.style.display='none';
  var rows='';
  for(i=0;i<data.length;i++){rows+='<tr>';for(j=0;j<cols.length;j++)rows+='<td>'+cH(cols[j],data[i])+'</td>';rows+='</tr>';}
  document.getElementById('tbody').innerHTML=rows;
}

function openCfg(){
  document.getElementById('c-per').value=CFG.perMax;document.getElementById('c-pbv').value=CFG.pbvMax;
  document.getElementById('c-ev').value=15;document.getElementById('c-roe').value=CFG.roeMin;
  document.getElementById('c-rsi').value=CFG.rsiSurvente;document.getElementById('c-stoch').value=CFG.stochSurvente;
  document.getElementById('c-cci').value=CFG.cciSurvente;document.getElementById('c-adx').value=CFG.adxMin;
  document.getElementById('c-pv').value=CFG.poidsValue;document.getElementById('c-pg').value=CFG.poidsGrowth;
  document.getElementById('c-pd').value=CFG.poidsDividende;document.getElementById('c-pt').value=CFG.poidsTechnique;
  document.getElementById('c-smin').value=CFG.scoreMin;
  document.getElementById('modal-cfg').className='modal-bg open';
}
function closeCfg(){document.getElementById('modal-cfg').className='modal-bg';}
function applyCfg(){
  var pv=parseFloat(document.getElementById('c-pv').value)||0,pg=parseFloat(document.getElementById('c-pg').value)||0,pd=parseFloat(document.getElementById('c-pd').value)||0,pt=parseFloat(document.getElementById('c-pt').value)||0;
  if(pv+pg+pd+pt!==100){alert('Les 4 poids doivent totaliser 100. Total: '+(pv+pg+pd+pt));return;}
  CFG.perMax=parseFloat(document.getElementById('c-per').value)||CFG.perMax;
  CFG.pbvMax=parseFloat(document.getElementById('c-pbv').value)||CFG.pbvMax;
  CFG.roeMin=parseFloat(document.getElementById('c-roe').value)||CFG.roeMin;
  CFG.rsiSurvente=parseFloat(document.getElementById('c-rsi').value)||CFG.rsiSurvente;
  CFG.stochSurvente=parseFloat(document.getElementById('c-stoch').value)||CFG.stochSurvente;
  CFG.cciSurvente=parseFloat(document.getElementById('c-cci').value)||CFG.cciSurvente;
  CFG.adxMin=parseFloat(document.getElementById('c-adx').value)||CFG.adxMin;
  CFG.poidsValue=pv;CFG.poidsGrowth=pg;CFG.poidsDividende=pd;CFG.poidsTechnique=pt;
  CFG.scoreMin=parseFloat(document.getElementById('c-smin').value)||CFG.scoreMin;
  document.getElementById('ms').value=CFG.scoreMin;
  closeCfg();render();
}

function openImp(){document.getElementById('modal-imp').className='modal-bg open';}
function closeImp(){document.getElementById('modal-imp').className='modal-bg';}
function importTickers(){
  var txt=document.getElementById('imp-text').value;
  var tks=txt.split(/[\n,;\s]+/).map(function(t){return t.trim().toUpperCase();}).filter(function(t){return t.length>0;});
  if(!tks.length){alert('Aucun ticker valide');return;}
  var found=[],notFound=[],i,j,hit;
  for(i=0;i<tks.length;i++){hit=null;for(j=0;j<RAW_ORIG.length;j++){if(RAW_ORIG[j].ticker.toUpperCase()===tks[i]){hit=RAW_ORIG[j];break;}}if(hit){var r=JSON.parse(JSON.stringify(hit));r._imported=true;found.push(r);}else notFound.push(tks[i]);}
  IMP_DATA=found;
  var it=document.getElementById('tab-imp');if(it)it.style.display='';
  closeImp();setTab('imp');
  if(notFound.length)alert(found.length+' ticker(s) trouves.\n'+notFound.length+' absents du dernier scan: '+notFound.join(', '));
}

function triggerScan(){
  if(!GH_TOKEN||GH_TOKEN==='VOTRE_GH_TOKEN'){alert('Token GitHub non configure.\nDans generate.py, remplacez VOTRE_USERNAME et VOTRE_GH_TOKEN.\n(github.com > Settings > Developer settings > Personal access tokens, scope: workflow)');return;}
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

render();
"""

page.append(js)
page.append('</script></body></html>')

out = Path('docs/index.html')
out.parent.mkdir(exist_ok=True)
out.write_text('\n'.join(page), encoding='utf-8')
print('HTML OK: ' + str(out))
