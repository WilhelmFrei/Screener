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


# ══ GENERATION HTML ══
import hashlib, os
raw_pwd = os.environ.get('SCREENER_PASSWORD', 'changeme2025')
PWD_HASH = hashlib.sha256(raw_pwd.encode('utf-8')).hexdigest()

DATA_JSON = json.dumps(resultats, ensure_ascii=False, default=str)
DATE_STR  = d_fin.strftime('%d/%m/%Y %H:%M')
CFG_JSON  = json.dumps({
    'perMax':PER_MAX,'pbvMax':PBV_MAX,'roeMin':ROE_MIN,
    'rsiSurvente':RSI_SURVENTE,'cciSurvente':CCI_SURVENTE,'stochSurvente':STOCH_SURVENTE,'adxMin':ADX_MIN,
    'poidsValue':POIDS_VALUE,'poidsGrowth':POIDS_GROWTH,'poidsDividende':POIDS_DIVIDENDE,'poidsTechnique':POIDS_TECHNIQUE,
    'scoreMin':SCORE_MIN,'psMax':4,'pfcfMax':30,'grahamMin':0,'crMin':1.0,'deMax':150,'fcfyMin':2,
    'willrSurvente':-60,'rocMin':5,
    'tPer':1,'tPbv':1,'tEv':1,'tPs':1,'tPfcf':1,'tGraham':1,
    'tRoe':1,'tBpa':1,'tPeg':1,'tCr':1,'tDe':1,'tFcfy':1,
    'tRsi':1,'tMacd':1,'tCci':1,'tStoch':1,'tMm':1,'tAdx':1,'tWillr':1,'tRoc':1,
})

# Lecture du template HTML
template_path = Path('template.html')
if not template_path.exists():
    print('ERREUR: template.html manquant')
    exit(1)

html = template_path.read_text(encoding='utf-8')
html = html.replace('%%PWD_HASH%%',   PWD_HASH)
html = html.replace('%%DATA%%',       DATA_JSON)
html = html.replace('%%CFG%%',        CFG_JSON)
html = html.replace('%%DATE%%',       DATE_STR)
html = html.replace('%%NSCAN%%',      str(len(tickers)))

out = Path('docs/index.html')
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding='utf-8')
print('HTML OK: ' + str(out))


# ══ HISTORIQUE DES SCORES ══
import datetime as _dt

hist_dir = Path('docs/history')
hist_dir.mkdir(exist_ok=True)
today_str = _dt.date.today().isoformat()
# Save today's snapshot (ticker -> score)
snapshot = {r['ticker']: r['score'] for r in resultats}
(hist_dir / (today_str + '.json')).write_text(
    json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')
# Keep only last 30 days
hist_files = sorted(hist_dir.glob('*.json'))
for old in hist_files[:-30]:
    old.unlink()
# Build score history for each ticker (last 30 days)
score_history = {}
for hf in sorted(hist_dir.glob('*.json'))[-30:]:
    day = hf.stem
    try:
        data = json.loads(hf.read_text(encoding='utf-8'))
        for tk, sc in data.items():
            if tk not in score_history:
                score_history[tk] = {}
            score_history[tk][day] = sc
    except Exception:
        pass
# Inject score_history into each result
for r in resultats:
    hist = score_history.get(r['ticker'], {})
    r['score_history'] = [hist[d] for d in sorted(hist.keys())]
print('Historique: ' + str(len(score_history)) + ' tickers sur ' + str(len(list(hist_dir.glob('*.json')))) + ' jours')

# ══ GENERATION HTML depuis template ══
import hashlib, os
raw_pwd = os.environ.get('SCREENER_PASSWORD', 'changeme2025')
PWD_HASH = hashlib.sha256(raw_pwd.encode('utf-8')).hexdigest()

DATA_JSON = json.dumps(resultats, ensure_ascii=False, default=str)
DATE_STR  = d_fin.strftime('%d/%m/%Y %H:%M')
CFG_JSON  = json.dumps({
    'perMax':PER_MAX,'pbvMax':PBV_MAX,'roeMin':ROE_MIN,
    'rsiSurvente':RSI_SURVENTE,'cciSurvente':CCI_SURVENTE,'stochSurvente':STOCH_SURVENTE,'adxMin':ADX_MIN,
    'poidsValue':POIDS_VALUE,'poidsGrowth':POIDS_GROWTH,'poidsDividende':POIDS_DIVIDENDE,'poidsTechnique':POIDS_TECHNIQUE,
    'scoreMin':SCORE_MIN,'psMax':4,'pfcfMax':30,'grahamMin':0,'crMin':1.0,'deMax':150,'fcfyMin':2,
    'willrSurvente':-60,'rocMin':5,
    'tPer':1,'tPbv':1,'tEv':1,'tPs':1,'tPfcf':1,'tGraham':1,
    'tRoe':1,'tBpa':1,'tPeg':1,'tCr':1,'tDe':1,'tFcfy':1,
    'tRsi':1,'tMacd':1,'tCci':1,'tStoch':1,'tMm':1,'tAdx':1,'tWillr':1,'tRoc':1,
})

template = Path('template.html').read_text(encoding='utf-8')
html = (template
    .replace('%%PWD_HASH%%', PWD_HASH)
    .replace('%%DATA%%',     DATA_JSON)
    .replace('%%CFG%%',      CFG_JSON)
    .replace('%%DATE%%',     DATE_STR)
    .replace('%%NSCAN%%',    str(len(tickers)))
)

out = Path('docs/index.html')
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding='utf-8')
print('HTML OK: ' + str(out))
