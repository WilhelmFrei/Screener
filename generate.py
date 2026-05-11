#!/usr/bin/env python3
"""
Stock Screener — génère un fichier HTML statique avec données Yahoo Finance
Lancé automatiquement par GitHub Actions chaque matin
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
SCORE_MIN        = 35
TOP_N            = 150
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
# LISTE D'ACTIONS
# ══════════════════════════════════════════════════════════════
ACTIONS_EUROPE = [
    # CAC 40
    'AIR.PA','AI.PA','ATO.PA','BNP.PA','EN.PA','CAP.PA','CA.PA','ACA.PA',
    'DSY.PA','ENGI.PA','EL.PA','RMS.PA','KER.PA','LR.PA','OR.PA','MC.PA',
    'ORA.PA','PUB.PA','RI.PA','RNO.PA','SAF.PA','SGO.PA','SAN.PA','SU.PA',
    'GLE.PA','HO.PA','TTE.PA','VIE.PA','DG.PA','WLN.PA','ERF.PA','CS.PA',
    'ALO.PA','MT.AS','STM.PA',
    # SBF 120
    'ML.PA','SPIE.PA','COFA.PA','CNP.PA','ELIS.PA','GTT.PA','MMB.PA','UBI.PA',
    # DAX
    'SAP.DE','SIE.DE','ALV.DE','MUV2.DE','BAYN.DE','BMW.DE','MBG.DE',
    'RWE.DE','DTE.DE','BAS.DE','IFX.DE','MRK.DE','VOW3.DE','ADS.DE',
    'DB1.DE','DHL.DE','EOAN.DE','FRE.DE','HEI.DE','HEN3.DE','LIN.DE',
    'VNA.DE','ZAL.DE','DHER.DE','BEI.DE','ENR.DE','PAH3.DE','SRT3.DE',
    'EVK.DE','HAB.DE','HLAG.DE','KGX.DE','LEG.DE','PSM.DE','PUM.DE',
    'SGL.DE','TUI1.DE','WAF.DE','WCH.DE',
    # Benelux
    'ASML.AS','INGA.AS','PHIA.AS','HEIA.AS','NN.AS','WKL.AS',
    'AKZA.AS','UNA.AS','IMCD.AS','BESI.AS','AMG.AS','AALB.AS','ADYEN.AS',
    'UCB.BR','ABI.BR','KBC.BR','SOLB.BR','COLR.BR','GBLB.BR','WDP.BR',
    # Ibérie
    'ITX.MC','BBVA.MC','SAN.MC','TEF.MC','IBE.MC','REP.MC','AMS.MC',
    'ELE.MC','MAP.MC','AENA.MC','ACS.MC','GRF.MC','IAG.MC','NTGY.MC',
    'EDP.LS','EDPR.LS','GALP.LS','JMT.LS','NOS.LS',
    # Italie
    'ENI.MI','ENEL.MI','ISP.MI','UCG.MI','LDO.MI','MB.MI','PRY.MI',
    'A2A.MI','BAMI.MI','G.MI','MONC.MI','RACE.MI','SRG.MI',
    # Suisse
    'NESN.SW','NOVN.SW','ROG.SW','ABBN.SW','ZURN.SW','UBSG.SW','ADEN.SW',
    'ALC.SW','GIVN.SW','SIKA.SW','SLHN.SW','CFR.SW','EMMN.SW',
    'GEBN.SW','KNIN.SW','PGHN.SW','SCMN.SW','TEMN.SW',
    # Nordiques
    'ORSTED.CO','NOVO-B.CO','DSV.CO','CARL-B.CO','MAERSK-B.CO',
    'VOLV-B.ST','ERIC-B.ST','SEB-A.ST','SWED-A.ST','SHB-A.ST',
    'ATCO-A.ST','ASSA-B.ST','ALFA.ST','SAND.ST','SKA-B.ST','SKF-B.ST',
    'TEL2-B.ST','TELIA.ST','DNB.OL','EQNR.OL','TEL.OL','MOWI.OL',
    'STERV.HE','NESTE.HE','KNEBV.HE',
    # UK
    'RELX','BP.L','SHEL.L','HSBA.L','LLOY.L','AZN.L','GSK.L','ULVR.L',
]

ACTIONS_USA = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','AVGO','ORCL','ADBE',
    'CRM','AMD','QCOM','TXN','INTC','IBM','CSCO','NOW','INTU','AMAT',
    'MU','KLAC','LRCX','ADI','MCHP','CDNS','SNPS','FTNT','PANW','CRWD',
    'NET','SNOW','PLTR','UBER','TTD','VEEV','WDAY','PAYC','EPAM',
    'JNJ','UNH','ABT','TMO','MRK','ABBV','PFE','BMY','AMGN','LLY',
    'MDT','ISRG','SYK','EW','BDX','CI','CVS','HUM','CNC','MOH',
    'IQV','LH','DGX','VRTX','REGN','BIIB','GILD','MRNA','ZTS','IDXX',
    'HOLX','MCK','CAH','GEHC','RMD','BAX','BSX','ZBH',
    'JPM','BAC','WFC','GS','MS','C','BLK','SPGI','MCO','ICE',
    'CME','AXP','V','MA','COF','BX','KKR','APO','ARES',
    'ARCC','MAIN','BEN','TROW','STT','NTRS','FITB','RF','HBAN','KEY',
    'PG','KO','PEP','MDLZ','KHC','GIS','CAG','MKC','HRL',
    'WMT','COST','TGT','HD','LOW','NKE','MCD','SBUX','YUM','DRI',
    'CMG','DPZ','QSR','PM','MO','BTI','EL','CL','CHD','CLX',
    'XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','HAL','BKR',
    'OXY','DVN','EQT','ENB','TRP',
    'GE','BA','CAT','HON','RTX','LMT','NOC','GD','ITW','EMR',
    'ETN','PH','DOV','ROK','AME','UPS','FDX','DAL','UAL','LUV',
    'UNP','CSX','NSC','CP','DE','AGCO',
    'NEE','DUK','SO','D','AEP','EXC','ED','WEC','CMS',
    'VZ','T','TMUS','CMCSA','DIS','NFLX',
    'PLD','AMT','CCI','EQIX','PSA','O','WELL','DLR','AVB','EQR',
    'NNN','VICI','IRM','REXR','EXR','WPC','STAG',
]

PEA_SET = set([
    'AIR.PA','AI.PA','ALO.PA','ATO.PA','BNP.PA','EN.PA','CAP.PA','CA.PA',
    'ACA.PA','DSY.PA','ENGI.PA','EL.PA','RMS.PA','KER.PA','LR.PA','OR.PA',
    'MC.PA','ORA.PA','PUB.PA','RI.PA','RNO.PA','SAF.PA','SGO.PA','SAN.PA',
    'SU.PA','GLE.PA','STM.PA','HO.PA','TTE.PA','VIE.PA','DG.PA','WLN.PA',
    'ERF.PA','CS.PA','ML.PA','SPIE.PA','COFA.PA','CNP.PA','ELIS.PA',
    'GTT.PA','MMB.PA','UBI.PA','MT.AS',
    'SAP.DE','SIE.DE','ALV.DE','MUV2.DE','BAYN.DE','BMW.DE','MBG.DE',
    'RWE.DE','DTE.DE','BAS.DE','IFX.DE','MRK.DE','VOW3.DE','ADS.DE',
    'DB1.DE','DHL.DE','EOAN.DE','FRE.DE','HEI.DE','HEN3.DE','LIN.DE',
    'VNA.DE','ZAL.DE','DHER.DE','BEI.DE','SGL.DE','SRT3.DE','EVK.DE',
    'ASML.AS','INGA.AS','PHIA.AS','HEIA.AS','NN.AS','WKL.AS','AKZA.AS',
    'UNA.AS','IMCD.AS','BESI.AS','AMG.AS','ADYEN.AS',
    'UCB.BR','ABI.BR','KBC.BR','SOLB.BR','COLR.BR','GBLB.BR','WDP.BR',
    'ITX.MC','BBVA.MC','SAN.MC','TEF.MC','IBE.MC','REP.MC','AMS.MC',
    'ELE.MC','MAP.MC','AENA.MC','ACS.MC','GRF.MC','IAG.MC','NTGY.MC',
    'EDP.LS','EDPR.LS','GALP.LS','JMT.LS',
    'ENI.MI','ENEL.MI','ISP.MI','UCG.MI','LDO.MI','MB.MI','PRY.MI',
    'NESN.SW','NOVN.SW','ROG.SW','ABBN.SW','ZURN.SW','UBSG.SW','ADEN.SW',
    'ALC.SW','GIVN.SW','SIKA.SW','CFR.SW','GEBN.SW','KNIN.SW',
    'ORSTED.CO','NOVO-B.CO','DSV.CO','CARL-B.CO',
    'VOLV-B.ST','ERIC-B.ST','SEB-A.ST','SWED-A.ST','ASSA-B.ST',
    'DNB.OL','EQNR.OL','TEL.OL','MOWI.OL',
    'STERV.HE','NESTE.HE','KNEBV.HE',
])

# ══════════════════════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ══════════════════════════════════════════════════════════════
def calc_rsi(s, p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def calc_macd(s, f=12, sl=26, sig=9):
    m=s.ewm(span=f,adjust=False).mean()-s.ewm(span=sl,adjust=False).mean()
    sg=m.ewm(span=sig,adjust=False).mean()
    return m,sg,m-sg

def calc_cci(h,l,c,p=20):
    tp=(h+l+c)/3; sma=tp.rolling(p).mean()
    mad=tp.rolling(p).apply(lambda x:np.mean(np.abs(x-np.mean(x))))
    return (tp-sma)/(0.015*mad.replace(0,np.nan))

def calc_stoch(h,l,c,k=14,d=3):
    lm=l.rolling(k).min(); hm=h.rolling(k).max()
    sk=100*(c-lm)/(hm-lm).replace(0,np.nan)
    return sk,sk.rolling(d).mean()

def calc_adx(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(p).mean()
    dp=(h-h.shift()).clip(lower=0); dm=(l.shift()-l).clip(lower=0)
    dp=dp.where(dp>dm,0); dm=dm.where(dm>dp,0)
    dip=100*dp.rolling(p).mean()/atr.replace(0,np.nan)
    dim=100*dm.rolling(p).mean()/atr.replace(0,np.nan)
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.rolling(p).mean()

def calc_boll(s,p=20,n=2):
    m=s.rolling(p).mean(); b=s.rolling(p).std()
    up=m+n*b; lo=m-n*b
    return (s-lo)/(up-lo).replace(0,np.nan)

# ══════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════
def score_value(info):
    p=0
    per=info.get('trailingPE') or info.get('forwardPE')
    pbv=info.get('priceToBook')
    ev=info.get('enterpriseToEbitda')
    if per and per>0: p+=40 if per<=10 else 28 if per<=15 else 15 if per<=PER_MAX else 0
    if pbv and pbv>0: p+=30 if pbv<=1 else 20 if pbv<=2 else 10 if pbv<=PBV_MAX else 0
    if ev  and ev>0:  p+=30 if ev<=8  else 20 if ev<=12 else 10 if ev<=15 else 0
    return min(100,p)

def score_growth(info):
    p=0
    roe=info.get('returnOnEquity'); eg=info.get('earningsGrowth')
    peg=info.get('pegRatio')
    if roe: rp=roe*100; p+=35 if rp>=25 else 25 if rp>=15 else 12 if rp>=ROE_MIN else 0
    if eg:  ep=eg*100;  p+=40 if ep>=25 else 28 if ep>=15 else 15 if ep>0 else 0
    if peg and peg>0:   p+=25 if peg<=1 else 15 if peg<=1.5 else 8 if peg<=2.5 else 0
    return min(100,p)

def score_div(info):
    p=0
    yld=info.get('dividendYield'); po=info.get('payoutRatio')
    if yld: yp=yld*100; p+=60 if yp>=6 else 45 if yp>=4 else 30 if yp>=3 else 15 if yp>0 else 0
    if po and 0<po<0.8: p+=20
    elif po and po<=1:  p+=10
    return min(100,p)

def score_tech(hist):
    if hist is None or len(hist)<30: return 0,{}
    try:
        c=hist['Close'].dropna(); h=hist['High'].dropna(); l=hist['Low'].dropna()
        if len(c)<30: return 0,{}
        r=calc_rsi(c).iloc[-1]
        ml,ms,mh=calc_macd(c); mh_v=mh.iloc[-1]; ml_v=ml.iloc[-1]; ms_v=ms.iloc[-1]
        cc=calc_cci(h,l,c).iloc[-1]
        sk,_=calc_stoch(h,l,c); st=sk.iloc[-1]
        ax=calc_adx(h,l,c).iloc[-1]
        bb=calc_boll(c).iloc[-1]
        px=float(c.iloc[-1])
        mms={f'MM{p}':float(c.rolling(p).mean().iloc[-1]) for p in [20,50,200] if len(c)>=p}
        pts=0; sigs=[]
        if not np.isnan(r):
            if r<=30:           pts+=28; sigs.append(f'RSI {r:.0f} 🟢')
            elif r<=RSI_SURVENTE:pts+=18; sigs.append(f'RSI {r:.0f}')
            elif r<=50:          pts+=8
        if not np.isnan(mh_v):
            if mh_v>0 and ml_v>ms_v: pts+=20; sigs.append('MACD 🟢')
            elif mh_v>0:              pts+=12; sigs.append('MACD')
            elif mh_v>-0.1:           pts+=5
        if not np.isnan(cc):
            if cc<CCI_SURVENTE:  pts+=18; sigs.append(f'CCI {cc:.0f} 🟢')
            elif 50<=cc<=150:    pts+=12; sigs.append(f'CCI {cc:.0f}')
            elif 0<cc<50:        pts+=6
        if not np.isnan(st):
            if st<=20:           pts+=14; sigs.append(f'Stoch {st:.0f} 🟢')
            elif st<=STOCH_SURVENTE:pts+=9; sigs.append(f'Stoch {st:.0f}')
        above=sum(1 for v in mms.values() if px>v)
        if above==len(mms)>0: pts+=15; sigs.append('MM ✅')
        elif above>=2:         pts+=8
        if not np.isnan(ax) and ax>=ADX_MIN: pts+=5; sigs.append(f'ADX {ax:.0f}')
        det={
            'RSI':   round(float(r),1)    if not np.isnan(r)    else None,
            'MACD':  round(float(mh_v),3) if not np.isnan(mh_v) else None,
            'CCI':   round(float(cc),1)   if not np.isnan(cc)   else None,
            'Stoch': round(float(st),1)   if not np.isnan(st)   else None,
            'ADX':   round(float(ax),1)   if not np.isnan(ax)   else None,
            'BB':    round(float(bb),2)   if not np.isnan(bb)   else None,
            'Signaux': ' | '.join(sigs),
        }
        for k,v in mms.items():
            det[k]=f'{((px/v)-1)*100:+.1f}%'
        return min(100,pts),det
    except: return 0,{}

# ══════════════════════════════════════════════════════════════
# SCAN PRINCIPAL
# ══════════════════════════════════════════════════════════════
tickers = list(dict.fromkeys(ACTIONS_EUROPE + ACTIONS_USA))
d_fin = datetime.today()
d_deb = d_fin - timedelta(days=JOURS_HISTORIQUE)
print(f'🔍 Scan de {len(tickers)} actions...')

# Téléchargement groupé
BATCH=50; all_hist={}
for i in range(0,len(tickers),BATCH):
    batch=tickers[i:i+BATCH]
    try:
        data=yf.download(batch,start=d_deb.strftime('%Y-%m-%d'),
                         end=d_fin.strftime('%Y-%m-%d'),
                         group_by='ticker',auto_adjust=True,
                         progress=False,threads=True)
        for tk in batch:
            try:
                all_hist[tk]=data[tk] if len(batch)>1 and tk in data.columns.get_level_values(0) else (data if len(batch)==1 else None)
            except: all_hist[tk]=None
    except:
        for tk in batch: all_hist[tk]=None
    print(f'  Téléchargé {min(i+BATCH,len(tickers))}/{len(tickers)}')

resultats=[]
for idx,ticker in enumerate(tickers):
    try:
        info=yf.Ticker(ticker).info or {}
        nom=info.get('longName') or info.get('shortName') or ticker
        secteur=info.get('sector') or 'N/A'
        pays=(info.get('country') or '')[:2].upper() or '?'
        devise=info.get('currency') or ('EUR' if any(x in ticker for x in ['.PA','.DE','.AS','.BR','.MC','.MI','.SW','.CO','.ST','.OL','.HE','.LS','.L']) else 'USD')
        hist=all_hist.get(ticker)
        prix_hist=float(hist['Close'].iloc[-1]) if hist is not None and len(hist)>0 else None
        prix=info.get('currentPrice') or info.get('regularMarketPrice') or prix_hist

        per=info.get('trailingPE') or info.get('forwardPE')
        pbv=info.get('priceToBook')
        roe_r=info.get('returnOnEquity'); roe=round(roe_r*100,1) if roe_r else None
        mg_r=info.get('profitMargins');   mg=round(mg_r*100,1)  if mg_r  else None
        yld_r=info.get('dividendYield'); yld=round(yld_r*100,2) if yld_r else None
        peg=info.get('pegRatio')
        eg_r=info.get('earningsGrowth'); eg=round(eg_r*100,1) if eg_r else None
        ev=info.get('enterpriseToEbitda')
        cap=info.get('marketCap')

        sv=score_value(info)
        sg=score_growth(info)
        sd=score_div(info)
        st,td=score_tech(hist)

        total_f=POIDS_VALUE+POIDS_GROWTH+POIDS_DIVIDENDE
        sf=round(sv*POIDS_VALUE/total_f + sg*POIDS_GROWTH/total_f + sd*POIDS_DIVIDENDE/total_f)
        score=round(sf*(1-POIDS_TECHNIQUE/100) + st*(POIDS_TECHNIQUE/100))

        if score<SCORE_MIN: continue

        row={
            'ticker':ticker,'nom':nom[:35],'secteur':secteur,
            'pays':pays,'pea':1 if ticker in PEA_SET else 0,
            'devise':devise,
            'prix':round(float(prix),2) if prix else None,
            'per':round(float(per),1) if per and per>0 else None,
            'pbv':round(float(pbv),2) if pbv else None,
            'roe':roe,'marge':mg,'div':yld,'bpa':eg,
            'ev':round(float(ev),1) if ev else None,
            'peg':round(float(peg),2) if peg and peg>0 else None,
            'sv':sv,'sg':sg,'sd':sd,'st':st,'score':score,
            'rsi':td.get('RSI'),'macd':td.get('MACD'),
            'cci':td.get('CCI'),'stoch':td.get('Stoch'),
            'adx':td.get('ADX'),'bb':td.get('BB'),
            'mm20':td.get('MM20'),'mm50':td.get('MM50'),'mm200':td.get('MM200'),
            'signaux':td.get('Signaux',''),
        }
        resultats.append(row)
        if (idx+1)%25==0:
            print(f'  Analysées {idx+1}/{len(tickers)} | {len(resultats)} opportunités')
    except Exception as e:
        pass

resultats.sort(key=lambda x: x['score'], reverse=True)
resultats=resultats[:TOP_N]
print(f'✅ {len(resultats)} opportunités trouvées')

# ══════════════════════════════════════════════════════════════
# GÉNÉRATION HTML
# ══════════════════════════════════════════════════════════════
data_json = json.dumps(resultats, ensure_ascii=False, default=str)
date_str = d_fin.strftime('%d/%m/%Y à %H:%M')
nb_scanned = len(tickers)

html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>📡 Stock Screener</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0e1a;color:#e2e8f0;font-family:system-ui,sans-serif;min-height:100vh}}
body::before{{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,170,.03) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,170,.03) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}}

/* LOGIN */
#login{{position:fixed;inset:0;background:#0a0e1a;z-index:100;display:flex;align-items:center;justify-content:center}}
.login-box{{background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:40px;width:320px;text-align:center}}
.login-box h2{{font-family:monospace;font-size:22px;color:#00d4aa;margin-bottom:8px}}
.login-box p{{font-size:12px;color:#64748b;margin-bottom:24px}}
.login-box input{{width:100%;background:#0d1520;border:1px solid #1e2d45;color:#e2e8f0;font-size:14px;padding:12px 14px;border-radius:9px;outline:none;text-align:center;letter-spacing:4px;font-family:monospace;margin-bottom:12px}}
.login-box input:focus{{border-color:#00d4aa}}
.login-box button{{width:100%;background:linear-gradient(135deg,#00d4aa,#00a87f);color:#000;font-weight:800;font-size:14px;padding:12px;border:none;border-radius:9px;cursor:pointer;font-family:monospace}}
.login-err{{color:#ef4444;font-size:12px;margin-top:8px;min-height:18px}}

/* APP */
#app{{display:none;position:relative;z-index:1}}
.hdr{{background:#111827;border-bottom:1px solid #1e2d45;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;position:sticky;top:0;z-index:10}}
.hdr-title{{font-family:monospace;font-size:18px;font-weight:900;color:#00d4aa}}
.hdr-sub{{font-size:10px;color:#64748b;margin-top:2px}}
.hdr-right{{display:flex;align-items:center;gap:10px}}
.update-badge{{font-family:monospace;font-size:10px;color:#64748b;background:#0d1520;padding:5px 10px;border-radius:6px;border:1px solid #1e2d45}}
.refresh-btn{{background:linear-gradient(135deg,#00d4aa,#00a87f);color:#000;font-weight:800;font-size:11px;padding:7px 14px;border:none;border-radius:8px;cursor:pointer;font-family:monospace;white-space:nowrap}}

.wrap{{max-width:1500px;margin:0 auto;padding:14px}}

/* MODE */
.mode-row{{display:flex;gap:8px;margin-bottom:12px}}
.mode-btn{{flex:1;padding:10px 6px;border-radius:10px;border:2px solid #1e2d45;background:#0d1520;cursor:pointer;text-align:center;transition:.15s}}
.mode-btn.on{{border-color:#00d4aa;background:#00d4aa18}}
.mode-btn .ico{{font-size:18px;margin-bottom:3px}}
.mode-btn .lbl{{font-size:10px;font-weight:700;color:#64748b;font-family:monospace}}
.mode-btn.on .lbl{{color:#00d4aa}}
.mode-desc{{background:#0d1520;border:1px solid #1e2d45;border-radius:9px;padding:9px 13px;font-size:11px;color:#64748b;line-height:1.6;margin-bottom:12px}}

/* FILTERS */
.filters{{background:#111827;border:1px solid #1e2d45;border-radius:11px;padding:12px 14px;display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;margin-bottom:12px}}
.fg{{display:flex;flex-direction:column;gap:4px}}
.fg label{{font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.8px}}
select,input[type=text],input[type=number]{{background:#0d1520;border:1px solid #1e2d45;color:#e2e8f0;font-size:12px;padding:7px 9px;border-radius:7px;outline:none;font-family:inherit}}
select:focus,input:focus{{border-color:#00d4aa}}

/* TABS */
.tabs{{display:flex;gap:5px;margin-bottom:10px}}
.tab{{padding:7px 16px;border-radius:8px;border:1px solid #1e2d45;background:#0d1520;color:#64748b;cursor:pointer;font-size:11px;font-weight:700;font-family:monospace;display:flex;align-items:center;gap:6px}}
.tab.on{{border-color:#00d4aa;background:#111827;color:#e2e8f0}}
.badge{{background:#1e2d45;color:#64748b;font-size:9px;font-weight:700;padding:1px 6px;border-radius:20px}}
.tab.on .badge{{background:#00d4aa;color:#000}}

/* STATS */
.stats{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}}
.stat{{background:#111827;border:1px solid #1e2d45;border-radius:9px;padding:9px 13px;flex:1;min-width:80px}}
.stat .sl{{font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.6px}}
.stat .sv{{font-family:monospace;font-size:18px;font-weight:900;margin-top:2px}}

/* TABLE */
.tbl-wrap{{background:#111827;border:1px solid #1e2d45;border-radius:12px;overflow:hidden}}
.tbl-scroll{{overflow-x:auto;max-height:60vh;overflow-y:auto}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
thead th{{background:#0d1520;padding:8px 9px;font-size:8px;font-family:monospace;text-transform:uppercase;letter-spacing:.7px;color:#64748b;cursor:pointer;border-bottom:1px solid #1e2d45;white-space:nowrap;position:sticky;top:0;z-index:2;text-align:left;user-select:none}}
thead th:hover{{color:#00d4aa}}
thead th.sorted{{color:#00d4aa}}
tbody td{{padding:8px 9px;border-bottom:1px solid #0d1520;vertical-align:middle;white-space:nowrap}}
tbody tr:nth-child(even) td{{background:#0a1020}}
tbody tr:hover td{{background:#152030!important}}
.tk{{font-family:monospace;font-weight:700;color:#00d4aa;font-size:12px}}
.pea-tag{{font-size:8px;padding:1px 4px;border-radius:3px;background:#00d4aa18;color:#00d4aa;border:1px solid #00d4aa33;margin-left:3px}}
.nm{{font-size:10px;color:#64748b;max-width:130px;overflow:hidden;text-overflow:ellipsis}}
.ct{{font-size:9px;color:#2a4060}}
.bar-w{{display:flex;align-items:center;gap:5px}}
.bar{{width:40px;height:4px;border-radius:2px;background:#1e2d45;overflow:hidden}}
.bar-f{{height:100%;border-radius:2px}}
.chip{{display:inline-block;font-size:9px;font-weight:700;padding:2px 5px;border-radius:20px;margin-right:2px;font-family:monospace;background:#00d4aa18;color:#00d4aa;border:1px solid #00d4aa33}}
.empty{{text-align:center;padding:50px 20px;color:#64748b}}

/* DISCLAIMER */
.disc{{margin-top:10px;padding:9px 14px;border:1px solid #f59e0b33;border-radius:9px;background:#f59e0b06;font-size:10px;color:#64748b;line-height:1.6}}
</style>
</head>
<body>

<!-- LOGIN -->
<div id="login">
  <div class="login-box">
    <div style="font-size:36px;margin-bottom:8px">📡</div>
    <h2>STOCK SCREENER</h2>
    <p>Accès sécurisé — entrez votre code</p>
    <input type="password" id="pwd" placeholder="••••••" onkeydown="if(event.key==='Enter')checkPwd()">
    <button onclick="checkPwd()">Accéder →</button>
    <div class="login-err" id="err"></div>
  </div>
</div>

<!-- APP -->
<div id="app">
  <div class="hdr">
    <div>
      <div class="hdr-title">📡 STOCK SCREENER</div>
      <div class="hdr-sub">Données Yahoo Finance · {nb_scanned} actions · Mis à jour le {date_str}</div>
    </div>
    <div class="hdr-right">
      <span class="update-badge" id="nextUpdate"></span>
      <button class="refresh-btn" onclick="alert('Prochain scan automatique demain matin 7h\\nPour forcer : GitHub → Actions → Run workflow')">🔄 Infos scan</button>
    </div>
  </div>

  <div class="wrap">
    <!-- MODE -->
    <div class="mode-row">
      <div class="mode-btn on" id="m-combine" onclick="setMode('combine')"><div class="ico">🎯</div><div class="lbl">Fond. + Tech.</div></div>
      <div class="mode-btn"   id="m-fond"    onclick="setMode('fond')">   <div class="ico">📊</div><div class="lbl">Fondamental</div></div>
      <div class="mode-btn"   id="m-tech"    onclick="setMode('tech')">   <div class="ico">📈</div><div class="lbl">Technique</div></div>
    </div>
    <div class="mode-desc" id="mode-desc"></div>

    <!-- FILTRES -->
    <div class="filters">
      <div class="fg"><label>Recherche</label><input type="text" id="q" placeholder="Ticker ou nom..." oninput="render()" style="width:140px"></div>
      <div class="fg"><label>Marché</label>
        <select id="mkt" onchange="render()">
          <option value="all">Europe + USA</option>
          <option value="eu">Europe</option>
          <option value="us">USA</option>
        </select>
      </div>
      <div class="fg"><label>Secteur</label><select id="sect" onchange="render()"></select></div>
      <div class="fg"><label>Score min</label><input type="number" id="ms" value="{SCORE_MIN}" min="0" max="100" step="5" onchange="render()" style="width:65px"></div>
    </div>

    <!-- TABS -->
    <div class="tabs">
      <div class="tab on" id="tab-all" onclick="setTab('all')">🌍 Toutes <span class="badge" id="b-all">0</span></div>
      <div class="tab"    id="tab-pea" onclick="setTab('pea')">🇫🇷 PEA   <span class="badge" id="b-pea">0</span></div>
    </div>

    <!-- STATS -->
    <div class="stats" id="stats"></div>

    <!-- TABLE -->
    <div class="tbl-wrap">
      <div class="tbl-scroll">
        <table><thead id="thead"></thead><tbody id="tbody"></tbody></table>
        <div class="empty" id="empty" style="display:none">📭 Aucun résultat avec ces filtres</div>
      </div>
    </div>

    <div class="disc">
      <strong style="color:#f59e0b">⚠️</strong>
      Données Yahoo Finance à titre indicatif. Vérifiez sur une source officielle avant tout investissement.
      <strong>Ne constitue pas un conseil en investissement.</strong>
    </div>
  </div>
</div>

<script>
const PWD = '1510';

function checkPwd() {
  const v = document.getElementById('pwd').value;
  if (v === PWD) {
    document.getElementById('login').style.display = 'none';
    document.getElementById('app').style.display = 'block';
    render();
  } else {
    document.getElementById('err').textContent = '❌ Code incorrect';
    document.getElementById('pwd').value = '';
  }
}


// ── Données ───────────────────────────────────────────────────
const RAW = {data_json};
let mode='combine', curTab='all', sortCol='score', sortDir=-1;

// Secteurs
const sects=['all',...new Set(RAW.map(r=>r.secteur).filter(Boolean))].sort();
const se=document.getElementById('sect');
sects.forEach(s=>{{ const o=document.createElement('option'); o.value=s; o.text=s==='all'?'Tous secteurs':s; se.appendChild(o); }});

// Prochaine MAJ
function nextUpdate(){{
  const now=new Date(); const nxt=new Date();
  nxt.setDate(now.getDate()+(now.getHours()>=7?1:0));
  nxt.setHours(7,0,0,0);
  const diff=nxt-now; const h=Math.floor(diff/3600000); const m=Math.floor((diff%3600000)/60000);
  document.getElementById('nextUpdate').textContent=`Prochain scan dans ${{h}}h${{m}}m`;
}}
nextUpdate(); setInterval(nextUpdate,60000);

// Mode
const DESCS={{
  'combine':'🎯 <strong style="color:#00d4aa">Mode Combiné</strong> — Score = Fondamentaux + Technique. Détecte les actions sous-valorisées ET avec un signal d\'entrée favorable.',
  'fond':   '📊 <strong style="color:#00d4aa">Mode Fondamental</strong> — PER, ROE, dividende, croissance. Idéal pour le moyen/long terme.',
  'tech':   '📈 <strong style="color:#00d4aa">Mode Technique</strong> — RSI survendu, MACD haussier, CCI, Stochastique, Moyennes mobiles. Idéal pour le court/moyen terme.',
}};
function setMode(m){{
  mode=m;
  ['combine','fond','tech'].forEach(x=>document.getElementById('m-'+x).className='mode-btn'+(x===m?' on':''));
  document.getElementById('mode-desc').innerHTML=DESCS[m];
  render();
}}
setMode('combine');

// Tab
function setTab(t){{
  curTab=t;
  ['all','pea'].forEach(x=>document.getElementById('tab-'+x).className='tab'+(x===t?' on':''));
  render();
}}

// Score selon mode
function getScore(r){{
  if(mode==='fond') return Math.round(r.sv*0.4+r.sg*0.35+r.sd*0.25);
  if(mode==='tech') return r.st;
  return r.score;
}}

// Filtres
function filter(){{
  const q=(document.getElementById('q').value||'').toLowerCase();
  const mkt=document.getElementById('mkt').value;
  const sec=document.getElementById('sect').value;
  const ms=parseFloat(document.getElementById('ms').value)||0;
  return RAW.filter(r=>{{
    const sc=getScore(r);
    if(sc<ms) return false;
    if(mkt==='eu'&&r.pays==='US') return false;
    if(mkt==='us'&&r.pays!=='US') return false;
    if(sec!=='all'&&r.secteur!==sec) return false;
    if(q&&!r.ticker.toLowerCase().includes(q)&&!r.nom.toLowerCase().includes(q)) return false;
    return true;
  }}).sort((a,b)=>sortDir*(getScore(b)-getScore(a)));
}}

// Formatage
function n(v,suf='',dec=1){{ return v!==null&&v!==undefined?`<span style="font-family:monospace;font-size:11px">${{parseFloat(v).toFixed(dec)}}${{suf}}</span>`:'<span style="color:#2a3a52">—</span>'; }}
function nc(v,lo,hi,inv=false){{
  if(v===null||v===undefined)return '<span style="color:#2a3a52">—</span>';
  const p=parseFloat(v); let c='#94a3b8';
  if(inv){{if(p>=hi)c='#10b981';else if(p<=lo)c='#ef4444';}}
  else{{if(p<=lo)c='#10b981';else if(p>=hi)c='#ef4444';}}
  return `<span style="font-family:monospace;font-size:11px;color:${{c}}">${{p.toFixed(1)}}</span>`;
}}
function barHtml(v){{
  const pct=Math.min(v||0,100); const c=pct>=70?'#10b981':pct>=50?'#f59e0b':'#64748b';
  return `<div class="bar-w"><div class="bar"><div class="bar-f" style="width:${{pct}}%;background:${{c}}"></div></div><span style="font-family:monospace;font-size:11px;color:${{c}};font-weight:700">${{Math.round(pct)}}</span></div>`;
}}
function chips(v){{
  if(!v)return '—';
  return v.split('|').map(s=>s.trim()).filter(Boolean).map(s=>`<span class="chip">${{s}}</span>`).join('');
}}

// Colonnes
const C_FOND=['ticker','prix','per','pbv','roe','marge','div','bpa','ev','sv','sg','sd','score'];
const C_TECH=['ticker','prix','rsi','macd','cci','stoch','adx','bb','mm20','mm50','mm200','st','signaux'];
const C_COMB=['ticker','prix','per','roe','div','rsi','macd','cci','mm200','sv','st','score','signaux'];
const C_LABELS={{
  ticker:'Action',prix:'Cours',per:'PER',pbv:'P/Book',roe:'ROE%',marge:'Marge%',
  div:'Div%',bpa:'BPA%',ev:'EV/EBITDA',sv:'Score V',sg:'Score G',sd:'Score D',
  st:'Score T',score:'Score ★',rsi:'RSI',macd:'MACD',cci:'CCI',stoch:'Stoch.',
  adx:'ADX',bb:'BB%',mm20:'vs MM20',mm50:'vs MM50',mm200:'vs MM200',signaux:'Signaux',
}};

function cellHtml(col,r){{
  const v=r[col];
  if(col==='ticker') return `<div><span class="tk">${{r.ticker}}</span>${{r.pea?'<span class="pea-tag">PEA</span>':''}}</div><div class="nm">${{r.nom}}</div><div class="ct">${{r.pays}} · ${{r.secteur}}</div>`;
  if(col==='prix')   return v!==null?`<span style="font-family:monospace">${{r.devise==='USD'?'$':'€'}}${{parseFloat(v).toFixed(2)}}</span>`:'—';
  if(col==='per')    return nc(v,12,25);
  if(col==='pbv')    return n(v,'x',2);
  if(col==='roe')    return nc(v,8,20,true);
  if(col==='marge')  return nc(v,5,15,true);
  if(col==='div')    return v?`<span style="font-family:monospace;color:${{v>=4?'#10b981':'#94a3b8'}}">${{parseFloat(v).toFixed(1)}}%</span>`:'—';
  if(col==='bpa')    return n(v,'%');
  if(col==='ev')     return n(v,'x');
  if(col==='sv'||col==='sg'||col==='sd'||col==='st') return v!==null?barHtml(v):'—';
  if(col==='score')  return v!==null?barHtml(getScore(r)):'—';
  if(col==='rsi')    return v!==null?`<span style="font-family:monospace;font-size:11px;font-weight:${{v<=45||v>=70?700:400}};color:${{v<=30?'#10b981':v<=45?'#f59e0b':v>=70?'#ef4444':'#94a3b8'}}">${{v}}</span>`:'—';
  if(col==='macd')   return v!==null?`<span style="font-family:monospace;font-size:11px;color:${{v>0?'#10b981':v<0?'#ef4444':'#94a3b8'}}">${{parseFloat(v).toFixed(2)}}</span>`:'—';
  if(col==='cci')    return v!==null?`<span style="font-family:monospace;font-size:11px;color:${{v<-100?'#10b981':v>150?'#ef4444':'#94a3b8'}}">${{Math.round(v)}}</span>`:'—';
  if(col==='stoch')  return v!==null?`<span style="font-family:monospace;font-size:11px;color:${{v<=30?'#10b981':v>=70?'#ef4444':'#94a3b8'}}">${{Math.round(v)}}</span>`:'—';
  if(col==='adx')    return v!==null?`<span style="font-family:monospace;font-size:11px;color:${{v>=25?'#00d4aa':'#94a3b8'}}">${{Math.round(v)}}</span>`:'—';
  if(col==='mm20'||col==='mm50'||col==='mm200') return v?`<span style="font-family:monospace;font-size:11px;color:${{v.startsWith('+')?'#10b981':'#ef4444'}}">${{v}}</span>`:'—';
  if(col==='signaux')return chips(v);
  return v!==null&&v!==undefined?String(v):'—';
}}

function render(){{
  const data=filter();
  const allN=data.length;
  const peaD=data.filter(r=>r.pea===1);
  document.getElementById('b-all').textContent=allN;
  document.getElementById('b-pea').textContent=peaD.length;
  const shown=curTab==='pea'?peaD:data;

  // Stats
  const s70=shown.filter(r=>getScore(r)>=70).length;
  const s50=shown.filter(r=>getScore(r)>=50&&getScore(r)<70).length;
  const avgD=shown.length?(shown.reduce((s,r)=>s+(r.div||0),0)/shown.length).toFixed(1)+'%':'—';
  const rsiLow=shown.filter(r=>r.rsi&&r.rsi<=45).length;
  document.getElementById('stats').innerHTML=[
    ['Total',shown.length,'#00d4aa'],['≥ 70',s70,'#10b981'],
    ['50-69',s50,'#f59e0b'],['Div. moy.',avgD,'#8b5cf6'],['RSI≤45',rsiLow,'#3b82f6'],
  ].map(([l,v,c])=>`<div class="stat"><div class="sl">${{l}}</div><div class="sv" style="color:${{c}}">${{v}}</div></div>`).join('');

  // Colonnes
  const cols=mode==='fond'?C_FOND:mode==='tech'?C_TECH:C_COMB;

  // Header
  document.getElementById('thead').innerHTML='<tr>'+cols.map(c=>
    `<th class="${{sortCol===c?'sorted':''}}" onclick="doSort('${{c}}')">${{C_LABELS[c]||c}}${{sortCol===c?(sortDir>0?' ▲':' ▼'):''}}</th>`
  ).join('')+'</tr>';

  // Body
  const empty=document.getElementById('empty');
  if(!shown.length){{ document.getElementById('tbody').innerHTML=''; empty.style.display='block'; return; }}
  empty.style.display='none';
  document.getElementById('tbody').innerHTML=shown.map(r=>
    `<tr>${{cols.map(c=>`<td>${{cellHtml(c,r)}}</td>`).join('')}}</tr>`
  ).join('');
}}

function doSort(col){{
  if(sortCol===col)sortDir*=-1; else{{sortCol=col;sortDir=-1;}}
  render();
}}
</script>
</body>
</html>'''

# Écriture
out = Path('docs/index.html')
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding='utf-8')
print(f'✅ HTML généré : {out} ({len(resultats)} opportunités)')
