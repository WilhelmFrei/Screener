#!/usr/bin/env python3
“””
Stock Screener — génère un fichier HTML statique avec données Yahoo Finance
Lancé automatiquement par GitHub Actions chaque matin
“””
import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings(‘ignore’)

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

# LISTE D’ACTIONS

# ══════════════════════════════════════════════════════════════

ACTIONS_EUROPE = [
# CAC 40
‘AIR.PA’,‘AI.PA’,‘ATO.PA’,‘BNP.PA’,‘EN.PA’,‘CAP.PA’,‘CA.PA’,‘ACA.PA’,
‘DSY.PA’,‘ENGI.PA’,‘EL.PA’,‘RMS.PA’,‘KER.PA’,‘LR.PA’,‘OR.PA’,‘MC.PA’,
‘ORA.PA’,‘PUB.PA’,‘RI.PA’,‘RNO.PA’,‘SAF.PA’,‘SGO.PA’,‘SAN.PA’,‘SU.PA’,
‘GLE.PA’,‘HO.PA’,‘TTE.PA’,‘VIE.PA’,‘DG.PA’,‘WLN.PA’,‘ERF.PA’,‘CS.PA’,
‘ALO.PA’,‘MT.AS’,‘STM.PA’,
# SBF 120
‘ML.PA’,‘SPIE.PA’,‘COFA.PA’,‘CNP.PA’,‘ELIS.PA’,‘GTT.PA’,‘MMB.PA’,‘UBI.PA’,
# DAX
‘SAP.DE’,‘SIE.DE’,‘ALV.DE’,‘MUV2.DE’,‘BAYN.DE’,‘BMW.DE’,‘MBG.DE’,
‘RWE.DE’,‘DTE.DE’,‘BAS.DE’,‘IFX.DE’,‘MRK.DE’,‘VOW3.DE’,‘ADS.DE’,
‘DB1.DE’,‘DHL.DE’,‘EOAN.DE’,‘FRE.DE’,‘HEI.DE’,‘HEN3.DE’,‘LIN.DE’,
‘VNA.DE’,‘ZAL.DE’,‘DHER.DE’,‘BEI.DE’,‘ENR.DE’,‘PAH3.DE’,‘SRT3.DE’,
‘EVK.DE’,‘HAB.DE’,‘HLAG.DE’,‘KGX.DE’,‘LEG.DE’,‘PSM.DE’,‘PUM.DE’,
‘SGL.DE’,‘TUI1.DE’,‘WAF.DE’,‘WCH.DE’,
# Benelux
‘ASML.AS’,‘INGA.AS’,‘PHIA.AS’,‘HEIA.AS’,‘NN.AS’,‘WKL.AS’,
‘AKZA.AS’,‘UNA.AS’,‘IMCD.AS’,‘BESI.AS’,‘AMG.AS’,‘AALB.AS’,‘ADYEN.AS’,
‘UCB.BR’,‘ABI.BR’,‘KBC.BR’,‘SOLB.BR’,‘COLR.BR’,‘GBLB.BR’,‘WDP.BR’,
# Ibérie
‘ITX.MC’,‘BBVA.MC’,‘SAN.MC’,‘TEF.MC’,‘IBE.MC’,‘REP.MC’,‘AMS.MC’,
‘ELE.MC’,‘MAP.MC’,‘AENA.MC’,‘ACS.MC’,‘GRF.MC’,‘IAG.MC’,‘NTGY.MC’,
‘EDP.LS’,‘EDPR.LS’,‘GALP.LS’,‘JMT.LS’,‘NOS.LS’,
# Italie
‘ENI.MI’,‘ENEL.MI’,‘ISP.MI’,‘UCG.MI’,‘LDO.MI’,‘MB.MI’,‘PRY.MI’,
‘A2A.MI’,‘BAMI.MI’,‘G.MI’,‘MONC.MI’,‘RACE.MI’,‘SRG.MI’,
# Suisse
‘NESN.SW’,‘NOVN.SW’,‘ROG.SW’,‘ABBN.SW’,‘ZURN.SW’,‘UBSG.SW’,‘ADEN.SW’,
‘ALC.SW’,‘GIVN.SW’,‘SIKA.SW’,‘SLHN.SW’,‘CFR.SW’,‘EMMN.SW’,
‘GEBN.SW’,‘KNIN.SW’,‘PGHN.SW’,‘SCMN.SW’,‘TEMN.SW’,
# Nordiques
‘ORSTED.CO’,‘NOVO-B.CO’,‘DSV.CO’,‘CARL-B.CO’,‘MAERSK-B.CO’,
‘VOLV-B.ST’,‘ERIC-B.ST’,‘SEB-A.ST’,‘SWED-A.ST’,‘SHB-A.ST’,
‘ATCO-A.ST’,‘ASSA-B.ST’,‘ALFA.ST’,‘SAND.ST’,‘SKA-B.ST’,‘SKF-B.ST’,
‘TEL2-B.ST’,‘TELIA.ST’,‘DNB.OL’,‘EQNR.OL’,‘TEL.OL’,‘MOWI.OL’,
‘STERV.HE’,‘NESTE.HE’,‘KNEBV.HE’,
# UK
‘RELX’,‘BP.L’,‘SHEL.L’,‘HSBA.L’,‘LLOY.L’,‘AZN.L’,‘GSK.L’,‘ULVR.L’,
]

ACTIONS_USA = [
‘AAPL’,‘MSFT’,‘GOOGL’,‘AMZN’,‘META’,‘NVDA’,‘TSLA’,‘AVGO’,‘ORCL’,‘ADBE’,
‘CRM’,‘AMD’,‘QCOM’,‘TXN’,‘INTC’,‘IBM’,‘CSCO’,‘NOW’,‘INTU’,‘AMAT’,
‘MU’,‘KLAC’,‘LRCX’,‘ADI’,‘MCHP’,‘CDNS’,‘SNPS’,‘FTNT’,‘PANW’,‘CRWD’,
‘NET’,‘SNOW’,‘PLTR’,‘UBER’,‘TTD’,‘VEEV’,‘WDAY’,‘PAYC’,‘EPAM’,
‘JNJ’,‘UNH’,‘ABT’,‘TMO’,‘MRK’,‘ABBV’,‘PFE’,‘BMY’,‘AMGN’,‘LLY’,
‘MDT’,‘ISRG’,‘SYK’,‘EW’,‘BDX’,‘CI’,‘CVS’,‘HUM’,‘CNC’,‘MOH’,
‘IQV’,‘LH’,‘DGX’,‘VRTX’,‘REGN’,‘BIIB’,‘GILD’,‘MRNA’,‘ZTS’,‘IDXX’,
‘HOLX’,‘MCK’,‘CAH’,‘GEHC’,‘RMD’,‘BAX’,‘BSX’,‘ZBH’,
‘JPM’,‘BAC’,‘WFC’,‘GS’,‘MS’,‘C’,‘BLK’,‘SPGI’,‘MCO’,‘ICE’,
‘CME’,‘AXP’,‘V’,‘MA’,‘COF’,‘BX’,‘KKR’,‘APO’,‘ARES’,
‘ARCC’,‘MAIN’,‘BEN’,‘TROW’,‘STT’,‘NTRS’,‘FITB’,‘RF’,‘HBAN’,‘KEY’,
‘PG’,‘KO’,‘PEP’,‘MDLZ’,‘KHC’,‘GIS’,‘CAG’,‘MKC’,‘HRL’,
‘WMT’,‘COST’,‘TGT’,‘HD’,‘LOW’,‘NKE’,‘MCD’,‘SBUX’,‘YUM’,‘DRI’,
‘CMG’,‘DPZ’,‘QSR’,‘PM’,‘MO’,‘BTI’,‘EL’,‘CL’,‘CHD’,‘CLX’,
‘XOM’,‘CVX’,‘COP’,‘SLB’,‘EOG’,‘MPC’,‘PSX’,‘VLO’,‘HAL’,‘BKR’,
‘OXY’,‘DVN’,‘EQT’,‘ENB’,‘TRP’,
‘GE’,‘BA’,‘CAT’,‘HON’,‘RTX’,‘LMT’,‘NOC’,‘GD’,‘ITW’,‘EMR’,
‘ETN’,‘PH’,‘DOV’,‘ROK’,‘AME’,‘UPS’,‘FDX’,‘DAL’,‘UAL’,‘LUV’,
‘UNP’,‘CSX’,‘NSC’,‘CP’,‘DE’,‘AGCO’,
‘NEE’,‘DUK’,‘SO’,‘D’,‘AEP’,‘EXC’,‘ED’,‘WEC’,‘CMS’,
‘VZ’,‘T’,‘TMUS’,‘CMCSA’,‘DIS’,‘NFLX’,
‘PLD’,‘AMT’,‘CCI’,‘EQIX’,‘PSA’,‘O’,‘WELL’,‘DLR’,‘AVB’,‘EQR’,
‘NNN’,‘VICI’,‘IRM’,‘REXR’,‘EXR’,‘WPC’,‘STAG’,
]

PEA_SET = set([
‘AIR.PA’,‘AI.PA’,‘ALO.PA’,‘ATO.PA’,‘BNP.PA’,‘EN.PA’,‘CAP.PA’,‘CA.PA’,
‘ACA.PA’,‘DSY.PA’,‘ENGI.PA’,‘EL.PA’,‘RMS.PA’,‘KER.PA’,‘LR.PA’,‘OR.PA’,
‘MC.PA’,‘ORA.PA’,‘PUB.PA’,‘RI.PA’,‘RNO.PA’,‘SAF.PA’,‘SGO.PA’,‘SAN.PA’,
‘SU.PA’,‘GLE.PA’,‘STM.PA’,‘HO.PA’,‘TTE.PA’,‘VIE.PA’,‘DG.PA’,‘WLN.PA’,
‘ERF.PA’,‘CS.PA’,‘ML.PA’,‘SPIE.PA’,‘COFA.PA’,‘CNP.PA’,‘ELIS.PA’,
‘GTT.PA’,‘MMB.PA’,‘UBI.PA’,‘MT.AS’,
‘SAP.DE’,‘SIE.DE’,‘ALV.DE’,‘MUV2.DE’,‘BAYN.DE’,‘BMW.DE’,‘MBG.DE’,
‘RWE.DE’,‘DTE.DE’,‘BAS.DE’,‘IFX.DE’,‘MRK.DE’,‘VOW3.DE’,‘ADS.DE’,
‘DB1.DE’,‘DHL.DE’,‘EOAN.DE’,‘FRE.DE’,‘HEI.DE’,‘HEN3.DE’,‘LIN.DE’,
‘VNA.DE’,‘ZAL.DE’,‘DHER.DE’,‘BEI.DE’,‘SGL.DE’,‘SRT3.DE’,‘EVK.DE’,
‘ASML.AS’,‘INGA.AS’,‘PHIA.AS’,‘HEIA.AS’,‘NN.AS’,‘WKL.AS’,‘AKZA.AS’,
‘UNA.AS’,‘IMCD.AS’,‘BESI.AS’,‘AMG.AS’,‘ADYEN.AS’,
‘UCB.BR’,‘ABI.BR’,‘KBC.BR’,‘SOLB.BR’,‘COLR.BR’,‘GBLB.BR’,‘WDP.BR’,
‘ITX.MC’,‘BBVA.MC’,‘SAN.MC’,‘TEF.MC’,‘IBE.MC’,‘REP.MC’,‘AMS.MC’,
‘ELE.MC’,‘MAP.MC’,‘AENA.MC’,‘ACS.MC’,‘GRF.MC’,‘IAG.MC’,‘NTGY.MC’,
‘EDP.LS’,‘EDPR.LS’,‘GALP.LS’,‘JMT.LS’,
‘ENI.MI’,‘ENEL.MI’,‘ISP.MI’,‘UCG.MI’,‘LDO.MI’,‘MB.MI’,‘PRY.MI’,
‘NESN.SW’,‘NOVN.SW’,‘ROG.SW’,‘ABBN.SW’,‘ZURN.SW’,‘UBSG.SW’,‘ADEN.SW’,
‘ALC.SW’,‘GIVN.SW’,‘SIKA.SW’,‘CFR.SW’,‘GEBN.SW’,‘KNIN.SW’,
‘ORSTED.CO’,‘NOVO-B.CO’,‘DSV.CO’,‘CARL-B.CO’,
‘VOLV-B.ST’,‘ERIC-B.ST’,‘SEB-A.ST’,‘SWED-A.ST’,‘ASSA-B.ST’,
‘DNB.OL’,‘EQNR.OL’,‘TEL.OL’,‘MOWI.OL’,
‘STERV.HE’,‘NESTE.HE’,‘KNEBV.HE’,
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
per=info.get(‘trailingPE’) or info.get(‘forwardPE’)
pbv=info.get(‘priceToBook’)
ev=info.get(‘enterpriseToEbitda’)
if per and per>0: p+=40 if per<=10 else 28 if per<=15 else 15 if per<=PER_MAX else 0
if pbv and pbv>0: p+=30 if pbv<=1 else 20 if pbv<=2 else 10 if pbv<=PBV_MAX else 0
if ev  and ev>0:  p+=30 if ev<=8  else 20 if ev<=12 else 10 if ev<=15 else 0
return min(100,p)

def score_growth(info):
p=0
roe=info.get(‘returnOnEquity’); eg=info.get(‘earningsGrowth’)
peg=info.get(‘pegRatio’)
if roe: rp=roe*100; p+=35 if rp>=25 else 25 if rp>=15 else 12 if rp>=ROE_MIN else 0
if eg:  ep=eg*100;  p+=40 if ep>=25 else 28 if ep>=15 else 15 if ep>0 else 0
if peg and peg>0:   p+=25 if peg<=1 else 15 if peg<=1.5 else 8 if peg<=2.5 else 0
return min(100,p)

def score_div(info):
p=0
yld=info.get(‘dividendYield’); po=info.get(‘payoutRatio’)
if yld: yp=yld*100; p+=60 if yp>=6 else 45 if yp>=4 else 30 if yp>=3 else 15 if yp>0 else 0
if po and 0<po<0.8: p+=20
elif po and po<=1:  p+=10
return min(100,p)

def score_tech(hist):
if hist is None or len(hist)<30: return 0,{}
try:
c=hist[‘Close’].dropna(); h=hist[‘High’].dropna(); l=hist[‘Low’].dropna()
if len(c)<30: return 0,{}
r=calc_rsi(c).iloc[-1]
ml,ms,mh=calc_macd(c); mh_v=mh.iloc[-1]; ml_v=ml.iloc[-1]; ms_v=ms.iloc[-1]
cc=calc_cci(h,l,c).iloc[-1]
sk,_=calc_stoch(h,l,c); st=sk.iloc[-1]
ax=calc_adx(h,l,c).iloc[-1]
bb=calc_boll(c).iloc[-1]
px=float(c.iloc[-1])
mms={f’MM{p}’:float(c.rolling(p).mean().iloc[-1]) for p in [20,50,200] if len(c)>=p}
pts=0; sigs=[]
if not np.isnan(r):
if r<=30:           pts+=28; sigs.append(f’RSI {r:.0f} 🟢’)
elif r<=RSI_SURVENTE:pts+=18; sigs.append(f’RSI {r:.0f}’)
elif r<=50:          pts+=8
if not np.isnan(mh_v):
if mh_v>0 and ml_v>ms_v: pts+=20; sigs.append(‘MACD 🟢’)
elif mh_v>0:              pts+=12; sigs.append(‘MACD’)
elif mh_v>-0.1:           pts+=5
if not np.isnan(cc):
if cc<CCI_SURVENTE:  pts+=18; sigs.append(f’CCI {cc:.0f} 🟢’)
elif 50<=cc<=150:    pts+=12; sigs.append(f’CCI {cc:.0f}’)
elif 0<cc<50:        pts+=6
if not np.isnan(st):
if st<=20:           pts+=14; sigs.append(f’Stoch {st:.0f} 🟢’)
elif st<=STOCH_SURVENTE:pts+=9; sigs.append(f’Stoch {st:.0f}’)
above=sum(1 for v in mms.values() if px>v)
if above==len(mms)>0: pts+=15; sigs.append(‘MM ✅’)
elif above>=2:         pts+=8
if not np.isnan(ax) and ax>=ADX_MIN: pts+=5; sigs.append(f’ADX {ax:.0f}’)
det={
‘RSI’:   round(float(r),1)    if not np.isnan(r)    else None,
‘MACD’:  round(float(mh_v),3) if not np.isnan(mh_v) else None,
‘CCI’:   round(float(cc),1)   if not np.isnan(cc)   else None,
‘Stoch’: round(float(st),1)   if not np.isnan(st)   else None,
‘ADX’:   round(float(ax),1)   if not np.isnan(ax)   else None,
‘BB’:    round(float(bb),2)   if not np.isnan(bb)   else None,
‘Signaux’: ’ | ‘.join(sigs),
}
for k,v in mms.items():
det[k]=f’{((px/v)-1)*100:+.1f}%’
return min(100,pts),det
except: return 0,{}

# ══════════════════════════════════════════════════════════════

# SCAN PRINCIPAL

# ══════════════════════════════════════════════════════════════

tickers = list(dict.fromkeys(ACTIONS_EUROPE + ACTIONS_USA))
d_fin = datetime.today()
d_deb = d_fin - timedelta(days=JOURS_HISTORIQUE)
print(f’🔍 Scan de {len(tickers)} actions…’)

# Téléchargement groupé

BATCH=50; all_hist={}
for i in range(0,len(tickers),BATCH):
batch=tickers[i:i+BATCH]
try:
data=yf.download(batch,start=d_deb.strftime(’%Y-%m-%d’),
end=d_fin.strftime(’%Y-%m-%d’),
group_by=‘ticker’,auto_adjust=True,
progress=False,threads=True)
for tk in batch:
try:
all_hist[tk]=data[tk] if len(batch)>1 and tk in data.columns.get_level_values(0) else (data if len(batch)==1 else None)
except: all_hist[tk]=None
except:
for tk in batch: all_hist[tk]=None
print(f’  Téléchargé {min(i+BATCH,len(tickers))}/{len(tickers)}’)

resultats=[]
for idx,ticker in enumerate(tickers):
try:
info=yf.Ticker(ticker).info or {}
nom=info.get(‘longName’) or info.get(‘shortName’) or ticker
secteur=info.get(‘sector’) or ‘N/A’
pays=(info.get(‘country’) or ‘’)[:2].upper() or ‘?’
devise=info.get(‘currency’) or (‘EUR’ if any(x in ticker for x in [’.PA’,’.DE’,’.AS’,’.BR’,’.MC’,’.MI’,’.SW’,’.CO’,’.ST’,’.OL’,’.HE’,’.LS’,’.L’]) else ‘USD’)
hist=all_hist.get(ticker)
prix_hist=float(hist[‘Close’].iloc[-1]) if hist is not None and len(hist)>0 else None
prix=info.get(‘currentPrice’) or info.get(‘regularMarketPrice’) or prix_hist

```
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
```

resultats.sort(key=lambda x: x[‘score’], reverse=True)
resultats=resultats[:TOP_N]
print(f’✅ {len(resultats)} opportunités trouvées’)

# ══════════════════════════════════════════════════════════════

# GÉNÉRATION HTML — sans f-string pour éviter les conflits {}

# ══════════════════════════════════════════════════════════════

import json
data_json = json.dumps(resultats, ensure_ascii=False, default=str)
date_str  = d_fin.strftime(’%d/%m/%Y a %H:%M’)
nb_scanned = len(tickers)

# Le JavaScript est dans une variable Python normale (pas f-string)

# Les accolades JS sont donc simples, pas doublées

JS = “””

<script>
const PWD = '1510';

function checkPwd() {
  var v = document.getElementById('pwd').value;
  var login = document.getElementById('login');
  var app = document.getElementById('app');
  if (v === PWD) {
    login.style.display = 'none';
    app.style.display = 'block';
    render();
  } else {
    document.getElementById('err').textContent = 'Code incorrect';
    document.getElementById('pwd').value = '';
  }
}

const RAW = DATA_PLACEHOLDER;
let mode='combine', curTab='all', sortCol='score', sortDir=-1;

const sects=['all',...new Set(RAW.map(r=>r.secteur).filter(Boolean))].sort();
const se=document.getElementById('sect');
sects.forEach(s=>{ const o=document.createElement('option'); o.value=s; o.text=s==='all'?'Tous secteurs':s; se.appendChild(o); });

function nextUpdate(){
  const now=new Date(); const nxt=new Date();
  nxt.setDate(now.getDate()+(now.getHours()>=7?1:0));
  nxt.setHours(7,0,0,0);
  const diff=nxt-now; const h=Math.floor(diff/3600000); const m=Math.floor((diff%3600000)/60000);
  document.getElementById('nextUpdate').textContent='Prochain scan dans '+h+'h'+m+'m';
}
nextUpdate(); setInterval(nextUpdate,60000);

const DESCS={
  'combine':'Mode Combine - Score = Fondamentaux + Technique.',
  'fond':   'Mode Fondamental - PER, ROE, dividende, croissance.',
  'tech':   'Mode Technique - RSI, MACD, CCI, Stochastique, Moyennes mobiles.',
};
function setMode(m){
  mode=m;
  ['combine','fond','tech'].forEach(x=>document.getElementById('m-'+x).className='mode-btn'+(x===m?' on':''));
  document.getElementById('mode-desc').textContent=DESCS[m];
  render();
}
setMode('combine');

function setTab(t){
  curTab=t;
  ['all','pea'].forEach(x=>document.getElementById('tab-'+x).className='tab'+(x===t?' on':''));
  render();
}

function getScore(r){
  if(mode==='fond') return Math.round(r.sv*0.4+r.sg*0.35+r.sd*0.25);
  if(mode==='tech') return r.st;
  return r.score;
}

function filter(){
  const q=(document.getElementById('q').value||'').toLowerCase();
  const mkt=document.getElementById('mkt').value;
  const sec=document.getElementById('sect').value;
  const ms=parseFloat(document.getElementById('ms').value)||0;
  return RAW.filter(r=>{
    if(getScore(r)<ms) return false;
    if(mkt==='eu'&&r.pays==='US') return false;
    if(mkt==='us'&&r.pays!=='US') return false;
    if(sec!=='all'&&r.secteur!==sec) return false;
    if(q&&!r.ticker.toLowerCase().includes(q)&&!r.nom.toLowerCase().includes(q)) return false;
    return true;
  }).sort((a,b)=>sortDir*(getScore(b)-getScore(a)));
}

function n(v,suf,dec){
  dec=dec||1; suf=suf||'';
  return v!==null&&v!==undefined?'<span style="font-family:monospace;font-size:11px">'+parseFloat(v).toFixed(dec)+suf+'</span>':'<span style="color:#2a3a52">-</span>';
}
function barHtml(v){
  var pct=Math.min(v||0,100); var c=pct>=70?'#10b981':pct>=50?'#f59e0b':'#64748b';
  return '<div class="bar-w"><div class="bar"><div class="bar-f" style="width:'+pct+'%;background:'+c+'"></div></div><span style="font-family:monospace;font-size:11px;color:'+c+';font-weight:700">'+Math.round(pct)+'</span></div>';
}
function chips(v){
  if(!v)return '-';
  return v.split('|').map(s=>s.trim()).filter(Boolean).map(s=>'<span class="chip">'+s+'</span>').join('');
}

var C_FOND=['ticker','prix','per','pbv','roe','marge','div','bpa','ev','sv','sg','sd','score'];
var C_TECH=['ticker','prix','rsi','macd','cci','stoch','adx','mm20','mm50','mm200','st','signaux'];
var C_COMB=['ticker','prix','per','roe','div','rsi','macd','cci','mm200','sv','st','score','signaux'];
var C_LABELS={ticker:'Action',prix:'Cours',per:'PER',pbv:'P/Book',roe:'ROE%',marge:'Marge%',
  div:'Div%',bpa:'BPA%',ev:'EV/EBITDA',sv:'Score V',sg:'Score G',sd:'Score D',
  st:'Score T',score:'Score',rsi:'RSI',macd:'MACD',cci:'CCI',stoch:'Stoch.',
  adx:'ADX',mm20:'vs MM20',mm50:'vs MM50',mm200:'vs MM200',signaux:'Signaux'};

function cellHtml(col,r){
  var v=r[col];
  if(col==='ticker') return '<div><span class="tk">'+r.ticker+'</span>'+(r.pea?'<span class="pea-tag">PEA</span>':'')+'</div><div class="nm">'+r.nom+'</div><div class="ct">'+r.pays+' - '+r.secteur+'</div>';
  if(col==='prix') return v!==null?'<span style="font-family:monospace">'+(r.devise==='USD'?'$':'€')+parseFloat(v).toFixed(2)+'</span>':'-';
  if(col==='per') return v?'<span style="font-family:monospace;font-size:11px;color:'+(v<=12?'#10b981':v<=25?'#f59e0b':'#ef4444')+'">'+parseFloat(v).toFixed(1)+'x</span>':'-';
  if(col==='pbv') return n(v,'x',2);
  if(col==='roe') return v?'<span style="font-family:monospace;font-size:11px;color:'+(v>=20?'#10b981':v>=10?'#f59e0b':'#94a3b8')+'">'+parseFloat(v).toFixed(1)+'%</span>':'-';
  if(col==='marge') return n(v,'%');
  if(col==='div') return v?'<span style="font-family:monospace;color:'+(v>=4?'#10b981':'#94a3b8')+'">'+parseFloat(v).toFixed(1)+'%</span>':'-';
  if(col==='bpa') return n(v,'%');
  if(col==='ev') return n(v,'x');
  if(col==='sv'||col==='sg'||col==='sd'||col==='st') return v!==null?barHtml(v):'-';
  if(col==='score') return v!==null?barHtml(getScore(r)):'-';
  if(col==='rsi') return v!==null?'<span style="font-family:monospace;font-size:11px;font-weight:'+(v<=45||v>=70?700:400)+';color:'+(v<=30?'#10b981':v<=45?'#f59e0b':v>=70?'#ef4444':'#94a3b8')+'">'+v+'</span>':'-';
  if(col==='macd') return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v>0?'#10b981':v<0?'#ef4444':'#94a3b8')+'">'+parseFloat(v).toFixed(2)+'</span>':'-';
  if(col==='cci') return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v<-100?'#10b981':v>150?'#ef4444':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='stoch') return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v<=30?'#10b981':v>=70?'#ef4444':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='adx') return v!==null?'<span style="font-family:monospace;font-size:11px;color:'+(v>=25?'#00d4aa':'#94a3b8')+'">'+Math.round(v)+'</span>':'-';
  if(col==='mm20'||col==='mm50'||col==='mm200') return v?'<span style="font-family:monospace;font-size:11px;color:'+(v.startsWith('+')?'#10b981':'#ef4444')+'">'+v+'</span>':'-';
  if(col==='signaux') return chips(v);
  return v!==null&&v!==undefined?String(v):'-';
}

function render(){
  var data=filter();
  var allN=data.length;
  var peaD=data.filter(r=>r.pea===1);
  document.getElementById('b-all').textContent=allN;
  document.getElementById('b-pea').textContent=peaD.length;
  var shown=curTab==='pea'?peaD:data;
  var s70=shown.filter(r=>getScore(r)>=70).length;
  var s50=shown.filter(r=>getScore(r)>=50&&getScore(r)<70).length;
  var avgD=shown.length?(shown.reduce((s,r)=>s+(r.div||0),0)/shown.length).toFixed(1)+'%':'0%';
  var rsiLow=shown.filter(r=>r.rsi&&r.rsi<=45).length;
  document.getElementById('stats').innerHTML=[
    ['Total',shown.length,'#00d4aa'],['Score 70+',s70,'#10b981'],
    ['Score 50-69',s50,'#f59e0b'],['Div. moy.',avgD,'#8b5cf6'],['RSI<=45',rsiLow,'#3b82f6'],
  ].map(function(x){ return '<div class="stat"><div class="sl">'+x[0]+'</div><div class="sv" style="color:'+x[2]+'">'+x[1]+'</div></div>'; }).join('');
  var cols=mode==='fond'?C_FOND:mode==='tech'?C_TECH:C_COMB;
  document.getElementById('thead').innerHTML='<tr>'+cols.map(function(c){ return '<th class="'+(sortCol===c?'sorted':'')+'" onclick="doSort(\''+c+'\')">'+( C_LABELS[c]||c)+(sortCol===c?(sortDir>0?' ^':' v'):'')+' </th>'; }).join('')+'</tr>';
  var empty=document.getElementById('empty');
  if(!shown.length){ document.getElementById('tbody').innerHTML=''; empty.style.display='block'; return; }
  empty.style.display='none';
  document.getElementById('tbody').innerHTML=shown.map(function(r){ return '<tr>'+cols.map(function(c){ return '<td>'+cellHtml(c,r)+'</td>'; }).join('')+'</tr>'; }).join('');
}

function doSort(col){
  if(sortCol===col)sortDir*=-1; else{sortCol=col;sortDir=-1;}
  render();
}
</script>

“””

# CSS séparé (pas de f-string non plus)

CSS = “””

<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0e1a;color:#e2e8f0;font-family:system-ui,sans-serif;min-height:100vh}
#login{position:fixed;inset:0;background:#0a0e1a;z-index:100;display:flex;align-items:center;justify-content:center}
.login-box{background:#111827;border:1px solid #1e2d45;border-radius:16px;padding:40px;width:320px;text-align:center}
.login-box h2{font-family:monospace;font-size:22px;color:#00d4aa;margin-bottom:8px}
.login-box p{font-size:12px;color:#64748b;margin-bottom:24px}
.login-box input{width:100%;background:#0d1520;border:1px solid #1e2d45;color:#e2e8f0;font-size:18px;padding:14px;border-radius:9px;outline:none;text-align:center;letter-spacing:6px;font-family:monospace;margin-bottom:14px;display:block}
.login-box button{width:100%;background:#00d4aa;color:#000;font-weight:800;font-size:16px;padding:14px;border:none;border-radius:9px;cursor:pointer;font-family:monospace}
.login-err{color:#ef4444;font-size:12px;margin-top:8px;min-height:18px}
#app{display:none}
.hdr{background:#111827;border-bottom:1px solid #1e2d45;padding:14px 18px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;position:sticky;top:0;z-index:10}
.hdr-title{font-family:monospace;font-size:18px;font-weight:900;color:#00d4aa}
.hdr-sub{font-size:10px;color:#64748b;margin-top:2px}
.update-badge{font-family:monospace;font-size:10px;color:#64748b;background:#0d1520;padding:5px 10px;border-radius:6px;border:1px solid #1e2d45}
.wrap{max-width:1500px;margin:0 auto;padding:14px}
.mode-row{display:flex;gap:8px;margin-bottom:12px}
.mode-btn{flex:1;padding:10px 6px;border-radius:10px;border:2px solid #1e2d45;background:#0d1520;cursor:pointer;text-align:center}
.mode-btn.on{border-color:#00d4aa;background:#00d4aa18}
.mode-btn .ico{font-size:18px;margin-bottom:3px}
.mode-btn .lbl{font-size:10px;font-weight:700;color:#64748b;font-family:monospace}
.mode-btn.on .lbl{color:#00d4aa}
.mode-desc{background:#0d1520;border:1px solid #1e2d45;border-radius:9px;padding:9px 13px;font-size:11px;color:#64748b;line-height:1.6;margin-bottom:12px}
.filters{background:#111827;border:1px solid #1e2d45;border-radius:11px;padding:12px 14px;display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end;margin-bottom:12px}
.fg{display:flex;flex-direction:column;gap:4px}
.fg label{font-size:9px;color:#64748b;text-transform:uppercase;letter-spacing:.8px}
select,input[type=text],input[type=number]{background:#0d1520;border:1px solid #1e2d45;color:#e2e8f0;font-size:12px;padding:7px 9px;border-radius:7px;outline:none;font-family:inherit}
.tabs{display:flex;gap:5px;margin-bottom:10px}
.tab{padding:7px 16px;border-radius:8px;border:1px solid #1e2d45;background:#0d1520;color:#64748b;cursor:pointer;font-size:11px;font-weight:700;font-family:monospace;display:flex;align-items:center;gap:6px}
.tab.on{border-color:#00d4aa;background:#111827;color:#e2e8f0}
.badge{background:#1e2d45;color:#64748b;font-size:9px;font-weight:700;padding:1px 6px;border-radius:20px}
.tab.on .badge{background:#00d4aa;color:#000}
.stats{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.stat{background:#111827;border:1px solid #1e2d45;border-radius:9px;padding:9px 13px;flex:1;min-width:80px}
.stat .sl{font-size:8px;color:#64748b;text-transform:uppercase;letter-spacing:.6px}
.stat .sv{font-family:monospace;font-size:18px;font-weight:900;margin-top:2px}
.tbl-wrap{background:#111827;border:1px solid #1e2d45;border-radius:12px;overflow:hidden}
.tbl-scroll{overflow-x:auto;max-height:60vh;overflow-y:auto}
table{width:100%;border-collapse:collapse;font-size:11px}
thead th{background:#0d1520;padding:8px 9px;font-size:8px;font-family:monospace;text-transform:uppercase;letter-spacing:.7px;color:#64748b;cursor:pointer;border-bottom:1px solid #1e2d45;white-space:nowrap;position:sticky;top:0;z-index:2;text-align:left;user-select:none}
thead th:hover{color:#00d4aa}
thead th.sorted{color:#00d4aa}
tbody td{padding:8px 9px;border-bottom:1px solid #0d1520;vertical-align:middle;white-space:nowrap}
tbody tr:nth-child(even) td{background:#0a1020}
tbody tr:hover td{background:#152030!important}
.tk{font-family:monospace;font-weight:700;color:#00d4aa;font-size:12px}
.pea-tag{font-size:8px;padding:1px 4px;border-radius:3px;background:#00d4aa18;color:#00d4aa;border:1px solid #00d4aa33;margin-left:3px}
.nm{font-size:10px;color:#64748b;max-width:130px;overflow:hidden;text-overflow:ellipsis}
.ct{font-size:9px;color:#2a4060}
.bar-w{display:flex;align-items:center;gap:5px}
.bar{width:40px;height:4px;border-radius:2px;background:#1e2d45;overflow:hidden}
.bar-f{height:100%;border-radius:2px}
.chip{display:inline-block;font-size:9px;font-weight:700;padding:2px 5px;border-radius:20px;margin-right:2px;font-family:monospace;background:#00d4aa18;color:#00d4aa;border:1px solid #00d4aa33}
.empty{text-align:center;padding:50px 20px;color:#64748b}
.disc{margin-top:10px;padding:9px 14px;border:1px solid #f59e0b33;border-radius:9px;background:#f59e0b06;font-size:10px;color:#64748b;line-height:1.6}
</style>

“””

# Construction HTML par concaténation — zéro f-string sur le JS/CSS

html  = ‘<!DOCTYPE html>\n<html lang="fr">\n<head>\n’
html += ‘<meta charset="UTF-8">\n’
html += ‘<meta name="viewport" content="width=device-width, initial-scale=1.0">\n’
html += ‘<title>Stock Screener</title>\n’
html += CSS
html += ‘</head>\n<body>\n’

# Login

html += ‘’’

<div id="login">
  <div class="login-box">
    <div style="font-size:40px;margin-bottom:10px">📡</div>
    <h2>STOCK SCREENER</h2>
    <p>Entrez votre code d'acces</p>
    <input type="password" id="pwd" placeholder="••••" autocomplete="off">
    <button onclick="checkPwd()">Acceder</button>
    <div class="login-err" id="err"></div>
  </div>
</div>
'''

# App

html += ‘<div id="app">\n’
html += ’  <div class="hdr">\n’
html += ’    <div><div class="hdr-title">📡 STOCK SCREENER</div>\n’
html += ’    <div class="hdr-sub">Yahoo Finance · ’ + str(nb_scanned) + ’ actions · ’ + date_str + ‘</div></div>\n’
html += ’    <span class="update-badge" id="nextUpdate"></span>\n’
html += ’  </div>\n’
html += ’  <div class="wrap">\n’

# Mode buttons

html += ‘’’
<div class="mode-row">
<div class="mode-btn on" id="m-combine" onclick="setMode('combine')"><div class="ico">🎯</div><div class="lbl">Fond. + Tech.</div></div>
<div class="mode-btn"   id="m-fond"    onclick="setMode('fond')">   <div class="ico">📊</div><div class="lbl">Fondamental</div></div>
<div class="mode-btn"   id="m-tech"    onclick="setMode('tech')">   <div class="ico">📈</div><div class="lbl">Technique</div></div>
</div>
<div class="mode-desc" id="mode-desc"></div>
‘’’

# Filters

html += ‘<div class="filters">\n’
html += ’  <div class="fg"><label>Recherche</label><input type="text" id="q" placeholder="Ticker..." oninput="render()" style="width:130px"></div>\n’
html += ’  <div class="fg"><label>Marche</label><select id="mkt" onchange="render()"><option value="all">Europe + USA</option><option value="eu">Europe</option><option value="us">USA</option></select></div>\n’
html += ’  <div class="fg"><label>Secteur</label><select id="sect" onchange="render()"></select></div>\n’
html += ’  <div class="fg"><label>Score min</label><input type="number" id="ms" value="' + str(SCORE_MIN) + '" min="0" max="100" step="5" onchange="render()" style="width:65px"></div>\n’
html += ‘</div>\n’

# Tabs

html += ‘’’
<div class="tabs">
<div class="tab on" id="tab-all" onclick="setTab('all')">🌍 Toutes <span class="badge" id="b-all">0</span></div>
<div class="tab"    id="tab-pea" onclick="setTab('pea')">🇫🇷 PEA <span class="badge" id="b-pea">0</span></div>
</div>
<div class="stats" id="stats"></div>
<div class="tbl-wrap"><div class="tbl-scroll">
<table><thead id="thead"></thead><tbody id="tbody"></tbody></table>
<div class="empty" id="empty" style="display:none">Aucun resultat</div>
</div></div>
<div class="disc">Donnees Yahoo Finance a titre indicatif. Ne constitue pas un conseil en investissement.</div>

  </div>
</div>
'''

# Injection des données puis JS

html += ’<script>const RAW_DATA = ’ + data_json + ‘;</script>\n’
html += JS.replace(‘const RAW = DATA_PLACEHOLDER;’, ‘const RAW = RAW_DATA;’)
html += ‘\n</body>\n</html>’

# Écriture

out = Path(‘docs/index.html’)
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding=‘utf-8’)
print(‘✅ HTML genere : ’ + str(out) + ’ (’ + str(len(resultats)) + ’ opportunites)’)
