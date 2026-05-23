# 📡 Stock Screener — PEA & Compte-Titres

Screener automatique d’actions — données Yahoo Finance — scan quotidien via GitHub Actions — résultats sur GitHub Pages.

## 🚀 Installation (5 minutes)

### 1. Activer GitHub Pages

- **Settings → Pages**
- Source : **Deploy from a branch**
- Branch : **main** / dossier : **docs**
- Sauvegarder

Votre URL : `https://VOTRE-USERNAME.github.io/Screener`

### 2. Définir le mot de passe d’accès

- **Settings → Secrets and variables → Actions → New repository secret**  
  *(URL directe : `https://github.com/VOTRE-USERNAME/Screener/settings/secrets/actions`)*
- Nom : `SCREENER_PASSWORD`
- Valeur : votre mot de passe choisi

Le mot de passe est haché en SHA-256 côté Python à la compilation — il n’apparaît jamais en clair dans le HTML.  
Si le secret n’est pas défini, le mot de passe par défaut est `changeme2025`.

### 3. Lancer le premier scan

- **Actions → Stock Screener Scan → Run workflow**
- Durée : 30 à 60 min selon le nombre d’actions
- La page se met à jour automatiquement après le scan

## 🔄 Scan automatique

- **Lundi–Vendredi à 7h00** (heure de Paris)
- Ou manuellement via **Actions → Run workflow**

## ⚙️ Configuration dans generate.py

|Variable         |Défaut|Description                                     |
|-----------------|------|------------------------------------------------|
|`SCORE_MIN`      |35    |Score minimum pour apparaître dans les résultats|
|`TOP_N`          |300   |Nombre maximum d’actions retournées             |
|`POIDS_VALUE`    |30    |Poids du score valorisation (%)                 |
|`POIDS_GROWTH`   |25    |Poids du score croissance & qualité (%)         |
|`POIDS_DIVIDENDE`|15    |Poids du score dividende (%)                    |
|`POIDS_TECHNIQUE`|30    |Poids du score technique (%)                    |
|`RSI_SURVENTE`   |45    |Seuil RSI zone survente                         |
|`CCI_SURVENTE`   |-100  |Seuil CCI zone survente                         |
|`STOCH_SURVENTE` |30    |Seuil Stochastique zone survente                |
|`ADX_MIN`        |20    |ADX minimum pour valider une tendance           |
|`PER_MAX`        |30    |PER maximum pour score value                    |
|`PBV_MAX`        |8     |P/Book maximum pour score value                 |
|`ROE_MIN`        |8     |ROE minimum (%) pour score croissance           |

Les poids peuvent aussi être ajustés en temps réel dans l’interface via **⚙️ Critères**.

## 📊 Indicateurs calculés

### Valorisation (Score Value)

|Indicateur|Description                                 |
|----------|--------------------------------------------|
|PER       |Price-to-Earnings Ratio                     |
|P/Book    |Price-to-Book Value                         |
|EV/EBITDA |Enterprise Value / EBITDA                   |
|P/S       |Price-to-Sales                              |
|P/FCF     |Price-to-Free Cash Flow                     |
|Graham %  |Écart cours vs valeur Graham (√22.5×BPA×VNC)|

### Croissance & Qualité (Score Growth)

|Indicateur   |Description                        |
|-------------|-----------------------------------|
|ROE          |Return on Equity                   |
|BPA %        |Croissance des bénéfices par action|
|PEG          |PER / Croissance bénéfices         |
|Current Ratio|Liquidité court terme              |
|Debt/Equity  |Levier financier                   |
|FCF Yield    |Rendement Free Cash Flow           |

### Dividende (Score Dividende)

|Indicateur  |Description                         |
|------------|------------------------------------|
|Rendement % |Dividende annuel / Cours            |
|Payout Ratio|Part du bénéfice versée en dividende|

### Technique (Score Technique)

|Indicateur         |Description                                    |
|-------------------|-----------------------------------------------|
|RSI                |Relative Strength Index (14j)                  |
|MACD               |Moving Average Convergence Divergence (12/26/9)|
|CCI                |Commodity Channel Index (20j)                  |
|Stochastique       |%K sur 14 périodes                             |
|ADX                |Average Directional Index — force de tendance  |
|Bollinger          |Position dans les bandes (20j, ±2σ)            |
|MM20 / MM50 / MM200|Écart cours / moyennes mobiles                 |
|Williams %R        |Oscillateur survente/surachat (14j)            |
|Rate of Change     |Momentum sur 12 périodes                       |

## 📋 Ajouter des actions (tickers.csv)

Format du fichier `tickers.csv` :

```csv
ticker,groupe,pea
MC.PA,CAC40,1
AIR.PA,CAC40,1
AAPL,US,0
```

- `ticker` : symbole Yahoo Finance (ex: `MC.PA`, `AAPL`)
- `groupe` : catégorie libre (non utilisé dans le scoring)
- `pea` : `1` = éligible PEA, `0` = non éligible

## ⚠️ Avertissement

Outil d’aide à la décision basé sur des données publiques Yahoo Finance.  
Ne constitue pas un conseil en investissement. Les données peuvent être incomplètes ou retardées.