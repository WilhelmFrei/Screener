# 📡 Stock Screener — PEA & Compte-Titres

Screener automatique — données Yahoo Finance — scan quotidien via GitHub Actions.

## 🔧 Installation (5 minutes)

### 1. Définir votre mot de passe

Dans `generate.py`, ligne ~280, cherchez :
```python
const PWD_HASH = btoa('VOTRE_MOT_DE_PASSE');
```
Remplacez `VOTRE_MOT_DE_PASSE` par le code de votre choix. Par exemple :
```python
const PWD_HASH = btoa('moncode2025');
```

### 2. Activer GitHub Pages

- Aller dans **Settings → Pages**
- Source : **Deploy from a branch**
- Branch : **main** / dossier : **docs**
- Sauvegarder

Votre URL sera : `https://VOTRE-USERNAME.github.io/screener`

### 3. Premier scan

- Aller dans **Actions → 📡 Stock Screener → Run workflow**
- Attendre ~5 minutes
- Ouvrir votre URL

## 🔄 Scan automatique

Le scan se lance automatiquement :
- **Lundi-Vendredi à 7h** (heure de Paris)
- **Samedi à 9h** (données J-1)

Ou manuellement : **Actions → Run workflow**

## ⚙️ Configuration

Dans `generate.py` :

| Variable | Défaut | Description |
|---|---|---|
| `SCORE_MIN` | 35 | Score minimum pour apparaître |
| `TOP_N` | 150 | Nombre d'actions dans les résultats |
| `POIDS_VALUE` | 30 | Poids valorisation |
| `POIDS_GROWTH` | 25 | Poids croissance |
| `POIDS_DIVIDENDE` | 15 | Poids dividende |
| `POIDS_TECHNIQUE` | 30 | Poids technique |
| `RSI_SURVENTE` | 45 | Seuil RSI survendu |

## ⚠️ Avertissement

Outil d'aide à la décision. Ne constitue pas un conseil en investissement.
