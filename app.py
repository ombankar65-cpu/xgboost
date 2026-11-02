"""
Balancewise - credit card balance predictor
===========================================
A single-file Flask app that serves a polished UI and a JSON API on top of
the XGBoost regression model stored in `xgboost.pkl`.

Run:
    pip install -r requirements.txt
    python app.py          ->  http://127.0.0.1:5000

Environment variables (all optional):
    MODEL_PATH   path to the pickle file   (default: ./xgboost.pkl next to app.py)
    HOST / PORT  where to listen           (default: 127.0.0.1 / 5000)
    FLASK_DEBUG  set to 1 for debug mode
"""

import json
import logging
import math
import os
import pickle
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "xgboost.pkl"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("balancewise")


def load_model(path: Path):
    """Load the pickled model (joblib handles plain pickles too)."""
    try:
        return joblib.load(path)
    except Exception:  # fall back to the standard library
        with open(path, "rb") as fh:
            return pickle.load(fh)


model = load_model(MODEL_PATH)
FEATURES = [str(c) for c in model.feature_names_in_]  # column order the model was trained with
log.info("Loaded %s from %s", type(model).__name__, MODEL_PATH)
log.info("Model features: %s", FEATURES)

# --------------------------------------------------------------------------- #
# Feature configuration
# --------------------------------------------------------------------------- #
# Categorical columns were label-encoded (alphabetical order) when the model
# was trained. If your encoding differs, change these three maps.
GENDER = {"Male": 0, "Female": 1}
YES_NO = {"No": 0, "Yes": 1}
ETHNICITY = {"African American": 0, "Asian": 1, "Caucasian": 2}

# `min`/`max` are hard limits enforced by the server.
# `slider` is the typical range the model saw in training; values outside it
# are accepted but come back with a caution note.
CONFIG = {
    "groups": [
        {
            "title": "Credit details",
            "fields": [
                {"type": "number", "name": "Limit", "label": "Credit limit", "prefix": "$",
                 "hint": "Maximum the issuer allows", "min": 100, "max": 50000,
                 "slider": [800, 14000], "step": 100, "default": 5000, "dtype": "int"},
                {"type": "number", "name": "Rating", "label": "Credit rating score",
                 "hint": "Issuer's internal score", "min": 50, "max": 2000,
                 "slider": [90, 990], "step": 5, "default": 400, "dtype": "int"},
                {"type": "number", "name": "Cards", "label": "Number of cards",
                 "hint": "Cards held in total", "min": 1, "max": 15,
                 "slider": [1, 9], "step": 1, "default": 3, "dtype": "int"},
            ],
        },
        {
            "title": "Personal details",
            "fields": [
                {"type": "number", "name": "Income", "label": "Annual income", "prefix": "$",
                 "suffix": "k", "hint": "In thousands of dollars", "min": 0, "max": 500,
                 "slider": [10, 190], "step": 0.5, "default": 45, "dtype": "float"},
                {"type": "number", "name": "Age", "label": "Age",
                 "hint": "In years", "min": 18, "max": 100,
                 "slider": [18, 100], "step": 1, "default": 45, "dtype": "int"},
                {"type": "number", "name": "Education", "label": "Years of education",
                 "hint": "Total years in school", "min": 0, "max": 25,
                 "slider": [5, 20], "step": 1, "default": 14, "dtype": "int"},
            ],
        },
        {
            "title": "Background",
            "fields": [
                {"type": "choice", "name": "Gender", "label": "Gender",
                 "options": list(GENDER), "default": "Female"},
                {"type": "choice", "name": "Married", "label": "Married",
                 "options": list(YES_NO), "default": "Yes"},
                {"type": "choice", "name": "Student", "label": "Student",
                 "options": list(YES_NO), "default": "No"},
                {"type": "choice", "name": "Ethnicity", "label": "Ethnicity",
                 "options": list(ETHNICITY), "default": "Caucasian"},
            ],
        },
    ]
}

NUMERIC = {f["name"]: f for g in CONFIG["groups"] for f in g["fields"] if f["type"] == "number"}
CHOICES = {
    "Gender": GENDER,
    "Married": YES_NO,
    "Student": YES_NO,
    "Ethnicity": ETHNICITY,
}

# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
class ValidationError(ValueError):
    pass


def parse_payload(data: dict):
    """Validate user input. Returns (feature_row, warnings)."""
    row, warnings = {}, []

    for name, spec in NUMERIC.items():
        raw = data.get(name)
        try:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
        except (TypeError, ValueError):
            raise ValidationError(f"{spec['label']} must be a number.")

        if not spec["min"] <= value <= spec["max"]:
            raise ValidationError(
                f"{spec['label']} must be between {spec['min']:g} and {spec['max']:g}."
            )

        lo, hi = spec["slider"]
        if not lo <= value <= hi:
            warnings.append(
                f"{spec['label']} is outside the usual range of {lo:g} to {hi:g}, "
                "so treat this estimate with extra caution."
            )
        row[name] = float(value) if spec["dtype"] == "float" else int(round(value))

    for name, mapping in CHOICES.items():
        choice = data.get(name)
        if choice not in mapping:
            raise ValidationError(f"Choose a valid option for {name.lower()}.")
        row[name] = mapping[choice]

    return row, warnings


def utilization_band(pct: float) -> str:
    if pct < 30:
        return "healthy"
    if pct < 60:
        return "elevated"
    return "high"


def run_prediction(row: dict) -> float:
    frame = pd.DataFrame([row])[FEATURES]
    frame = frame.astype({c: ("float64" if NUMERIC.get(c, {}).get("dtype") == "float" else "int64")
                          for c in FEATURES})
    return float(model.predict(frame)[0])


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)
app.json.sort_keys = False


@app.get("/")
def index():
    return Response(PAGE.replace("__CONFIG__", json.dumps(CONFIG)), mimetype="text/html")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    try:
        row, warnings = parse_payload(data)
        raw = run_prediction(row)
    except ValidationError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:  # pragma: no cover - unexpected model failure
        log.exception("Prediction failed")
        return jsonify(error="The model could not produce an estimate. Check the server log."), 500

    limit = row["Limit"]
    balance = min(max(raw, 0.0), float(limit))  # a balance can't be negative or exceed the limit
    utilization = balance / limit * 100
    if abs(balance - raw) > 0.5:
        warnings.append("The raw model output was adjusted to fit between $0 and the credit limit.")

    return jsonify(
        balance=round(balance, 2),
        limit=limit,
        available=round(limit - balance, 2),
        utilization=round(utilization, 1),
        band=utilization_band(utilization),
        warnings=warnings,
    )


@app.get("/health")
def health():
    return jsonify(status="ok", model=type(model).__name__, features=FEATURES)


# --------------------------------------------------------------------------- #
# Front end (served as one page; config is injected from CONFIG above)
# --------------------------------------------------------------------------- #
PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Balancewise | Credit card balance predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --ink:#16164A; --ink-2:#22226F; --lagoon:#19B8A6; --lagoon-d:#0B7C70;
  --gold:#F3B84B; --mist:#F4F6FB; --paper:#FFFFFF; --coral:#E0524F;
  --text:#1D1F3A; --muted:#5B6085; --line:#E1E5F1;
  --display:'Bricolage Grotesque','Segoe UI',system-ui,sans-serif;
  --body:'DM Sans','Segoe UI',system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font-family:var(--body);font-size:16px;line-height:1.55;color:var(--text);background:var(--mist);-webkit-font-smoothing:antialiased}
button,input{font:inherit}

.shell{display:grid;grid-template-columns:minmax(380px,5fr) 6fr;min-height:100vh}

/* ---------- left stage ---------- */
.stage{position:sticky;top:0;height:100vh;overflow:hidden;color:#fff;padding:44px 52px;display:flex;flex-direction:column;gap:28px;
  background:radial-gradient(900px 600px at 15% 0%,var(--ink-2) 0%,transparent 60%),var(--ink)}
.guilloche{position:absolute;right:-220px;bottom:-240px;width:760px;height:760px;opacity:.16;pointer-events:none;animation:spin 180s linear infinite}
.guilloche ellipse{fill:none;stroke:var(--lagoon);stroke-width:1}
@keyframes spin{to{transform:rotate(360deg)}}

.brand{display:flex;align-items:center;gap:12px;font-family:var(--display);font-weight:700;font-size:20px;letter-spacing:.2px;position:relative}
.brand svg{width:34px;height:34px}
.stage h1{position:relative;margin:0;font-family:var(--display);font-weight:700;font-size:clamp(30px,3.2vw,44px);line-height:1.08;letter-spacing:-.8px;max-width:15ch}
.stage .lede{position:relative;margin:0;color:#C5C9EE;max-width:36ch}

.tilt{position:relative;width:min(100%,410px);perspective:900px;margin-top:4px}
.card{position:relative;aspect-ratio:1.586;border-radius:22px;padding:24px 26px;color:#fff;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;
  background:linear-gradient(135deg,#19B8A6 0%,#1B8FB0 46%,#3A3FA8 100%);
  box-shadow:0 30px 60px -20px rgba(0,0,0,.55),0 0 0 1px rgba(255,255,255,.18) inset;
  transform:rotateX(var(--rx,0deg)) rotateY(var(--ry,0deg));transition:transform .18s ease-out;
  animation:deal .95s cubic-bezier(.2,.8,.2,1) both}
.card::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(260px circle at var(--mx,30%) var(--my,10%),rgba(255,255,255,.28),transparent 60%)}
@keyframes deal{from{opacity:0;transform:translateY(46px) rotate(-7deg) scale(.94)}to{opacity:1}}
.card-top{display:flex;justify-content:space-between;align-items:center}
.chip{width:46px;height:35px;border-radius:7px;background:linear-gradient(135deg,#FFD982,#E9A93A);position:relative;overflow:hidden;box-shadow:0 0 0 1px rgba(0,0,0,.15) inset}
.chip::before,.chip::after{content:"";position:absolute;background:rgba(120,70,0,.35)}
.chip::before{left:0;right:0;top:50%;height:1px}
.chip::after{top:0;bottom:0;left:50%;width:1px}
.nfc{width:28px;height:28px;opacity:.9}
.card-num{font-family:var(--display);font-size:17px;letter-spacing:3px;opacity:.85}
.card-bottom{display:flex;justify-content:space-between;align-items:flex-end;gap:12px}
.card-bottom small{display:block;font-size:12.5px;opacity:.85}
.amount{font-family:var(--display);font-weight:700;font-size:clamp(34px,3.6vw,46px);line-height:1;letter-spacing:-1px;font-variant-numeric:tabular-nums}
.amount.idle{opacity:.55}
.card-limit{text-align:right}
.card-limit b{font-weight:600;font-variant-numeric:tabular-nums}

.readout{position:relative;display:grid;grid-template-columns:1fr 1fr;gap:18px 28px;max-width:410px}
.meter{grid-column:1/-1}
.meter-head{display:flex;justify-content:space-between;font-size:14px;color:#C5C9EE;margin-bottom:8px}
.meter-head b{color:#fff;font-weight:600}
.track{height:8px;border-radius:99px;background:rgba(255,255,255,.14);overflow:hidden}
.fill{height:100%;width:0;border-radius:99px;background:var(--lagoon);transition:width 1s cubic-bezier(.2,.8,.2,1),background .4s}
[data-band=elevated] .fill{background:var(--gold)}
[data-band=high] .fill{background:var(--coral)}
.stat span{display:block;font-size:13.5px;color:#C5C9EE}
.stat b{font-family:var(--display);font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.badge{display:inline-block;padding:2px 12px;border-radius:99px;font-family:var(--body);font-size:14px;font-weight:600;background:rgba(255,255,255,.14)}
[data-band=healthy] .badge{background:rgba(25,184,166,.28)}
[data-band=elevated] .badge{background:rgba(243,184,75,.28)}
[data-band=high] .badge{background:rgba(224,82,79,.34)}
.notes{position:relative;margin:0;padding:0;list-style:none;font-size:13.5px;color:#FFE2A8;max-width:410px}
.notes li{padding-left:18px;position:relative;margin-top:4px}
.notes li::before{content:"!";position:absolute;left:0;top:0;font-weight:700}

/* ---------- right form ---------- */
.panel{padding:56px clamp(24px,5vw,72px) 64px;max-width:760px}
.panel h2{margin:0 0 6px;font-family:var(--display);font-size:30px;line-height:1.15;letter-spacing:-.5px}
.panel .sub{margin:0 0 30px;color:var(--muted)}
fieldset.group{border:0;margin:0 0 22px;padding:22px 24px 8px;background:var(--paper);border-radius:18px;box-shadow:0 1px 0 var(--line),0 10px 30px -18px rgba(22,22,74,.25)}
fieldset.group legend{float:left;width:100%;padding:0;margin:0 0 14px;font-family:var(--display);font-weight:700;font-size:18px;color:var(--ink)}
fieldset.group legend + *{clear:both}
.field{padding:12px 0 16px;border-top:1px solid var(--line)}
.field:first-of-type{border-top:0}
.field-head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:8px}
.field-head label,.field-head .lbl{font-weight:600;font-size:15px}
.hint{font-size:13px;color:var(--muted)}
.ctl{display:grid;grid-template-columns:150px 1fr;gap:18px;align-items:center}
.num{display:flex;align-items:center;gap:4px;height:44px;padding:0 12px;border:1.5px solid var(--line);border-radius:11px;background:#fff;transition:border-color .15s,box-shadow .15s}
.num:focus-within{border-color:var(--lagoon);box-shadow:0 0 0 4px rgba(25,184,166,.18)}
.num span{color:var(--muted);font-weight:500}
.num input{flex:1;min-width:0;border:0;outline:0;background:transparent;color:var(--text);font-weight:600;font-variant-numeric:tabular-nums}
.num input::-webkit-inner-spin-button{opacity:.4}

.rng{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:99px;outline:none;cursor:pointer;
  background:linear-gradient(to right,var(--lagoon) var(--p,50%),var(--line) var(--p,50%))}
.rng::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:#fff;border:4px solid var(--ink);box-shadow:0 2px 8px rgba(22,22,74,.3);transition:transform .15s}
.rng::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#fff;border:4px solid var(--ink);box-shadow:0 2px 8px rgba(22,22,74,.3)}
.rng:active::-webkit-slider-thumb{transform:scale(1.18)}
.rng:focus-visible::-webkit-slider-thumb{outline:3px solid var(--gold);outline-offset:2px}
.rng:focus-visible::-moz-range-thumb{outline:3px solid var(--gold);outline-offset:2px}

.seg{display:flex;flex-wrap:wrap;gap:8px}
.seg label{position:relative;cursor:pointer}
.seg input{position:absolute;opacity:0;inset:0;margin:0;cursor:pointer}
.seg span{display:block;padding:9px 18px;border:1.5px solid var(--line);border-radius:99px;font-weight:500;font-size:15px;background:#fff;transition:background .18s,color .18s,border-color .18s}
.seg label:hover span{border-color:var(--lagoon)}
.seg input:checked + span{background:var(--ink);border-color:var(--ink);color:#fff}
.seg input:focus-visible + span{outline:3px solid var(--gold);outline-offset:2px}

.cta{width:100%;height:56px;border:0;border-radius:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:10px;
  background:var(--ink);color:#fff;font-family:var(--display);font-weight:700;font-size:18px;letter-spacing:.2px;
  box-shadow:0 14px 28px -14px rgba(22,22,74,.7);transition:background .2s,transform .15s}
.cta:hover{background:var(--lagoon-d)}
.cta:active{transform:translateY(1px)}
.cta:focus-visible{outline:3px solid var(--gold);outline-offset:3px}
.cta[disabled]{opacity:.8;cursor:progress}
.spin{width:18px;height:18px;border:3px solid rgba(255,255,255,.35);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}
.cta[disabled] .spin{display:block}
.alert{margin:14px 0 0;padding:12px 16px;border-radius:12px;background:#FDECEC;color:#8E2321;font-weight:500;display:none}
.alert.show{display:block}
.fine{margin:22px 0 0;font-size:13px;color:var(--muted);max-width:60ch}

/* ---------- responsive ---------- */
@media (max-width:980px){
  .shell{grid-template-columns:1fr}
  .stage{position:relative;height:auto;padding:32px 24px 40px}
  .panel{max-width:none;margin:0 auto;width:100%}
}
@media (max-width:520px){
  .ctl{grid-template-columns:1fr;gap:12px}
  fieldset.group{padding:18px 16px 4px}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important;transition:none!important}
}
</style>
</head>
<body>
<div class="shell">

  <!-- ===== Stage ===== -->
  <aside class="stage" id="stage" data-band="idle">
    <svg class="guilloche" id="guilloche" viewBox="0 0 600 600" aria-hidden="true"></svg>

    <div class="brand">
      <svg viewBox="0 0 40 40" aria-hidden="true">
        <circle cx="20" cy="20" r="18" fill="none" stroke="#19B8A6" stroke-width="3"/>
        <path d="M11 24c4-9 8-9 10-4s5 4 8-4" fill="none" stroke="#F3B84B" stroke-width="3" stroke-linecap="round"/>
      </svg>
      Balancewise
    </div>

    <h1>Estimate a cardholder's monthly balance</h1>
    <p class="lede">Enter the cardholder's details and the model predicts the average balance they are likely to carry.</p>

    <div class="tilt" id="tilt">
      <div class="card" id="card">
        <div class="card-top">
          <div class="chip"></div>
          <svg class="nfc" viewBox="0 0 32 32" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" aria-hidden="true">
            <path d="M9 10a9 9 0 0 1 0 12"/><path d="M14 7a14 14 0 0 1 0 18"/><path d="M19 4a19 19 0 0 1 0 24"/>
          </svg>
        </div>
        <div class="card-num">4821 &nbsp;0937 &nbsp;5502 &nbsp;1160</div>
        <div class="card-bottom">
          <div>
            <small>Predicted balance</small>
            <div class="amount idle" id="amount">$0</div>
          </div>
          <div class="card-limit">
            <small>Credit limit</small>
            <b id="cardLimit">$5,000</b>
          </div>
        </div>
      </div>
    </div>

    <div class="readout" id="readout" aria-live="polite">
      <div class="meter">
        <div class="meter-head"><span>Limit used</span><b id="utilText">Waiting for a prediction</b></div>
        <div class="track"><div class="fill" id="fill"></div></div>
      </div>
      <div class="stat"><span>Credit still available</span><b id="avail">-</b></div>
      <div class="stat"><span>Usage level</span><b><span class="badge" id="badge">-</span></b></div>
    </div>
    <ul class="notes" id="notes"></ul>
  </aside>

  <!-- ===== Form ===== -->
  <main class="panel">
    <h2>Cardholder profile</h2>
    <p class="sub">Adjust the sliders or type exact values, then select Predict balance.</p>

    <form id="form" novalidate>
      <div id="groups"></div>
      <button class="cta" id="submit" type="submit"><span class="spin" aria-hidden="true"></span><span id="ctaText">Predict balance</span></button>
      <div class="alert" id="alert" role="alert"></div>
    </form>

    <p class="fine">Estimates come from an XGBoost regression model trained on historical credit card data. They are statistical guidance, not a credit decision or financial advice.</p>
  </main>
</div>

<script>
const CFG = __CONFIG__;
const $ = (s, r = document) => r.querySelector(s);
const money = n => new Intl.NumberFormat('en-US', {style:'currency', currency:'USD', maximumFractionDigits:0}).format(n);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---- decorative guilloche (the security-print pattern found on banknotes) ---- */
(() => {
  const svg = $('#guilloche'), ns = 'http://www.w3.org/2000/svg';
  for (let i = 0; i < 36; i++) {
    const e = document.createElementNS(ns, 'ellipse');
    e.setAttribute('cx', 300); e.setAttribute('cy', 300);
    e.setAttribute('rx', 285); e.setAttribute('ry', 95);
    e.setAttribute('transform', `rotate(${i * 5} 300 300)`);
    svg.appendChild(e);
  }
})();

/* ---- build the form from the server-side config ---- */
const groupsEl = $('#groups');
CFG.groups.forEach(g => {
  const fs = document.createElement('fieldset');
  fs.className = 'group';
  fs.innerHTML = `<legend>${g.title}</legend>`;
  g.fields.forEach(f => fs.appendChild(f.type === 'number' ? numberField(f) : choiceField(f)));
  groupsEl.appendChild(fs);
});

function numberField(f) {
  const el = document.createElement('div');
  el.className = 'field';
  el.innerHTML = `
    <div class="field-head"><label for="${f.name}">${f.label}</label><span class="hint">${f.hint}</span></div>
    <div class="ctl">
      <div class="num">${f.prefix ? `<span>${f.prefix}</span>` : ''}
        <input id="${f.name}" name="${f.name}" type="number" inputmode="decimal"
               min="${f.min}" max="${f.max}" step="${f.step}" value="${f.default}" required>
        ${f.suffix ? `<span>${f.suffix}</span>` : ''}
      </div>
      <input class="rng" type="range" aria-label="${f.label} slider"
             min="${f.slider[0]}" max="${f.slider[1]}" step="${f.step}" value="${f.default}">
    </div>`;
  const num = $('input[type=number]', el), rng = $('.rng', el);
  const paint = () => {
    const p = (rng.value - rng.min) / (rng.max - rng.min) * 100;
    rng.style.setProperty('--p', Math.min(100, Math.max(0, p)) + '%');
  };
  rng.addEventListener('input', () => { num.value = rng.value; paint(); syncLimit(); });
  num.addEventListener('input', () => { if (num.value !== '') { rng.value = num.value; paint(); syncLimit(); } });
  paint();
  return el;
}

function choiceField(f) {
  const el = document.createElement('div');
  el.className = 'field';
  el.innerHTML = `
    <div class="field-head"><span class="lbl" id="lbl-${f.name}">${f.label}</span></div>
    <div class="seg" role="radiogroup" aria-labelledby="lbl-${f.name}">
      ${f.options.map(o => `<label><input type="radio" name="${f.name}" value="${o}" ${o === f.default ? 'checked' : ''}><span>${o}</span></label>`).join('')}
    </div>`;
  return el;
}

/* keep the limit printed on the card in step with the form */
function syncLimit() {
  const v = parseFloat($('#Limit').value);
  if (isFinite(v)) $('#cardLimit').textContent = money(v);
}
syncLimit();

/* ---- card tilt that follows the pointer ---- */
const tilt = $('#tilt'), card = $('#card');
if (!reduceMotion) {
  tilt.addEventListener('pointermove', e => {
    const r = tilt.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    card.style.setProperty('--ry', ((x - .5) * 14).toFixed(2) + 'deg');
    card.style.setProperty('--rx', ((.5 - y) * 14).toFixed(2) + 'deg');
    card.style.setProperty('--mx', (x * 100) + '%');
    card.style.setProperty('--my', (y * 100) + '%');
  });
  tilt.addEventListener('pointerleave', () => {
    card.style.setProperty('--rx', '0deg'); card.style.setProperty('--ry', '0deg');
  });
}

/* ---- count-up for the predicted balance ---- */
let shown = 0;
function countTo(target) {
  const el = $('#amount');
  el.classList.remove('idle');
  if (reduceMotion) { el.textContent = money(target); shown = target; return; }
  const from = shown, t0 = performance.now(), dur = 1100;
  const tick = now => {
    const k = Math.min(1, (now - t0) / dur), ease = 1 - Math.pow(1 - k, 3);
    el.textContent = money(from + (target - from) * ease);
    if (k < 1) requestAnimationFrame(tick); else shown = target;
  };
  requestAnimationFrame(tick);
}

/* ---- submit ---- */
const form = $('#form'), btn = $('#submit'), alertEl = $('#alert');
const bandLabel = {healthy:'Healthy', elevated:'Elevated', high:'High'};

form.addEventListener('submit', async e => {
  e.preventDefault();
  alertEl.classList.remove('show');
  btn.disabled = true; $('#ctaText').textContent = 'Predicting';
  const payload = Object.fromEntries(new FormData(form).entries());
  try {
    const res = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Something went wrong. Try again.');
    show(data);
  } catch (err) {
    alertEl.textContent = err.message;
    alertEl.classList.add('show');
  } finally {
    btn.disabled = false; $('#ctaText').textContent = 'Predict balance';
  }
});

function show(d) {
  countTo(d.balance);
  $('#cardLimit').textContent = money(d.limit);
  $('#stage').dataset.band = d.band;
  $('#fill').style.width = Math.min(100, d.utilization) + '%';
  $('#utilText').textContent = d.utilization.toFixed(1) + '% of limit';
  $('#avail').textContent = money(d.available);
  $('#badge').textContent = bandLabel[d.band];
  $('#notes').innerHTML = d.warnings.map(w => `<li>${w}</li>`).join('');
  if (matchMedia('(max-width:980px)').matches) $('#stage').scrollIntoView({behavior: reduceMotion ? 'auto' : 'smooth'});
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )