"""
utils/visuals.py
Premium SVG illustrations and HTML components for BreastScan AI.
"""
import base64
import math


def _svg_img(svg: str, h: str = "auto") -> str:
    b64 = base64.b64encode(svg.encode()).decode()
    return (f'<img src="data:image/svg+xml;base64,{b64}" '
            f'style="width:100%;height:{h};display:block;border-radius:12px;" />')


# ─── Hero Visual ─────────────────────────────────────────────────────────────

def get_hero_svg() -> str:
    svg = """<svg width="560" height="320" viewBox="0 0 560 320" xmlns="http://www.w3.org/2000/svg">
<defs>
  <radialGradient id="hglow" cx="50%" cy="50%" r="50%">
    <stop offset="0%" stop-color="#7b6cff" stop-opacity="0.18"/>
    <stop offset="100%" stop-color="#7b6cff" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="hglow2" cx="50%" cy="50%" r="50%">
    <stop offset="0%" stop-color="#22d3a0" stop-opacity="0.14"/>
    <stop offset="100%" stop-color="#22d3a0" stop-opacity="0"/>
  </radialGradient>
  <filter id="blur4"><feGaussianBlur stdDeviation="4"/></filter>
  <filter id="blur8"><feGaussianBlur stdDeviation="8"/></filter>
</defs>

<!-- Background -->
<rect width="560" height="320" fill="#0d0d1a" rx="16"/>

<!-- Glow blobs -->
<circle cx="180" cy="160" r="120" fill="url(#hglow)" filter="url(#blur8)"/>
<circle cx="390" cy="160" r="100" fill="url(#hglow2)" filter="url(#blur8)"/>

<!-- Grid lines -->
<g stroke="rgba(255,255,255,0.04)" stroke-width="1">
  <line x1="0" y1="80"  x2="560" y2="80"/>
  <line x1="0" y1="160" x2="560" y2="160"/>
  <line x1="0" y1="240" x2="560" y2="240"/>
  <line x1="112" y1="0" x2="112" y2="320"/>
  <line x1="224" y1="0" x2="224" y2="320"/>
  <line x1="336" y1="0" x2="336" y2="320"/>
  <line x1="448" y1="0" x2="448" y2="320"/>
</g>

<!-- Central circle — scan ring -->
<circle cx="200" cy="160" r="90" fill="none" stroke="rgba(123,108,255,0.15)" stroke-width="1"/>
<circle cx="200" cy="160" r="70" fill="none" stroke="rgba(123,108,255,0.2)" stroke-width="1"/>
<circle cx="200" cy="160" r="50" fill="none" stroke="rgba(123,108,255,0.3)" stroke-width="1"/>

<!-- Spinning arc -->
<circle cx="200" cy="160" r="90" fill="none"
  stroke="url(#hglow)" stroke-width="2.5"
  stroke-dasharray="80 480" stroke-linecap="round"/>

<!-- Cell nucleus -->
<circle cx="200" cy="160" r="28" fill="rgba(123,108,255,0.25)" stroke="#7b6cff" stroke-width="1.5"/>
<circle cx="200" cy="160" r="14" fill="rgba(123,108,255,0.5)"/>
<circle cx="193" cy="153" r="5" fill="rgba(255,255,255,0.7)"/>

<!-- Scan line -->
<line x1="110" y1="160" x2="290" y2="160" stroke="rgba(123,108,255,0.4)" stroke-width="1" stroke-dasharray="4 4"/>
<line x1="200" y1="70" x2="200" y2="250" stroke="rgba(123,108,255,0.4)" stroke-width="1" stroke-dasharray="4 4"/>

<!-- Label dots on ring -->
<circle cx="200" cy="70" r="4" fill="#7b6cff"/>
<circle cx="290" cy="160" r="4" fill="#7b6cff"/>
<circle cx="200" cy="250" r="4" fill="#7b6cff"/>
<circle cx="110" cy="160" r="4" fill="#7b6cff"/>

<!-- Right panel — stats -->
<rect x="330" y="40" width="200" height="240" rx="12" fill="rgba(30,30,58,0.8)" stroke="rgba(255,255,255,0.07)" stroke-width="1"/>

<!-- Stat rows -->
<text x="355" y="75" font-family="sans-serif" font-size="10" fill="#7c7c9e" letter-spacing="1">ACCURACY</text>
<text x="355" y="95" font-family="sans-serif" font-size="22" font-weight="700" fill="#e8e8f4">97.4%</text>
<rect x="355" y="102" width="150" height="3" rx="2" fill="rgba(255,255,255,0.06)"/>
<rect x="355" y="102" width="146" height="3" rx="2" fill="#7b6cff"/>

<text x="355" y="135" font-family="sans-serif" font-size="10" fill="#7c7c9e" letter-spacing="1">ROC-AUC</text>
<text x="355" y="155" font-family="sans-serif" font-size="22" font-weight="700" fill="#e8e8f4">0.993</text>
<rect x="355" y="162" width="150" height="3" rx="2" fill="rgba(255,255,255,0.06)"/>
<rect x="355" y="162" width="149" height="3" rx="2" fill="#22d3a0"/>

<text x="355" y="195" font-family="sans-serif" font-size="10" fill="#7c7c9e" letter-spacing="1">FEATURES</text>
<text x="355" y="215" font-family="sans-serif" font-size="22" font-weight="700" fill="#e8e8f4">30</text>
<rect x="355" y="222" width="150" height="3" rx="2" fill="rgba(255,255,255,0.06)"/>
<rect x="355" y="222" width="60" height="3" rx="2" fill="#f59e0b"/>

<text x="355" y="255" font-family="sans-serif" font-size="10" fill="#7c7c9e" letter-spacing="1">SAMPLES</text>
<text x="355" y="275" font-family="sans-serif" font-size="22" font-weight="700" fill="#e8e8f4">569</text>

<!-- Title text -->
<text x="200" y="298" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#7c7c9e" letter-spacing="2">BREAST CANCER AI DIAGNOSTICS</text>
</svg>"""
    return f'<div style="border-radius:16px;overflow:hidden;margin-bottom:16px;">{_svg_img(svg, "280px")}</div>'


# ─── Anatomy ─────────────────────────────────────────────────────────────────

def get_anatomy_svg() -> str:
    svg = """<svg width="680" height="480" viewBox="0 0 680 480" xmlns="http://www.w3.org/2000/svg">
<defs>
  <radialGradient id="sk" cx="50%" cy="40%" r="55%"><stop offset="0%" stop-color="#F5C5A3"/><stop offset="100%" stop-color="#D4956A"/></radialGradient>
  <radialGradient id="ft" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#FAE8C8" stop-opacity="0.9"/><stop offset="100%" stop-color="#E8C87A" stop-opacity="0.6"/></radialGradient>
  <radialGradient id="bg" cx="40%" cy="40%" r="55%"><stop offset="0%" stop-color="#7ED9A0"/><stop offset="100%" stop-color="#2E8B57"/></radialGradient>
  <radialGradient id="mg" cx="40%" cy="40%" r="55%"><stop offset="0%" stop-color="#F97070"/><stop offset="100%" stop-color="#C0202A"/></radialGradient>
  <clipPath id="cl1"><ellipse cx="180" cy="240" rx="138" ry="145"/></clipPath>
  <clipPath id="cl2"><ellipse cx="500" cy="240" rx="138" ry="145"/></clipPath>
</defs>
<rect width="680" height="480" fill="#0d0d1a"/>
<text x="340" y="30" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#e8e8f4" letter-spacing="1">BREAST TISSUE — BENIGN vs MALIGNANT</text>

<!-- LEFT: Benign -->
<ellipse cx="180" cy="240" rx="138" ry="145" fill="url(#sk)" opacity="0.9"/>
<g clip-path="url(#cl1)">
  <ellipse cx="148" cy="200" rx="35" ry="25" fill="url(#ft)" opacity="0.75"/>
  <ellipse cx="205" cy="175" rx="28" ry="20" fill="url(#ft)" opacity="0.7"/>
  <ellipse cx="228" cy="222" rx="30" ry="22" fill="url(#ft)" opacity="0.68"/>
  <ellipse cx="155" cy="262" rx="33" ry="21" fill="url(#ft)" opacity="0.75"/>
  <path d="M180 335 Q180 290 175 252 Q170 224 165 196 Q162 182 170 172" stroke="#E8A0B8" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.8"/>
  <path d="M180 335 Q186 284 192 250 Q202 220 207 190" stroke="#E8A0B8" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.7"/>
  <!-- Benign lump -->
  <ellipse cx="163" cy="207" rx="22" ry="17" fill="url(#bg)" opacity="0.95"/>
  <ellipse cx="163" cy="207" rx="22" ry="17" fill="none" stroke="#22d3a0" stroke-width="2"/>
  <text x="163" y="204" text-anchor="middle" font-family="sans-serif" font-size="7" font-weight="700" fill="#052e16">BENIGN</text>
  <text x="163" y="214" text-anchor="middle" font-family="sans-serif" font-size="6" fill="#052e16">smooth</text>
</g>
<ellipse cx="180" cy="240" rx="138" ry="145" fill="none" stroke="#C47040" stroke-width="2.5"/>
<ellipse cx="180" cy="368" rx="10" ry="7" fill="#C47040" opacity="0.8"/>

<!-- Benign labels -->
<line x1="142" y1="207" x2="28" y2="176" stroke="#334155" stroke-width="0.7" stroke-dasharray="4 3"/>
<text x="24" y="172" text-anchor="end" font-family="sans-serif" font-size="10" fill="#94a3b8">Benign lump</text>
<text x="24" y="184" text-anchor="end" font-family="sans-serif" font-size="9" fill="#22d3a0">(smooth, movable)</text>
<line x1="148" y1="252" x2="28" y2="252" stroke="#334155" stroke-width="0.7" stroke-dasharray="4 3"/>
<text x="24" y="248" text-anchor="end" font-family="sans-serif" font-size="10" fill="#94a3b8">Fat lobule</text>
<line x1="174" y1="285" x2="28" y2="315" stroke="#334155" stroke-width="0.7" stroke-dasharray="4 3"/>
<text x="24" y="312" text-anchor="end" font-family="sans-serif" font-size="10" fill="#94a3b8">Milk duct</text>

<text x="180" y="415" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="800" fill="#22d3a0">BENIGN</text>
<text x="180" y="432" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#7c7c9e">Clear borders · Soft · Mobile</text>

<!-- RIGHT: Malignant -->
<ellipse cx="500" cy="240" rx="138" ry="145" fill="url(#sk)" opacity="0.9"/>
<g clip-path="url(#cl2)">
  <ellipse cx="468" cy="200" rx="35" ry="25" fill="url(#ft)" opacity="0.75"/>
  <ellipse cx="525" cy="175" rx="28" ry="20" fill="url(#ft)" opacity="0.7"/>
  <ellipse cx="548" cy="222" rx="30" ry="22" fill="url(#ft)" opacity="0.68"/>
  <ellipse cx="475" cy="262" rx="33" ry="21" fill="url(#ft)" opacity="0.75"/>
  <path d="M500 335 Q500 290 495 252 Q490 224 485 196" stroke="#E8A0B8" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.8"/>
  <!-- Malignant tumor — irregular shape -->
  <path d="M476 205 Q472 192 483 186 Q489 175 497 183 Q504 171 511 181 Q518 172 524 184 Q533 180 531 194 Q542 198 538 212 Q548 220 539 227 Q543 238 531 238 Q528 249 518 246 Q511 257 503 248 Q494 258 487 248 Q476 244 477 232 Q465 226 470 214 Q463 207 476 205Z" fill="url(#mg)" opacity="0.92"/>
  <path d="M476 205 Q472 192 483 186 Q489 175 497 183 Q504 171 511 181 Q518 172 524 184 Q533 180 531 194 Q542 198 538 212 Q548 220 539 227 Q543 238 531 238 Q528 249 518 246 Q511 257 503 248 Q494 258 487 248 Q476 244 477 232 Q465 226 470 214 Q463 207 476 205Z" fill="none" stroke="#f25c5c" stroke-width="1.5"/>
  <text x="503" y="210" text-anchor="middle" font-family="sans-serif" font-size="6.5" font-weight="700" fill="#7f1d1d">MALIGNANT</text>
  <text x="503" y="220" text-anchor="middle" font-family="sans-serif" font-size="6" fill="#7f1d1d">irregular</text>
  <!-- Invasion spikes -->
  <line x1="538" y1="218" x2="556" y2="214" stroke="#C02020" stroke-width="1" stroke-dasharray="3 2" opacity="0.7"/>
  <line x1="535" y1="230" x2="552" y2="238" stroke="#C02020" stroke-width="1" stroke-dasharray="3 2" opacity="0.6"/>
  <line x1="476" y1="232" x2="458" y2="240" stroke="#C02020" stroke-width="1" stroke-dasharray="3 2" opacity="0.6"/>
</g>
<ellipse cx="500" cy="240" rx="138" ry="145" fill="none" stroke="#C47040" stroke-width="2.5"/>
<ellipse cx="500" cy="368" rx="10" ry="7" fill="#C47040" opacity="0.8"/>

<!-- Malignant labels -->
<line x1="532" y1="202" x2="652" y2="168" stroke="#334155" stroke-width="0.7" stroke-dasharray="4 3"/>
<text x="656" y="164" font-family="sans-serif" font-size="10" fill="#94a3b8">Malignant tumor</text>
<text x="656" y="176" font-family="sans-serif" font-size="9" fill="#f25c5c">(irregular, fixed)</text>
<line x1="550" y1="230" x2="652" y2="252" stroke="#334155" stroke-width="0.7" stroke-dasharray="4 3"/>
<text x="656" y="248" font-family="sans-serif" font-size="10" fill="#94a3b8">Invasion zone</text>
<line x1="475" y1="260" x2="652" y2="312" stroke="#334155" stroke-width="0.7" stroke-dasharray="4 3"/>
<text x="656" y="308" font-family="sans-serif" font-size="10" fill="#94a3b8">Distorted ducts</text>

<text x="500" y="415" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="800" fill="#f25c5c">MALIGNANT</text>
<text x="500" y="432" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#7c7c9e">Irregular · Hard · Fixed</text>

<!-- Divider -->
<line x1="340" y1="60" x2="340" y2="400" stroke="#1e1e3a" stroke-width="1.5" stroke-dasharray="6 4"/>
</svg>"""
    return f'<div style="border-radius:16px;overflow:hidden;margin:12px 0;">{_svg_img(svg, "400px")}</div>'


# ─── Cell Comparison ──────────────────────────────────────────────────────────

def get_cell_comparison_svg() -> str:
    svg = """<svg width="340" height="240" viewBox="0 0 340 240" xmlns="http://www.w3.org/2000/svg">
<defs>
  <radialGradient id="nc" cx="40%" cy="35%" r="60%"><stop offset="0%" stop-color="#a3e4c1"/><stop offset="100%" stop-color="#2E8B57" stop-opacity="0.6"/></radialGradient>
  <radialGradient id="cc" cx="40%" cy="35%" r="60%"><stop offset="0%" stop-color="#fca5a5"/><stop offset="100%" stop-color="#C0202A" stop-opacity="0.6"/></radialGradient>
  <radialGradient id="nn" cx="45%" cy="40%" r="50%"><stop offset="0%" stop-color="#86efac"/><stop offset="100%" stop-color="#166534"/></radialGradient>
  <radialGradient id="cn" cx="45%" cy="40%" r="50%"><stop offset="0%" stop-color="#fda4af"/><stop offset="100%" stop-color="#9f1239"/></radialGradient>
</defs>
<rect width="340" height="240" fill="#12122b" rx="12"/>
<text x="170" y="22" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#e8e8f4" letter-spacing="1">CELL COMPARISON</text>

<!-- Normal cells -->
<circle cx="68"  cy="110" r="38" fill="url(#nc)" opacity="0.85"/><circle cx="68"  cy="110" r="38" fill="none" stroke="#22d3a0" stroke-width="1.2"/><circle cx="68"  cy="107" r="13" fill="url(#nn)"/>
<circle cx="126" cy="122" r="36" fill="url(#nc)" opacity="0.78"/><circle cx="126" cy="122" r="36" fill="none" stroke="#22d3a0" stroke-width="1.2"/><circle cx="126" cy="119" r="12" fill="url(#nn)"/>
<circle cx="95"  cy="80"  r="34" fill="url(#nc)" opacity="0.75"/><circle cx="95"  cy="80"  r="34" fill="none" stroke="#22d3a0" stroke-width="1.2"/><circle cx="95"  cy="77"  r="11" fill="url(#nn)"/>

<text x="95"  y="185" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#22d3a0">Normal Cells</text>
<text x="95"  y="200" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#7c7c9e">Uniform · Small nucleus</text>

<!-- Divider -->
<line x1="170" y1="35" x2="170" y2="215" stroke="#1e1e3a" stroke-width="1.5"/>

<!-- Cancer cells -->
<path d="M234 98 Q230 85 241 79 Q247 68 255 76 Q262 66 269 77 Q277 72 275 86 Q286 90 282 104 Q292 112 283 120 Q286 131 274 131 Q271 142 259 139 Q251 149 244 139 Q234 141 233 129 Q221 123 226 110 Q219 103 228 96 Q232 89 234 98Z" fill="url(#cc)" opacity="0.88"/>
<path d="M234 98 Q230 85 241 79 Q247 68 255 76 Q262 66 269 77 Q277 72 275 86 Q286 90 282 104 Q292 112 283 120 Q286 131 274 131 Q271 142 259 139 Q251 149 244 139 Q234 141 233 129 Q221 123 226 110 Q219 103 228 96 Q232 89 234 98Z" fill="none" stroke="#f25c5c" stroke-width="1.2"/>
<ellipse cx="257" cy="108" rx="20" ry="17" fill="url(#cn)" transform="rotate(-12 257 108)"/>
<circle cx="252" cy="104" r="4" fill="#f43f5e" opacity="0.85"/>

<text x="257" y="185" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#f25c5c">Cancer Cells</text>
<text x="257" y="200" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#7c7c9e">Irregular · Large nucleus</text>
</svg>"""
    return f'<div style="border-radius:12px;overflow:hidden;margin:4px 0;">{_svg_img(svg, "210px")}</div>'


# ─── Detection Stages ─────────────────────────────────────────────────────────

def get_detection_stages_svg() -> str:
    svg = """<svg width="680" height="200" viewBox="0 0 680 200" xmlns="http://www.w3.org/2000/svg">
<rect width="680" height="200" fill="#0d0d1a"/>
<text x="340" y="26" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#e8e8f4" letter-spacing="1">EARLY DETECTION SAVES LIVES</text>

<!-- Connecting line -->
<line x1="80" y1="95" x2="600" y2="95" stroke="#1e1e3a" stroke-width="2"/>

<!-- Stage 0 -->
<circle cx="80"  cy="95" r="10" fill="#22d3a0"/>
<circle cx="80"  cy="95" r="16" fill="none" stroke="rgba(34,211,160,0.3)" stroke-width="2"/>
<text x="80"  y="128" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#22d3a0">Stage 0</text>
<text x="80"  y="142" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#7c7c9e">In situ</text>
<text x="80"  y="156" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#22d3a0">~100%</text>

<!-- Stage I -->
<circle cx="240" cy="95" r="16" fill="#84cc16"/>
<circle cx="240" cy="95" r="22" fill="none" stroke="rgba(132,204,22,0.3)" stroke-width="2"/>
<text x="240" y="132" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#84cc16">Stage I</text>
<text x="240" y="146" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#7c7c9e">&lt; 2 cm</text>
<text x="240" y="160" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#84cc16">~99%</text>

<!-- Stage II -->
<circle cx="420" cy="95" r="24" fill="#f59e0b"/>
<circle cx="420" cy="95" r="31" fill="none" stroke="rgba(245,158,11,0.3)" stroke-width="2"/>
<text x="420" y="140" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#f59e0b">Stage II</text>
<text x="420" y="154" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#7c7c9e">2–5 cm</text>
<text x="420" y="168" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#f59e0b">~86%</text>

<!-- Stage III -->
<circle cx="600" cy="95" r="32" fill="#f25c5c"/>
<circle cx="600" cy="95" r="40" fill="none" stroke="rgba(242,92,92,0.3)" stroke-width="2"/>
<text x="600" y="148" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="700" fill="#f25c5c">Stage III</text>
<text x="600" y="162" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#7c7c9e">Spread</text>
<text x="600" y="176" text-anchor="middle" font-family="sans-serif" font-size="9"  fill="#f25c5c">~57%</text>

<!-- Survival label -->
<text x="340" y="192" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#7c7c9e">5-year survival rate</text>
</svg>"""
    return f'<div style="border-radius:12px;overflow:hidden;margin:8px 0;">{_svg_img(svg, "180px")}</div>'


# ─── Risk Banner ─────────────────────────────────────────────────────────────

def get_risk_banner_html(label: str, prob_mal: float) -> str:
    pct       = int(prob_mal * 100)
    is_benign = label == "Benign"
    color     = "#22d3a0" if is_benign else "#f25c5c"
    bg        = "rgba(34,211,160,0.06)" if is_benign else "rgba(242,92,92,0.06)"
    border    = "rgba(34,211,160,0.35)" if is_benign else "rgba(242,92,92,0.35)"
    icon      = "✓" if is_benign else "⚠"
    sub       = ("Non-cancerous indicators. Routine monitoring advised."
                 if is_benign else
                 "Malignant characteristics detected. Specialist consult recommended.")
    return f"""
<div style="background:{bg};border:2px solid {border};border-radius:20px;
     padding:28px 32px;margin:20px 0;display:flex;align-items:center;gap:24px;flex-wrap:wrap;">
  <div style="width:60px;height:60px;border-radius:50%;background:{color}22;
       display:flex;align-items:center;justify-content:center;
       font-size:28px;color:{color};border:2px solid {color};flex-shrink:0;">{icon}</div>
  <div style="flex:1;min-width:200px;">
    <div style="font-size:1.8rem;font-weight:800;color:{color};
         letter-spacing:2px;font-family:sans-serif;">{label.upper()}</div>
    <div style="color:#7c7c9e;font-size:13px;margin-top:4px;font-family:sans-serif;">{sub}</div>
  </div>
  <div style="text-align:center;min-width:80px;">
    <div style="font-size:2.4rem;font-weight:800;color:{color};font-family:sans-serif;line-height:1;">{pct}%</div>
    <div style="color:#7c7c9e;font-size:11px;font-family:sans-serif;">malignancy risk</div>
  </div>
</div>
<div style="margin-top:-8px;margin-bottom:20px;background:#1a1a35;
     border-radius:8px;height:8px;overflow:hidden;">
  <div style="height:100%;width:{min(pct,100)}%;background:linear-gradient(90deg,{color}99,{color});
       border-radius:8px;transition:width 1s ease;"></div>
</div>
"""


# ─── Feature Radar HTML ───────────────────────────────────────────────────────

def get_feature_radar_html(feature_values: dict, feature_stats: dict, feature_names: list) -> str:
    """Render a mini radar chart of normalized mean/worst/SE group scores."""
    groups = {
        "Mean": [f for f in feature_names if "mean" in f],
        "Worst": [f for f in feature_names if "worst" in f],
        "SE": [f for f in feature_names if "error" in f.lower() and "mean" not in f and "worst" not in f],
    }

    def group_score(feats):
        if not feats:
            return 0.5
        vals = []
        for f in feats:
            s = feature_stats.get(f, {})
            mn, mx = s.get("min", 0), s.get("max", 1)
            v = feature_values.get(f, mn)
            vals.append((v - mn) / (mx - mn + 1e-9))
        return sum(vals) / len(vals)

    scores = {g: group_score(fs) for g, fs in groups.items()}

    # Build a simple horizontal bar visual in HTML
    bars = ""
    for g, sc in scores.items():
        pct   = int(sc * 100)
        color = "#f25c5c" if sc > 0.65 else ("#f59e0b" if sc > 0.4 else "#22d3a0")
        bars += f"""
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:12px;color:#7c7c9e;font-family:sans-serif;">{g} Features</span>
            <span style="font-size:12px;font-weight:700;color:{color};font-family:sans-serif;">{pct}%</span>
          </div>
          <div style="height:6px;background:#1a1a35;border-radius:3px;">
            <div style="height:100%;width:{pct}%;background:{color};border-radius:3px;"></div>
          </div>
        </div>
        """

    return f"""
    <div style="background:#12122b;border:1px solid rgba(255,255,255,0.07);
         border-radius:14px;padding:20px 24px;margin:16px 0;">
      <div style="font-size:12px;font-weight:700;color:#7c7c9e;text-transform:uppercase;
           letter-spacing:1px;margin-bottom:16px;font-family:sans-serif;">Feature Group Scores (normalized)</div>
      {bars}
    </div>
    """