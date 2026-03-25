"""
utils/visuals.py - SVG illustrations rendered as base64 img tags for Streamlit.
"""
import base64


def _svg_to_img(svg: str, height: str = "auto") -> str:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return (
        f'<img src="data:image/svg+xml;base64,{b64}" '
        f'style="width:100%;height:{height};display:block;border-radius:12px;" />'
    )


def get_anatomy_svg() -> str:
    svg = """<svg width="680" height="500" viewBox="0 0 680 500" xmlns="http://www.w3.org/2000/svg">
<defs>
  <radialGradient id="sk" cx="50%" cy="40%" r="55%"><stop offset="0%" stop-color="#F5C5A3"/><stop offset="100%" stop-color="#D4956A"/></radialGradient>
  <radialGradient id="ft" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#FAE8C8" stop-opacity="0.9"/><stop offset="100%" stop-color="#E8C87A" stop-opacity="0.7"/></radialGradient>
  <radialGradient id="gl" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#F4A0B0"/><stop offset="100%" stop-color="#D4607A"/></radialGradient>
  <radialGradient id="bg" cx="40%" cy="40%" r="55%"><stop offset="0%" stop-color="#7ED9A0"/><stop offset="100%" stop-color="#2E8B57"/></radialGradient>
  <radialGradient id="mg" cx="40%" cy="40%" r="55%"><stop offset="0%" stop-color="#F97070"/><stop offset="100%" stop-color="#C0202A"/></radialGradient>
  <clipPath id="cl1"><ellipse cx="185" cy="250" rx="140" ry="148"/></clipPath>
  <clipPath id="cl2"><ellipse cx="495" cy="250" rx="140" ry="148"/></clipPath>
</defs>
<rect width="680" height="500" fill="#0f0f1e"/>
<text x="340" y="32" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="600" fill="#e2e8f0">Breast Tissue Cross-Section — Benign vs Malignant</text>
<ellipse cx="185" cy="250" rx="140" ry="148" fill="url(#sk)" opacity="0.92"/>
<g clip-path="url(#cl1)">
  <ellipse cx="150" cy="210" rx="36" ry="26" fill="url(#ft)" opacity="0.8"/>
  <ellipse cx="210" cy="182" rx="28" ry="20" fill="url(#ft)" opacity="0.75"/>
  <ellipse cx="234" cy="230" rx="32" ry="24" fill="url(#ft)" opacity="0.7"/>
  <ellipse cx="160" cy="270" rx="34" ry="22" fill="url(#ft)" opacity="0.8"/>
  <ellipse cx="200" cy="298" rx="28" ry="18" fill="url(#ft)" opacity="0.72"/>
  <path d="M185 348 Q185 300 180 260 Q175 232 170 202 Q167 188 175 178" stroke="#E8A0B8" stroke-width="3.5" fill="none" stroke-linecap="round" opacity="0.85"/>
  <path d="M185 348 Q190 292 196 256 Q206 226 211 196" stroke="#E8A0B8" stroke-width="2.5" fill="none" stroke-linecap="round" opacity="0.75"/>
  <path d="M185 348 Q180 305 168 276 Q156 252 142 238" stroke="#E8A0B8" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.65"/>
  <ellipse cx="182" cy="220" rx="14" ry="10" fill="url(#gl)" opacity="0.85"/>
  <ellipse cx="206" cy="244" rx="12" ry="9" fill="url(#gl)" opacity="0.8"/>
  <ellipse cx="162" cy="244" rx="11" ry="8" fill="url(#gl)" opacity="0.75"/>
  <ellipse cx="163" cy="215" rx="20" ry="16" fill="url(#bg)" opacity="0.93"/>
  <ellipse cx="163" cy="215" rx="20" ry="16" fill="none" stroke="#22c55e" stroke-width="2"/>
  <text x="163" y="212" text-anchor="middle" font-family="sans-serif" font-size="8" font-weight="700" fill="#14532d">BENIGN</text>
  <text x="163" y="222" text-anchor="middle" font-family="sans-serif" font-size="7" fill="#14532d">smooth</text>
</g>
<ellipse cx="185" cy="250" rx="140" ry="148" fill="none" stroke="#C47040" stroke-width="3"/>
<ellipse cx="185" cy="380" rx="10" ry="7" fill="#C47040" opacity="0.85"/>
<line x1="145" y1="215" x2="32" y2="182" stroke="#475569" stroke-width="0.8" stroke-dasharray="4 3"/>
<text x="28" y="178" text-anchor="end" font-family="sans-serif" font-size="11" fill="#94a3b8">Benign lump</text>
<text x="28" y="190" text-anchor="end" font-family="sans-serif" font-size="10" fill="#22c55e">(smooth, movable)</text>
<line x1="155" y1="262" x2="32" y2="260" stroke="#475569" stroke-width="0.8" stroke-dasharray="4 3"/>
<text x="28" y="257" text-anchor="end" font-family="sans-serif" font-size="11" fill="#94a3b8">Fat lobule</text>
<line x1="180" y1="295" x2="32" y2="328" stroke="#475569" stroke-width="0.8" stroke-dasharray="4 3"/>
<text x="28" y="325" text-anchor="end" font-family="sans-serif" font-size="11" fill="#94a3b8">Milk duct</text>
<text x="185" y="428" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#22c55e">BENIGN</text>
<text x="185" y="444" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#94a3b8">Clear borders · Soft · Mobile</text>
<ellipse cx="495" cy="250" rx="140" ry="148" fill="url(#sk)" opacity="0.92"/>
<g clip-path="url(#cl2)">
  <ellipse cx="460" cy="210" rx="36" ry="26" fill="url(#ft)" opacity="0.8"/>
  <ellipse cx="520" cy="182" rx="28" ry="20" fill="url(#ft)" opacity="0.75"/>
  <ellipse cx="544" cy="230" rx="32" ry="24" fill="url(#ft)" opacity="0.7"/>
  <ellipse cx="470" cy="270" rx="34" ry="22" fill="url(#ft)" opacity="0.8"/>
  <ellipse cx="510" cy="298" rx="28" ry="18" fill="url(#ft)" opacity="0.72"/>
  <path d="M495 348 Q495 300 490 260 Q485 232 480 202 Q477 188 485 178" stroke="#E8A0B8" stroke-width="3.5" fill="none" stroke-linecap="round" opacity="0.85"/>
  <path d="M495 348 Q500 292 506 256 Q516 226 521 196" stroke="#E8A0B8" stroke-width="2.5" fill="none" stroke-linecap="round" opacity="0.75"/>
  <path d="M495 348 Q490 305 478 276 Q466 252 452 238" stroke="#E8A0B8" stroke-width="2" fill="none" stroke-linecap="round" opacity="0.65"/>
  <ellipse cx="492" cy="220" rx="14" ry="10" fill="url(#gl)" opacity="0.85"/>
  <ellipse cx="516" cy="244" rx="12" ry="9" fill="url(#gl)" opacity="0.8"/>
  <ellipse cx="472" cy="244" rx="11" ry="8" fill="url(#gl)" opacity="0.75"/>
  <path d="M468 215 Q463 200 473 195 Q478 185 485 192 Q489 180 496 188 Q503 176 509 186 Q515 178 520 190 Q528 185 527 197 Q537 200 533 212 Q541 220 533 226 Q537 236 527 237 Q527 248 516 246 Q511 256 503 249 Q496 258 489 249 Q480 255 478 244 Q467 242 468 231 Q458 225 468 215Z" fill="url(#mg)" opacity="0.93"/>
  <text x="499" y="213" text-anchor="middle" font-family="sans-serif" font-size="8" font-weight="700" fill="#7f1d1d">MALIGNANT</text>
  <text x="499" y="223" text-anchor="middle" font-family="sans-serif" font-size="7" fill="#7f1d1d">irregular</text>
  <path d="M526 226 Q545 222 555 220" stroke="#C02020" stroke-width="1.2" fill="none" stroke-dasharray="3 2" opacity="0.7"/>
  <path d="M524 238 Q540 242 550 248" stroke="#C02020" stroke-width="1" fill="none" stroke-dasharray="3 2" opacity="0.6"/>
  <path d="M470 240 Q458 247 450 252" stroke="#C02020" stroke-width="1" fill="none" stroke-dasharray="3 2" opacity="0.6"/>
</g>
<ellipse cx="495" cy="250" rx="140" ry="148" fill="none" stroke="#C47040" stroke-width="3"/>
<ellipse cx="495" cy="380" rx="10" ry="7" fill="#C47040" opacity="0.85"/>
<line x1="526" y1="210" x2="648" y2="175" stroke="#475569" stroke-width="0.8" stroke-dasharray="4 3"/>
<text x="652" y="171" font-family="sans-serif" font-size="11" fill="#94a3b8">Malignant tumor</text>
<text x="652" y="183" font-family="sans-serif" font-size="10" fill="#ef4444">(irregular, fixed)</text>
<line x1="545" y1="242" x2="648" y2="262" stroke="#475569" stroke-width="0.8" stroke-dasharray="4 3"/>
<text x="652" y="259" font-family="sans-serif" font-size="11" fill="#94a3b8">Invasion zone</text>
<line x1="470" y1="270" x2="648" y2="328" stroke="#475569" stroke-width="0.8" stroke-dasharray="4 3"/>
<text x="652" y="325" font-family="sans-serif" font-size="11" fill="#94a3b8">Distorted ducts</text>
<text x="495" y="428" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700" fill="#ef4444">MALIGNANT</text>
<text x="495" y="444" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#94a3b8">Irregular · Hard · Fixed</text>
<line x1="340" y1="70" x2="340" y2="410" stroke="#334155" stroke-width="1" stroke-dasharray="6 4"/>
</svg>"""
    return f'<div style="border-radius:16px;overflow:hidden;margin:12px 0;">{_svg_to_img(svg, "420px")}</div>'


def get_cell_comparison_svg() -> str:
    svg = """<svg width="680" height="270" viewBox="0 0 680 270" xmlns="http://www.w3.org/2000/svg">
<defs>
  <radialGradient id="nc2" cx="40%" cy="35%" r="60%"><stop offset="0%" stop-color="#a3e4c1"/><stop offset="100%" stop-color="#2E8B57" stop-opacity="0.7"/></radialGradient>
  <radialGradient id="cc2" cx="40%" cy="35%" r="60%"><stop offset="0%" stop-color="#fca5a5"/><stop offset="100%" stop-color="#C0202A" stop-opacity="0.7"/></radialGradient>
  <radialGradient id="nn2" cx="45%" cy="40%" r="50%"><stop offset="0%" stop-color="#86efac"/><stop offset="100%" stop-color="#166534"/></radialGradient>
  <radialGradient id="cn2" cx="45%" cy="40%" r="50%"><stop offset="0%" stop-color="#fda4af"/><stop offset="100%" stop-color="#9f1239"/></radialGradient>
</defs>
<rect width="680" height="270" fill="#0f0f1e"/>
<text x="340" y="26" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="600" fill="#e2e8f0">Cell Comparison: Normal vs Cancer</text>
<circle cx="120" cy="135" r="48" fill="url(#nc2)" opacity="0.85"/>
<circle cx="120" cy="135" r="48" fill="none" stroke="#22c55e" stroke-width="1.5"/>
<circle cx="120" cy="132" r="16" fill="url(#nn2)"/>
<circle cx="196" cy="148" r="46" fill="url(#nc2)" opacity="0.8"/>
<circle cx="196" cy="148" r="46" fill="none" stroke="#22c55e" stroke-width="1.5"/>
<circle cx="196" cy="145" r="15" fill="url(#nn2)"/>
<circle cx="157" cy="95" r="44" fill="url(#nc2)" opacity="0.78"/>
<circle cx="157" cy="95" r="44" fill="none" stroke="#22c55e" stroke-width="1.5"/>
<circle cx="157" cy="92" r="14" fill="url(#nn2)"/>
<text x="158" y="218" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#22c55e">Normal Cells</text>
<text x="158" y="234" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#94a3b8">Uniform size · Small nucleus · Organized</text>
<line x1="340" y1="45" x2="340" y2="235" stroke="#334155" stroke-width="1" stroke-dasharray="5 4"/>
<path d="M478 105 Q496 87 514 95 Q534 83 539 103 Q555 100 553 118 Q568 128 560 145 Q572 158 556 165 Q558 182 540 182 Q532 198 515 191 Q500 202 494 188 Q476 190 474 175 Q458 168 462 151 Q448 140 457 125 Q451 112 466 106 Q470 99 478 105Z" fill="url(#cc2)" opacity="0.87"/>
<path d="M478 105 Q496 87 514 95 Q534 83 539 103 Q555 100 553 118 Q568 128 560 145 Q572 158 556 165 Q558 182 540 182 Q532 198 515 191 Q500 202 494 188 Q476 190 474 175 Q458 168 462 151 Q448 140 457 125 Q451 112 466 106 Q470 99 478 105Z" fill="none" stroke="#ef4444" stroke-width="1.5"/>
<ellipse cx="512" cy="146" rx="26" ry="22" fill="url(#cn2)" transform="rotate(-15 512 146)"/>
<circle cx="506" cy="141" r="5" fill="#f43f5e" opacity="0.9"/>
<circle cx="519" cy="152" r="4" fill="#f43f5e" opacity="0.8"/>
<path d="M574 82 Q587 72 598 78 Q607 69 611 82 Q621 80 619 92 Q628 100 621 108 Q627 118 616 122 Q614 132 602 128 Q594 136 587 128 Q578 130 576 120 Q566 115 570 104 Q563 95 572 89Z" fill="url(#cc2)" opacity="0.8"/>
<ellipse cx="596" cy="104" rx="13" ry="11" fill="url(#cn2)" transform="rotate(10 596 104)"/>
<text x="518" y="218" text-anchor="middle" font-family="sans-serif" font-size="13" font-weight="700" fill="#ef4444">Cancer Cells</text>
<text x="518" y="234" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#94a3b8">Irregular shape · Large nucleus · Disorganized</text>
</svg>"""
    return f'<div style="border-radius:12px;overflow:hidden;margin:8px 0;">{_svg_to_img(svg, "240px")}</div>'


def get_detection_stages_svg() -> str:
    svg = """<svg width="680" height="215" viewBox="0 0 680 215" xmlns="http://www.w3.org/2000/svg">
<rect width="680" height="215" fill="#0f0f1e"/>
<text x="340" y="28" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="600" fill="#e2e8f0">Early Detection Saves Lives</text>
<circle cx="80"  cy="105" r="10" fill="#22c55e" opacity="0.92"/>
<circle cx="80"  cy="105" r="14" fill="none" stroke="#22c55e" stroke-width="2" opacity="0.35"/>
<text x="80"  y="138" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#22c55e">Stage 0</text>
<text x="80"  y="152" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#94a3b8">Cells in ducts</text>
<circle cx="253" cy="105" r="16" fill="#84cc16" opacity="0.92"/>
<circle cx="253" cy="105" r="22" fill="none" stroke="#84cc16" stroke-width="2" opacity="0.35"/>
<text x="253" y="144" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#84cc16">Stage I</text>
<text x="253" y="158" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#94a3b8">Tumor &lt; 2cm</text>
<circle cx="430" cy="105" r="23" fill="#f59e0b" opacity="0.92"/>
<circle cx="430" cy="105" r="30" fill="none" stroke="#f59e0b" stroke-width="2" opacity="0.35"/>
<text x="430" y="152" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#f59e0b">Stage II</text>
<text x="430" y="166" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#94a3b8">Tumor 2-5cm</text>
<circle cx="610" cy="105" r="32" fill="#ef4444" opacity="0.92"/>
<circle cx="610" cy="105" r="40" fill="none" stroke="#ef4444" stroke-width="2" opacity="0.35"/>
<text x="610" y="160" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="700" fill="#ef4444">Stage III</text>
<text x="610" y="174" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#94a3b8">Large / spread</text>
<line x1="96"  y1="105" x2="230" y2="105" stroke="#475569" stroke-width="1.5" stroke-dasharray="4 3"/>
<line x1="276" y1="105" x2="406" y2="105" stroke="#475569" stroke-width="1.5" stroke-dasharray="4 3"/>
<line x1="455" y1="105" x2="578" y2="105" stroke="#475569" stroke-width="1.5" stroke-dasharray="4 3"/>
<text x="340" y="205" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#6c63ff">★ Early detection (Stage 0-I) has 99%+ survival rate</text>
</svg>"""
    return f'<div style="border-radius:12px;overflow:hidden;margin:8px 0;">{_svg_to_img(svg, "195px")}</div>'


def get_risk_banner_svg(label: str, prob_mal: float) -> str:
    pct = int(prob_mal * 100)
    if label == "Benign":
        border, bg, title = "#22c55e", "#052e16", "BENIGN"
        sub    = "Tumor appears non-cancerous based on input features"
        icon_d = "M5 13l4 4L19 7"
    else:
        border, bg, title = "#ef4444", "#2d0a0a", "MALIGNANT"
        sub    = "Malignant indicators detected — please consult a specialist"
        icon_d = "M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"

    return f"""
<div style="border:2px solid {border};border-radius:16px;padding:24px;
     background:{bg};margin:16px 0;">
  <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none"
      stroke="{border}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="{icon_d}"/>
    </svg>
    <div style="flex:1;">
      <div style="font-size:26px;font-weight:800;color:{border};
           letter-spacing:2px;font-family:sans-serif;">{title}</div>
      <div style="color:#94a3b8;font-size:13px;margin-top:4px;font-family:sans-serif;">{sub}</div>
    </div>
    <div style="text-align:center;">
      <div style="font-size:34px;font-weight:800;color:{border};font-family:sans-serif;">{pct}%</div>
      <div style="color:#94a3b8;font-size:11px;font-family:sans-serif;">malignancy risk</div>
    </div>
  </div>
  <div style="margin-top:16px;background:#1e293b;border-radius:8px;height:10px;overflow:hidden;">
    <div style="height:100%;width:{min(pct,100)}%;background:{border};border-radius:8px;"></div>
  </div>
</div>
"""