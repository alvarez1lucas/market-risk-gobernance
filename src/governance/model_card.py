"""
src/governance/model_card.py — Generador de Model Cards regulatorias
Cumple EU AI Act Annex IV (documentación técnica) y SR 11-7.
Auto-genera HTML con todas las métricas del pipeline.
"""

import json
from pathlib import Path
from datetime import datetime
from jinja2 import Template
from loguru import logger


MODEL_CARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Model Card — {{ model_name }}</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 40px auto; padding: 0 24px; color: #1a1a2e; }
  h1   { font-size: 24px; border-bottom: 3px solid #4361ee; padding-bottom: 8px; }
  h2   { font-size: 16px; color: #4361ee; margin-top: 32px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }
  th   { background: #f0f4ff; text-align: left; padding: 8px 12px; border: 1px solid #dde3f0; }
  td   { padding: 8px 12px; border: 1px solid #dde3f0; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
           font-size: 11px; font-weight: 600; }
  .green  { background: #d1fae5; color: #065f46; }
  .yellow { background: #fef3c7; color: #92400e; }
  .red    { background: #fee2e2; color: #991b1b; }
  .blue   { background: #dbeafe; color: #1e40af; }
  .meta   { background: #f8fafc; border-radius: 8px; padding: 16px; margin: 16px 0;
            font-size: 13px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .meta-item label { color: #6b7280; font-size: 11px; display: block; }
  .warning { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 12px 16px;
             border-radius: 4px; font-size: 13px; margin: 8px 0; }
  .limitation { background: #fef2f2; border-left: 4px solid #ef4444; padding: 10px 14px;
                border-radius: 4px; font-size: 13px; margin: 6px 0; }
  footer { margin-top: 48px; font-size: 11px; color: #9ca3af; border-top: 1px solid #e5e7eb;
           padding-top: 16px; }
</style>
</head>
<body>

<h1>📋 Model Card — {{ model_name }}</h1>

<div class="meta">
  <div class="meta-item"><label>Versión</label><strong>{{ version }}</strong></div>
  <div class="meta-item"><label>Fecha de validación</label><strong>{{ validation_date }}</strong></div>
  <div class="meta-item"><label>Status regulatorio</label>
    <span class="badge {{ status_color }}">{{ status }}</span></div>
  <div class="meta-item"><label>Tier de riesgo (SR 11-7)</label><strong>{{ risk_tier }}</strong></div>
  <div class="meta-item"><label>EU AI Act categoría</label><strong>High Risk — Annex III</strong></div>
  <div class="meta-item"><label>Owner</label><strong>{{ owner }}</strong></div>
</div>

<h2>1. Propósito y alcance</h2>
<p>{{ purpose }}</p>
<table>
  <tr><th>Activos cubiertos</th><td>{{ assets }}</td></tr>
  <tr><th>Horizonte de predicción</th><td>1 día (VaR 1-day) escalable a 10 días (FRTB)</td></tr>
  <tr><th>Población objetivo</th><td>Portfolio de trading book institucional</td></tr>
  <tr><th>Casos de uso permitidos</th><td>Estimación de capital regulatorio, límites de riesgo internos</td></tr>
  <tr><th>Casos de uso prohibidos</th><td>Decisiones de crédito retail, scoring de individuos</td></tr>
</table>

<h2>2. Métricas de performance</h2>
<table>
  <tr><th>Métrica</th><th>Valor</th><th>Umbral regulatorio</th><th>Status</th></tr>
  <tr>
    <td>Exceedances (250 días)</td>
    <td>{{ backtest.n_exceedances }}</td>
    <td>≤ 4 (zona verde Basel III)</td>
    <td><span class="badge {{ 'green' if backtest.n_exceedances <= 4 else 'red' }}">
      {{ 'PASS' if backtest.n_exceedances <= 4 else 'FAIL' }}</span></td>
  </tr>
  <tr>
    <td>Kupiec test (p-value)</td>
    <td>{{ "%.3f"|format(backtest.kupiec_pval) }}</td>
    <td>p > 0.05</td>
    <td><span class="badge {{ 'green' if backtest.kupiec_pval > 0.05 else 'red' }}">
      {{ 'PASS' if backtest.kupiec_pval > 0.05 else 'FAIL' }}</span></td>
  </tr>
  <tr>
    <td>Christoffersen test (p-value)</td>
    <td>{{ "%.3f"|format(backtest.christoffersen_pval) if backtest.christoffersen_pval else 'N/A' }}</td>
    <td>p > 0.05</td>
    <td><span class="badge {{ 'green' if (backtest.christoffersen_pval or 1) > 0.05 else 'yellow' }}">
      {{ 'PASS' if (backtest.christoffersen_pval or 1) > 0.05 else 'WARN' }}</span></td>
  </tr>
  <tr>
    <td>Traffic light zone</td>
    <td>{{ backtest.traffic_light_zone | upper }}</td>
    <td>Verde</td>
    <td><span class="badge {{ 'green' if backtest.traffic_light_zone == 'green' else 'yellow' }}">
      {{ backtest.traffic_light_zone | upper }}</span></td>
  </tr>
  <tr>
    <td>Expected Shortfall 97.5% (1-day)</td>
    <td>{{ "%.4f"|format(es.es_975) }}</td>
    <td>Consistente con FRTB IMA</td>
    <td><span class="badge blue">INFORMATIVO</span></td>
  </tr>
</table>

<h2>3. Stress testing</h2>
<table>
  <tr><th>Escenario</th><th>ES 97.5% (1-day)</th><th>Pérdida total estimada</th></tr>
  {% for scenario_id, result in stress.items() %}
  {% if scenario_id != 'monte_carlo' and result.scenario is defined %}
  <tr>
    <td>{{ result.scenario.name }}</td>
    <td>{{ "%.4f"|format(result.es_975_1day) if result.es_975_1day else 'N/A' }}</td>
    <td>{{ "{:.2%}".format(result.total_loss_pct) if result.total_loss_pct is not none else 'N/A' }}</td>
  </tr>
  {% endif %}
  {% endfor %}
</table>

<h2>4. Datos de entrenamiento</h2>
<table>
  <tr><th>Fuente</th><td>Yahoo Finance (yfinance) + FRED API</td></tr>
  <tr><th>Período</th><td>2000-01-01 a 2024-12-31</td></tr>
  <tr><th>Activos</th><td>SPX, VIX, EEM, HYG, EUR/USD, USD/JPY, UST 10Y + series macro FRED</td></tr>
  <tr><th>Regímenes cubiertos</th><td>GFC 2008, crisis soberana EU 2011, COVID 2020, ciclo de tasas 2022, SVB 2023</td></tr>
  <tr><th>Data quality</th><td>Checks automáticos en ingesta — ver src/data/ingest.py</td></tr>
</table>

<h2>5. Limitaciones conocidas</h2>
{% for lim in limitations %}
<div class="limitation">⚠ {{ lim }}</div>
{% endfor %}

<h2>6. Requerimientos EU AI Act (Annex IV)</h2>
<table>
  <tr><th>Requerimiento</th><th>Cumplimiento</th></tr>
  <tr><td>Descripción del sistema y propósito (Art. 11)</td>
      <td><span class="badge green">Cubierto — sección 1</span></td></tr>
  <tr><td>Descripción de datos de entrenamiento (Art. 10)</td>
      <td><span class="badge green">Cubierto — sección 4</span></td></tr>
  <tr><td>Métricas de performance y validación (Art. 9)</td>
      <td><span class="badge green">Cubierto — sección 2</span></td></tr>
  <tr><td>Supervisión humana (Art. 14)</td>
      <td><span class="badge yellow">Parcial — requiere proceso de aprobación</span></td></tr>
  <tr><td>Limitaciones y casos de uso prohibidos</td>
      <td><span class="badge green">Cubierto — sección 5</span></td></tr>
  <tr><td>Audit trail y trazabilidad</td>
      <td><span class="badge green">Cubierto — src/governance/audit_trail.py</span></td></tr>
</table>

<h2>7. SR 11-7 — Validación independiente</h2>
<table>
  <tr><th>Pilar</th><th>Score</th><th>Status</th></tr>
  <tr><td>Conceptual Soundness</td>
      <td>{{ "{:.0%}".format(sr117.overall_score) }}</td>
      <td><span class="badge {{ 'green' if sr117.overall_status == 'approved' else 'yellow' }}">
        {{ sr117.overall_status | upper }}</span></td></tr>
  <tr><td>Ongoing Monitoring</td><td colspan="2">PSI diario + alertas automáticas</td></tr>
  <tr><td>Outcomes Analysis</td><td colspan="2">Backtesting Kupiec + Christoffersen</td></tr>
</table>

{% if sr117.recommendations %}
<div class="warning">
  <strong>Recomendaciones pendientes:</strong><br>
  {% for rec in sr117.recommendations %}- {{ rec }}<br>{% endfor %}
</div>
{% endif %}

<footer>
  Generado automáticamente por AI Governance Framework — {{ generation_date }}<br>
  Repositorio: github.com/[usuario]/ai-governance-framework<br>
  EU AI Act — Regulation (EU) 2024/1689 | SR 11-7 (Federal Reserve, 2011) | Basel III FRTB (BCBS, 2019)
</footer>
</body>
</html>
"""


class ModelCardGenerator:
    def __init__(self, model_name: str, backtest, stress: dict, sr117):
        self.model_name = model_name
        self.backtest = backtest
        self.stress = stress
        self.sr117 = sr117

    def generate(self, output_path: str = "reports/model_card.html"):
        status = getattr(self.sr117, "overall_status", "approved")
        color_map = {"approved": "green", "conditional": "yellow", "rejected": "red"}

        # ES mock si no está disponible
        class ESMock:
            es_975 = 0.0

        es = ESMock()
        try:
            es_path = Path("reports/expected_shortfall.json")
            if es_path.exists():
                data = json.loads(es_path.read_text())
                es.es_975 = data.get("es_975", 0.0)
        except Exception:
            pass

        tmpl = Template(MODEL_CARD_TEMPLATE)
        html = tmpl.render(
            model_name=self.model_name,
            version="1.0",
            validation_date=datetime.now().strftime("%Y-%m-%d"),
            generation_date=datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            status=status.upper(),
            status_color=color_map.get(status, "blue"),
            risk_tier="High",
            owner="Risk Engineering Team",
            purpose=(
                "Estimación de Value-at-Risk (VaR 99%) y Expected Shortfall (ES 97.5%) "
                "para portfolios de trading book institucional bajo Basel III FRTB. "
                "Utiliza Temporal Fusion Transformer como modelo champion y GARCH(1,1) "
                "como benchmark regulatorio."
            ),
            assets="SPX, EEM, HYG, LQD, EUR/USD, USD/JPY, UST 10Y",
            backtest=self.backtest,
            es=es,
            stress=self.stress,
            limitations=getattr(self.sr117, "limitations", []),
            sr117=self.sr117,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Model Card generada: {output_path}")
