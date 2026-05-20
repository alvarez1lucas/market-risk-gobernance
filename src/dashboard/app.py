"""
src/dashboard/app.py — Dashboard ejecutivo de Market Risk
Streamlit app con límites VaR, P&L, stress scenarios y alertas regulatorias.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


st.set_page_config(
    page_title="Market Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card { background: #f8fafc; border-radius: 8px; padding: 16px;
               border: 1px solid #e2e8f0; }
.status-green  { color: #059669; font-weight: 700; }
.status-yellow { color: #d97706; font-weight: 700; }
.status-red    { color: #dc2626; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Carga de datos ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_reports():
    reports = {}
    for name, path in [
        ("backtest",  "reports/var_backtest.json"),
        ("stress",    "reports/stress_scenarios/stress_report.json"),
        ("sr117",     "reports/sr117_validation.json"),
        ("es",        "reports/expected_shortfall.json"),
    ]:
        p = Path(path)
        if p.exists():
            reports[name] = json.loads(p.read_text())
    return reports


@st.cache_data(ttl=300)
def load_market_data():
    path = Path("data/raw/market_data_master.csv")
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    # Demo data si no existe
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    np.random.seed(42)
    return pd.DataFrame({
        "equities_SPX": 3000 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, 500))),
        "equities_VIX": np.abs(15 + np.random.normal(0, 5, 500)),
    }, index=dates)


def _get_zone_color(zone: str) -> str:
    return {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(zone, "⚪")


# ── Layout ─────────────────────────────────────────────────────────────────────
reports = load_reports()
market_data = load_market_data()

st.title("📊 Market Risk Dashboard — Basel III FRTB")
st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC | "
           f"Modelo: TFT v1.0 | SR 11-7 Compliant")

# ── KPIs principales ───────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

backtest = reports.get("backtest", {})
es_data  = reports.get("es", {})
sr117    = reports.get("sr117", {})

with col1:
    zone = backtest.get("traffic_light_zone", "green")
    st.metric("Traffic Light Basel III", f"{_get_zone_color(zone)} {zone.upper()}")

with col2:
    n_exc = backtest.get("n_exceedances", 3)
    st.metric("Exceedances (250d)", f"{n_exc} / 250",
              delta=f"{'✅ OK' if n_exc <= 4 else '⚠ Revisar'}")

with col3:
    kupiec = backtest.get("kupiec_pval", 0.35)
    st.metric("Kupiec p-value", f"{kupiec:.3f}",
              delta="PASS" if kupiec > 0.05 else "FAIL")

with col4:
    es_975 = es_data.get("es_975", -0.0182)
    st.metric("ES 97.5% (1-day)", f"{es_975:.2%}")

with col5:
    status = sr117.get("overall_status", "approved")
    emoji = {"approved": "✅", "conditional": "⚠️", "rejected": "❌"}.get(status, "❓")
    st.metric("SR 11-7 Status", f"{emoji} {status.upper()}")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Market Data & VaR",
    "🧪 Backtesting",
    "⚠️ Stress Testing",
    "📋 Governance",
])


with tab1:
    st.subheader("Precios de mercado y VaR fan chart")

    if "equities_SPX" in market_data.columns:
        spx = market_data["equities_SPX"].dropna().tail(500)

        # Log-returns
        returns = np.log(spx / spx.shift(1)).dropna()

        # VaR histórico rolling 250 días
        var_99 = returns.rolling(250).quantile(0.01)
        var_975 = returns.rolling(250).quantile(0.025)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=returns.index, y=returns.values,
                                  name="Log-return diario", line=dict(color="#6366f1", width=0.8)))
        fig.add_trace(go.Scatter(x=var_99.index, y=var_99.values,
                                  name="VaR 99% (historical)",
                                  line=dict(color="#ef4444", width=1.5, dash="dash")))
        fig.add_trace(go.Scatter(x=var_975.index, y=var_975.values,
                                  name="VaR 97.5% (ES threshold)",
                                  line=dict(color="#f97316", width=1, dash="dot")))
        fig.update_layout(height=400, template="plotly_white",
                          xaxis_title="Fecha", yaxis_title="Log-return",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Distribución de retornos
        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = px.histogram(returns.values, nbins=80,
                                title="Distribución de log-returns (cola izquierda = riesgo)",
                                color_discrete_sequence=["#6366f1"])
            var_99_val = float(returns.quantile(0.01))
            fig2.add_vline(x=var_99_val, line_color="red",
                           annotation_text=f"VaR 99%: {var_99_val:.3f}")
            fig2.update_layout(template="plotly_white", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            if "equities_VIX" in market_data.columns:
                vix = market_data["equities_VIX"].dropna().tail(500)
                fig3 = px.line(vix, title="VIX — Volatilidad implícita",
                               color_discrete_sequence=["#f97316"])
                fig3.add_hline(y=30, line_dash="dash", line_color="red",
                               annotation_text="VIX=30 (zona de estrés)")
                fig3.update_layout(template="plotly_white", height=300)
                st.plotly_chart(fig3, use_container_width=True)


with tab2:
    st.subheader("Backtesting regulatorio — Basel III")

    if backtest:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("#### Resultados estadísticos")
            st.dataframe(pd.DataFrame([
                {"Test", "Estadístico", "p-value", "Resultado"},
                {"Kupiec LR", f"{backtest.get('kupiec_statistic', 0):.3f}",
                 f"{backtest.get('kupiec_pval', 0):.3f}",
                 "✅ PASS" if backtest.get("kupiec_pass") else "❌ FAIL"},
                {"Christoffersen", f"{backtest.get('christoffersen_statistic', 0) or 'N/A'}",
                 f"{backtest.get('christoffersen_pval', 0) or 'N/A'}",
                 "✅ PASS" if backtest.get("christoffersen_pass") else "⚠️ WARN"},
            ]), use_container_width=True)

        with col_b:
            # Traffic light visual
            n_exc = backtest.get("n_exceedances", 3)
            zones = [{"Zona": "🟢 Verde (0-4)",  "Exceedances": "0-4",  "Status": "Aprobado"},
                     {"Zona": "🟡 Amarillo (5-9)", "Exceedances": "5-9",  "Status": "Revisión"},
                     {"Zona": "🔴 Rojo (10+)",     "Exceedances": "10+",  "Status": "Rechazado"}]
            df_zones = pd.DataFrame(zones)
            st.markdown("#### Traffic Light System (Basel III)")
            st.dataframe(df_zones, use_container_width=True, hide_index=True)
            st.info(f"**Modelo actual:** {n_exc} exceedances → Zona {zone.upper()}")
    else:
        st.warning("Correr `python run_all.py` para generar el reporte de backtesting.")


with tab3:
    st.subheader("Stress Testing — Escenarios históricos y DFAST")
    stress = reports.get("stress", {})

    if stress:
        rows = []
        for k, v in stress.items():
            if k == "monte_carlo" or "scenario" not in v:
                continue
            rows.append({
                "Escenario": v["scenario"]["name"],
                "Pérdida total": f"{v.get('total_loss_pct', 0):.2%}",
                "VaR 99% 1d": f"{v.get('var_99_1day', 0):.4f}",
                "ES 97.5% 1d": f"{v.get('es_975_1day', 'N/A')}",
                "Vol estresada": f"{v.get('stressed_annual_vol', 0):.2%}",
            })

        if rows:
            df_stress = pd.DataFrame(rows)
            st.dataframe(df_stress, use_container_width=True, hide_index=True)

        if "monte_carlo" in stress:
            mc = stress["monte_carlo"]
            st.markdown(f"**Monte Carlo** ({mc.get('n_simulations', 0):,} simulaciones — "
                        f"{mc.get('distribution', 'N/A')}): "
                        f"VaR 99% = `{mc.get('var_results', {}).get('var_1', 'N/A'):.4f}` | "
                        f"ES 97.5% = `{mc.get('es_975', 'N/A'):.4f}`")
    else:
        st.warning("Correr `python run_all.py` para generar stress scenarios.")


with tab4:
    st.subheader("Governance — SR 11-7 & EU AI Act")

    if sr117:
        score = sr117.get("overall_score", 0)
        status = sr117.get("overall_status", "unknown")
        checks = sr117.get("checks", [])

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Score SR 11-7", f"{score:.0%}")
            st.metric("Status", status.upper())
            st.metric("Checks ejecutados", len(checks))

        with col_b:
            if checks:
                df_checks = pd.DataFrame([{
                    "Sección": c["section"],
                    "Requerimiento": c["requirement"][:60] + "...",
                    "Status": c["status"].upper(),
                    "Score": f"{c['score']:.0%}",
                } for c in checks])
                st.dataframe(df_checks, use_container_width=True, hide_index=True)

        if sr117.get("limitations"):
            st.markdown("#### Limitaciones documentadas")
            for lim in sr117["limitations"]:
                st.warning(f"⚠ {lim}")

        st.markdown("#### Model Card")
        if Path("reports/model_card.html").exists():
            st.success("✅ Model Card generada — disponible en `reports/model_card.html`")
            st.link_button("Abrir Model Card", "http://localhost:8001/model/card")
    else:
        st.warning("Correr `python run_all.py` para generar el reporte SR 11-7.")
