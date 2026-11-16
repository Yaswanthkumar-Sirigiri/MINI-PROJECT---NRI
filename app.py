import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup + styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="FreshGuard AI", page_icon="🥬", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #12352b 0%, #0b1210 55%) fixed; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { background: #0e1714; border-right: 1px solid #1d2b26; }

.hero { padding: 6px 0 18px 0; }
.hero h1 { font-size: 2.5rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;
           background: linear-gradient(90deg,#6ee7b7,#34d399 40%,#a7f3d0); -webkit-background-clip: text;
           -webkit-text-fill-color: transparent; }
.hero p { color: #8aa69b; margin: 4px 0 0 0; font-size: 1.02rem; }

.card { background: linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
        border: 1px solid rgba(255,255,255,.08); border-radius: 18px; padding: 20px 22px;
        backdrop-filter: blur(6px); height: 100%; }
.card .label { color: #8aa69b; font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 600; }
.card .big { font-size: 3rem; font-weight: 800; line-height: 1.05; margin-top: 6px; }
.card .sub { color: #9db5ab; font-size: .92rem; margin-top: 4px; }

.pill { display:inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 1.05rem; }
.pill.Low    { background: rgba(52,211,153,.15); color: #34d399; border: 1px solid rgba(52,211,153,.4); }
.pill.Medium { background: rgba(251,191,36,.15); color: #fbbf24; border: 1px solid rgba(251,191,36,.4); }
.pill.High   { background: rgba(248,113,113,.15); color: #f87171; border: 1px solid rgba(248,113,113,.4); }

.tip { display:flex; gap:12px; align-items:flex-start; background: rgba(255,255,255,.04);
       border: 1px solid rgba(255,255,255,.07); border-left: 4px solid var(--c); border-radius: 12px;
       padding: 12px 16px; margin-bottom: 10px; color: #d5e5dd; font-size: .95rem; }
.tip b { color: #fff; }

div[data-testid="stTabs"] button { font-weight: 600; }
.stDownloadButton button, .stButton button { border-radius: 10px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

COLORS = {"Low": "#34d399", "Medium": "#fbbf24", "High": "#f87171"}
BASE = Path(__file__).parent


# ----------------------------------------------------------------------------
# Model loading + helpers
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load(BASE / "spoilage_risk_model.pkl")
    reg = joblib.load(BASE / "shelf_life_model.pkl")
    enc = joblib.load(BASE / "risk_label_encoder.pkl")
    feats = joblib.load(BASE / "feature_columns.pkl")
    return clf, reg, enc, feats


clf, reg, enc, FEATURES = load_models()
CLASSES = list(enc.classes_)  # order used by predict_proba

FOODS = ["Apples", "Berries", "Cheese", "Chicken", "Citrus Fruits", "Fish", "Leafy Vegetables", "Milk", "Tomatoes", "Yogurt"]
EMOJI = {"Apples": "🍎", "Berries": "🫐", "Cheese": "🧀", "Chicken": "🍗", "Citrus Fruits": "🍊",
         "Fish": "🐟", "Leafy Vegetables": "🥬", "Milk": "🥛", "Tomatoes": "🍅", "Yogurt": "🍦"}
PACKS = ["Modified Atmosphere", "Open", "Plastic Wrap", "Vacuum Sealed"]
CONDS = ["Ambient", "Cold Storage", "Refrigerated"]


def predict(df: pd.DataFrame):
    X = df[FEATURES]
    proba = clf.predict_proba(X)
    labels = enc.inverse_transform(np.argmax(proba, axis=1))
    days = np.clip(reg.predict(X), 0, None)
    p = pd.DataFrame(proba, columns=CLASSES)
    return labels, p, days


def spoilage_index(p: pd.Series) -> float:
    """0-100 index: High counts fully, Medium half."""
    return float(100 * (p["High"] + 0.5 * p["Medium"]))


def gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "", "font": {"size": 44, "color": "#ecfdf5"}, "valueformat": ".0f"},
        title={"text": title, "font": {"size": 14, "color": "#8aa69b"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#4b635a", "tickfont": {"color": "#8aa69b"}},
               "bar": {"color": "#ecfdf5", "thickness": 0.18},
               "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
               "steps": [{"range": [0, 33], "color": "rgba(52,211,153,.55)"},
                         {"range": [33, 66], "color": "rgba(251,191,36,.55)"},
                         {"range": [66, 100], "color": "rgba(248,113,113,.6)"}],
               "threshold": {"line": {"color": "#ffffff", "width": 4}, "thickness": 0.85, "value": value}}))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      font={"family": "Inter"})
    return fig


def style(fig, h=340):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color="#b9cfc5"), margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def card(label, big, sub="", color="#ecfdf5"):
    return f'<div class="card"><div class="label">{label}</div><div class="big" style="color:{color}">{big}</div><div class="sub">{sub}</div></div>'


# ----------------------------------------------------------------------------
# Sidebar inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧪 Item details")
    food = st.selectbox("Food category", FOODS, format_func=lambda f: f"{EMOJI[f]}  {f}")
    pack = st.selectbox("Packaging", PACKS, index=2)
    cond = st.selectbox("Storage condition", CONDS, index=2)
    pres = st.radio("Preservative used", ["No", "Yes"], horizontal=True)

    st.markdown("### 🌡️ Environment")
    temp = st.slider("Storage temperature (°C)", -5.0, 35.0, 5.0, 0.1)
    hum = st.slider("Storage humidity (%)", 30.0, 100.0, 80.0, 0.5)

    st.markdown("### ⏱️ Timeline")
    dur = st.slider("Storage duration (days)", 0.0, 50.0, 5.0, 0.1)
    packed = st.slider("Days since packed", 0.0, 50.0, 6.0, 0.1)
    delay = st.slider("Transport delay (hours)", 0.0, 25.0, 4.0, 0.1)

    st.markdown("### ✅ Quality")
    fresh = st.slider("Initial freshness score", 0, 100, 85)
    handling = st.slider("Handling quality (1-5)", 1, 5, 3)

current = {
    "food_category": food, "initial_freshness_score": fresh, "storage_temperature_c": temp,
    "storage_humidity_pct": hum, "storage_duration_days": dur, "packaging_type": pack,
    "transportation_delay_hours": delay, "storage_condition": cond, "handling_quality_score": handling,
    "preservative_used": pres, "days_since_packed": packed,
}
row = pd.DataFrame([current])
label, proba, days = predict(row)
label, days, p = label[0], float(days[0]), proba.iloc[0]
index = spoilage_index(p)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>🥬 FreshGuard AI</h1>'
    "<p>Smart food spoilage prediction: risk level and remaining shelf life, powered by machine learning.</p></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "🧭 What-if analysis", "📦 Batch prediction", "ℹ️ About"])

# ----------------------------------------------------------------------------
# TAB 1: prediction
# ----------------------------------------------------------------------------
with tab1:
    c1, c2, c3 = st.columns([1.1, 1, 1.1])
    with c1:
        st.markdown(
            f'<div class="card"><div class="label">Predicted spoilage risk</div>'
            f'<div style="margin:14px 0 10px 0"><span class="pill {label}">{label} risk</span></div>'
            f'<div class="sub">{EMOJI[food]} {food} · {pack} · {cond}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        col = "#34d399" if days > 7 else "#fbbf24" if days > 2 else "#f87171"
        st.markdown(card("Remaining shelf life", f"{days:.1f}<span style='font-size:1.2rem;color:#8aa69b'> days</span>",
                         "Estimated by the regression model", col), unsafe_allow_html=True)
    with c3:
        st.markdown(card("Model confidence", f"{p.max()*100:.0f}%", f"in the “{label}” class"), unsafe_allow_html=True)

    st.write("")
    g, b = st.columns([1, 1.15])
    with g:
        st.plotly_chart(gauge(index, "SPOILAGE INDEX"), width="stretch")
    with b:
        fig = go.Figure(go.Bar(
            x=[p[c] * 100 for c in ["Low", "Medium", "High"]], y=["Low", "Medium", "High"], orientation="h",
            marker_color=[COLORS[c] for c in ["Low", "Medium", "High"]],
            text=[f"{p[c]*100:.0f}%" for c in ["Low", "Medium", "High"]], textposition="outside"))
        fig.update_layout(title="Class probabilities", xaxis=dict(range=[0, 110], ticksuffix="%"))
        st.plotly_chart(style(fig, 270), width="stretch")

    st.markdown("#### 💡 Recommendations")
    tips = []
    if temp > 8 and cond != "Ambient":
        tips.append(("#f87171", f"Storage temperature is <b>{temp:.1f} °C</b>, above the safe cold-chain range. Lower it toward 2-4 °C."))
    if cond == "Ambient" and food in ["Chicken", "Fish", "Milk", "Yogurt", "Cheese"]:
        tips.append(("#f87171", f"<b>{food}</b> is highly perishable; ambient storage sharply cuts its shelf life. Move it to refrigeration."))
    if pack in ["Open", "Plastic Wrap"] and food in ["Fish", "Chicken", "Leafy Vegetables"]:
        tips.append(("#fbbf24", "Switch to <b>vacuum sealed</b> or <b>modified atmosphere</b> packaging to slow spoilage."))
    if delay > 8:
        tips.append(("#fbbf24", f"Transport delay of <b>{delay:.0f} h</b> is long. Prioritise faster logistics for this batch."))
    if hum > 92 or hum < 50:
        tips.append(("#fbbf24", f"Humidity at <b>{hum:.0f}%</b> is outside the typical 60-90% comfort band."))
    if handling <= 2:
        tips.append(("#fbbf24", "Low handling quality. Review handling and hygiene practices."))
    if label == "High" or days < 2:
        tips.append(("#f87171", "<b>Act now:</b> sell, process or discount this batch soon; shelf life is nearly exhausted."))
    if not tips:
        tips.append(("#34d399", "Conditions look good. Keep the current storage setup and monitor regularly."))
    for c, t in tips:
        st.markdown(f'<div class="tip" style="--c:{c}"><div>{t}</div></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 2: what-if
# ----------------------------------------------------------------------------
with tab2:
    st.caption("Everything below uses your current sidebar settings and changes one thing at a time.")
    a, b2 = st.columns(2)

    with a:
        xs = np.linspace(-2, 30, 40)
        sweep = pd.concat([row.assign(storage_temperature_c=t) for t in xs], ignore_index=True)
        _, _, d = predict(sweep)
        fig = go.Figure(go.Scatter(x=xs, y=d, mode="lines", line=dict(color="#34d399", width=3),
                                   fill="tozeroy", fillcolor="rgba(52,211,153,.12)"))
        fig.add_trace(go.Scatter(x=[temp], y=[days], mode="markers", marker=dict(size=14, color="#fff", line=dict(color="#34d399", width=3)),
                                 name="Current"))
        fig.update_layout(title="Shelf life vs temperature", xaxis_title="Temperature (°C)", yaxis_title="Days", showlegend=False)
        st.plotly_chart(style(fig), width="stretch")

    with b2:
        xs = np.linspace(0, 45, 40)
        sweep = pd.concat([row.assign(storage_duration_days=t, days_since_packed=t + (packed - dur)) for t in xs], ignore_index=True)
        sweep["days_since_packed"] = sweep["days_since_packed"].clip(lower=0)
        _, pr, d = predict(sweep)
        idx = [spoilage_index(pr.iloc[i]) for i in range(len(pr))]
        fig = go.Figure(go.Scatter(x=xs, y=idx, mode="lines", line=dict(color="#f87171", width=3),
                                   fill="tozeroy", fillcolor="rgba(248,113,113,.12)"))
        fig.add_trace(go.Scatter(x=[dur], y=[index], mode="markers", marker=dict(size=14, color="#fff", line=dict(color="#f87171", width=3))))
        fig.update_layout(title="Spoilage index over storage time", xaxis_title="Storage duration (days)",
                          yaxis_title="Spoilage index", showlegend=False, yaxis=dict(range=[0, 100]))
        st.plotly_chart(style(fig), width="stretch")

    st.markdown("##### Best packaging × storage combination for this item")
    combos = [row.assign(packaging_type=pk, storage_condition=sc) for pk in PACKS for sc in CONDS]
    big = pd.concat(combos, ignore_index=True)
    _, _, d = predict(big)
    grid = pd.DataFrame({"pack": big["packaging_type"], "cond": big["storage_condition"], "days": d})
    piv = grid.pivot(index="pack", columns="cond", values="days")[CONDS]
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale="Tealgrn",
                               text=np.round(piv.values, 1), texttemplate="%{text} d", textfont=dict(size=15),
                               colorbar=dict(title="Days")))
    st.plotly_chart(style(fig, 300), width="stretch")
    bp = grid.loc[grid["days"].idxmax()]
    st.success(f"Best option: **{bp['pack']}** + **{bp['cond']}** → about **{bp['days']:.1f} days**. "
               "(Note: this changes packaging and storage type only, not the temperature slider.)")

    st.markdown("##### Compare all food categories under these conditions")
    allf = pd.concat([row.assign(food_category=f) for f in FOODS], ignore_index=True)
    lab, _, d = predict(allf)
    comp = pd.DataFrame({"food": [f"{EMOJI[f]} {f}" for f in FOODS], "days": d, "risk": lab}).sort_values("days")
    fig = go.Figure(go.Bar(x=comp["days"], y=comp["food"], orientation="h",
                           marker_color=[COLORS[r] for r in comp["risk"]],
                           text=[f"{x:.1f} d" for x in comp["days"]], textposition="outside"))
    fig.update_layout(xaxis_title="Predicted remaining shelf life (days)")
    st.plotly_chart(style(fig, 400), width="stretch")

# ----------------------------------------------------------------------------
# TAB 3: batch
# ----------------------------------------------------------------------------
with tab3:
    st.markdown("Upload a CSV with the same 11 input columns to predict a whole batch at once.")
    tmpl = pd.DataFrame([current])
    st.download_button("⬇️ Download CSV template", tmpl.to_csv(index=False), "batch_template.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            lab, pr, d = predict(data)
            out = data.copy()
            out["predicted_risk"] = lab
            out["predicted_shelf_life_days"] = d.round(1)
            m1, m2, m3 = st.columns(3)
            m1.metric("Items", len(out))
            m2.metric("High-risk items", int((out["predicted_risk"] == "High").sum()))
            m3.metric("Avg shelf life", f"{out['predicted_shelf_life_days'].mean():.1f} d")
            cnt = out["predicted_risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
            fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.62, marker_colors=[COLORS[c] for c in cnt.index]))
            st.plotly_chart(style(fig, 300), width="stretch")
            st.dataframe(out, width="stretch", height=320)
            st.download_button("⬇️ Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")

# ----------------------------------------------------------------------------
# TAB 4: about
# ----------------------------------------------------------------------------
with tab4:
    st.markdown(
        """
**How it works**

- **Model 1** classifies spoilage risk (Low / Medium / High) and reports class probabilities.
- **Model 2** regresses the remaining shelf life in days.
- Inputs: food category, packaging, storage condition, preservative, temperature, humidity, storage duration,
  days since packed, transport delay, initial freshness and handling quality.
- The *spoilage index* is a 0-100 score: P(High) + ½·P(Medium).

**Important:** the models were trained on a **synthetic dataset** with simulated relationships.
Use this for learning and prototyping, never for real food-safety decisions.
        """
    )
