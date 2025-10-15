# app.py
# HOPE Rwanda – Interactive Stormwater Solutions Sandbox
# Streamlit app to combine feasible solutions (Rainwater Harvesting, Vetiver, Hügelkultur, Permeable Pavements)
# and visualize their combined effects on runoff, storage, and qualitative road protection.
#
# Dependencies (add to requirements.txt): streamlit, numpy, pandas, matplotlib
#   streamlit==1.38.0
#   numpy==1.26.4
#   pandas==2.2.2
#   matplotlib==3.8.4
#
# Notes:
# - This is an illustrative planner, not a site-calibrated engineering model.
# - It uses an SCS-CN runoff estimate for a design storm and applies simple deltas for each solution.

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="HOPE Rwanda Stormwater Sandbox", layout="wide")
st.title("🌧️ HOPE Rwanda – Stormwater Solutions Sandbox")

with st.expander("About this tool", expanded=False):
    st.markdown(
        """
This interactive sandbox lets you explore **feasible** solutions from the HOPE Rwanda technical concept:
- **Rainwater Harvesting** (rooftop barrels/tanks)
- **Vetiver Grass** (bioengineering hedgerows that boost infiltration & slope stability)
- **Hügelkultur** (wood-core mounds that store water and reduce surface flow)
- **Permeable Pavements** (porous surfacing that infiltrates instead of shedding water)

It estimates a simple water balance with the SCS-Curve Number method and shows combined effects on
**runoff reduction**, **temporary storage**, and a qualitative **Road Protection Score**.
        """
    )

# -------------------------
# Sidebar: Site & Storm Inputs
# -------------------------
st.sidebar.header("1) Site & Storm Inputs")

colA, colB = st.sidebar.columns(2)
P_mm = colA.slider("Design storm depth P (mm)", 20, 200, 80, 5)
catchment_area_m2 = colB.number_input("Road catchment draining to problem spots (m²)", 100.0, 20000.0, 3000.0, 50.0)

colC, colD = st.sidebar.columns(2)
roof_area_m2 = colC.number_input("Available rooftop catchment for harvesting (m²)", 0.0, 5000.0, 400.0, 10.0)
base_CN = colD.slider("Base Curve Number for dirt roads/soils", 70, 95, 85, 1)

slope_note = st.sidebar.selectbox("Slope condition (qualitative)", ["Low", "Moderate", "Steep"], index=1)
slope_factor = {"Low": 0.95, "Moderate": 1.00, "Steep": 1.05}[slope_note]

# -------------------------
# Sidebar: Solutions
# -------------------------
st.sidebar.header("2) Solutions (toggle & configure)")

# Rainwater Harvesting
harv_on = st.sidebar.toggle("Enable Rainwater Harvesting (rooftop barrels/tanks)", value=True)
if harv_on:
    colH1, colH2 = st.sidebar.columns(2)
    storage_per_unit_L = colH1.number_input("Unit size (L)", 200.0, 10000.0, 1000.0, 50.0)
    n_units = colH2.number_input("# of units (barrels/tanks)", 0, 500, 10, 1)
    first_flush_mm = st.sidebar.slider("First-flush diverter (mm skimmed)", 0, 5, 2, 1)
else:
    storage_per_unit_L, n_units, first_flush_mm = 0.0, 0, 0

# Vetiver Grass
vetiver_on = st.sidebar.toggle("Enable Vetiver Grass hedgerows", value=True)
if vetiver_on:
    vetiver_CN_delta = st.sidebar.slider("Effective CN reduction from vetiver (points)", 0, 10, 4, 1)
    vetiver_infiltration_boost = st.sidebar.slider("Additional infiltration share from vetiver (%)", 0, 30, 10, 1) / 100.0
else:
    vetiver_CN_delta, vetiver_infiltration_boost = 0, 0.0

# Hügelkultur
hug_on = st.sidebar.toggle("Enable Hügelkultur beds", value=True)
if hug_on:
    colHu1, colHu2, colHu3 = st.sidebar.columns(3)
    n_beds = colHu1.number_input("# of beds", 0, 200, 20, 1)
    bed_length_m = colHu2.number_input("Bed length (m)", 1.0, 50.0, 6.0, 0.5)
    bed_width_m = colHu3.number_input("Bed width (m)", 0.5, 5.0, 1.2, 0.1)
    bed_core_depth_m = st.sidebar.number_input("Wood/organic core thickness (m)", 0.1, 2.0, 0.6, 0.05)
    core_porosity = st.sidebar.slider("Core porosity (void fraction)", 0.20, 0.80, 0.50, 0.05)
    border_loss_factor = st.sidebar.slider("Edge/border losses (fraction of capacity unusable)", 0.0, 0.5, 0.15, 0.05)
    hug_intercept_share = st.sidebar.slider("Share of road runoff intercepted by hugel beds (%)", 0, 80, 30, 5) / 100.0
else:
    n_beds, bed_length_m, bed_width_m, bed_core_depth_m, core_porosity = 0, 0.0, 0.0, 0.0, 0.0
    border_loss_factor, hug_intercept_share = 0.0, 0.0

# Permeable Pavements
pp_on = st.sidebar.toggle("Enable Permeable Pavements", value=False)
if pp_on:
    pp_CN_delta = st.sidebar.slider("Effective CN reduction from permeable pavement (points)", 0, 20, 8, 1)
    pp_infiltration_share = st.sidebar.slider("Direct infiltration of rainfall on permeable surface (%)", 0, 90, 40, 5) / 100.0
    pp_fraction_of_catch = st.sidebar.slider("Fraction of road catchment converted to permeable", 0.0, 1.0, 0.25, 0.05)
else:
    pp_CN_delta, pp_infiltration_share, pp_fraction_of_catch = 0, 0.0, 0.0

# -------------------------
# Helpers
# -------------------------
def scs_runoff_depth_mm(P, CN):
    """SCS-CN runoff depth (mm)."""
    S = 25400.0 / CN - 254.0  # mm
    Ia = 0.2 * S
    if P <= Ia:
        return 0.0
    return ((P - Ia) ** 2) / (P + 0.8 * S)

def liters(m3):
    return m3 * 1000.0

def m3_from_mm_over_area(mm, area_m2):
    return (mm / 1000.0) * area_m2

# -------------------------
# 1) Baseline runoff (no solutions)
# -------------------------
CN_effective = base_CN
if vetiver_on:
    CN_effective = max(30, CN_effective - vetiver_CN_delta)
if pp_on:
    # Weighted CN: part of catchment converted to permeable w/ CN reduction
    CN_effective = (1 - pp_fraction_of_catch) * CN_effective + pp_fraction_of_catch * max(30, CN_effective - pp_CN_delta)

CN_effective = CN_effective * slope_factor

Q_mm = scs_runoff_depth_mm(P_mm, CN_effective)
baseline_runoff_m3 = m3_from_mm_over_area(Q_mm, catchment_area_m2)

# -------------------------
# 2) Rainwater Harvesting (rooftops)
# -------------------------
harvest_P_effective_mm = max(0.0, P_mm - first_flush_mm) if harv_on else 0.0
roof_yield_m3 = m3_from_mm_over_area(harvest_P_effective_mm, roof_area_m2)  # all rooftop runoff assumed captured before losses
total_rooftop_storage_m3 = liters(storage_per_unit_L * n_units) / 1000.0
captured_roof_m3 = min(roof_yield_m3, total_rooftop_storage_m3) if harv_on else 0.0
roof_overflow_m3 = max(0.0, roof_yield_m3 - captured_roof_m3)

# -------------------------
# 3) Hügelkultur storage & interception
# -------------------------
hugel_core_vol_m3 = n_beds * bed_length_m * bed_width_m * bed_core_depth_m
hugel_storage_m3 = hugel_core_vol_m3 * core_porosity * (1.0 - border_loss_factor)
# Fraction of road runoff intercepted by the distributed beds (simple share)
hugel_intercepted_m3 = min(baseline_runoff_m3 * hug_intercept_share, hugel_storage_m3) if hug_on else 0.0
hugel_overflow_m3 = max(0.0, baseline_runoff_m3 * hug_intercept_share - hugel_intercepted_m3)

# -------------------------
# 4) Permeable pavements additional infiltration (on permeable fraction)
# -------------------------
# Roughly, share of incident rainfall on permeable fraction infiltrates immediately.
pp_incident_rain_m3 = m3_from_mm_over_area(P_mm, catchment_area_m2 * pp_fraction_of_catch) if pp_on else 0.0
pp_direct_infiltration_m3 = pp_incident_rain_m3 * pp_infiltration_share if pp_on else 0.0

# -------------------------
# 5) Vetiver additional infiltration
# -------------------------
# Apply small additional infiltration share to remaining road runoff after hugel interception
remaining_runoff_after_hugel = max(0.0, baseline_runoff_m3 - hugel_intercepted_m3)
vetiver_extra_infiltration_m3 = remaining_runoff_after_hugel * vetiver_infiltration_boost if vetiver_on else 0.0

# -------------------------
# 6) Combine flows
# -------------------------
# Road system water balance:
#   Baseline road runoff
#   - hugel_intercepted
#   - vetiver_extra_infiltration
#   - pp_direct_infiltration (counted separately, but it ultimately reduces runoff generation)
#   + hugel_overflow (still runoff)
#   + roof_overflow (assume overflow conveyed to ground/swales)
#
# For visualization, treat pp_direct_infiltration as reducing effective runoff generation on the permeable area.
effective_runoff_m3 = (
    max(0.0, baseline_runoff_m3 - hugel_intercepted_m3 - vetiver_extra_infiltration_m3)
    + hugel_overflow_m3
)
# Subtract permeable direct infiltration from effective runoff (since it never becomes surface flow)
effective_runoff_m3 = max(0.0, effective_runoff_m3 - pp_direct_infiltration_m3)

# Add rooftop overflow to ground system
effective_runoff_m3 += roof_overflow_m3

# -------------------------
# Road Protection Score (0-100)
# -------------------------
# Heuristic combining:
#  - Reduction in runoff vs baseline
#  - Presence of vetiver (stability)
#  - Permeable fraction (less flow concentration)
#  - Hügel storage utilized
reduction_ratio = 1.0 - (effective_runoff_m3 / (baseline_runoff_m3 + 1e-9))
score = 40 * max(0.0, reduction_ratio) \
        + 20 * (1.0 if vetiver_on else 0.0) \
        + 20 * pp_fraction_of_catch \
        + 20 * min(1.0, (hug_on * (hugel_intercepted_m3 / (hugel_storage_m3 + 1e-9))))

score = float(np.clip(score, 0, 100))

# -------------------------
# Outputs
# -------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline runoff (m³)", f"{baseline_runoff_m3:,.1f}")
col2.metric("Effective runoff after solutions (m³)", f"{effective_runoff_m3:,.1f}")
col3.metric("Water captured in tanks (m³)", f"{captured_roof_m3:,.1f}")
col4.metric("Road Protection Score (0–100)", f"{score:.0f}")

st.divider()

# Breakdown table
data = [
    ["Design storm depth (mm)", P_mm],
    ["Catchment area (m²)", catchment_area_m2],
    ["Effective CN (after solutions)", round(CN_effective, 1)],
    ["Baseline runoff (m³)", round(baseline_runoff_m3, 2)],
    ["Rooftop yield (m³)", round(roof_yield_m3, 2)],
    ["Captured in tanks (m³)", round(captured_roof_m3, 2)],
    ["Rooftop overflow to ground (m³)", round(roof_overflow_m3, 2)],
    ["Hügel storage capacity (m³)", round(hugel_storage_m3, 2)],
    ["Hügel intercepted (m³)", round(hugel_intercepted_m3, 2)],
    ["Hügel overflow (m³)", round(hugel_overflow_m3, 2)],
    ["Permeable direct infiltration (m³)", round(pp_direct_infiltration_m3, 2)],
    ["Vetiver extra infiltration (m³)", round(vetiver_extra_infiltration_m3, 2)],
    ["Effective runoff after solutions (m³)", round(effective_runoff_m3, 2)],
    ["Road Protection Score", round(score, 0)],
]
df = pd.DataFrame(data, columns=["Metric", "Value"])
st.subheader("Water Balance & Effects Summary")
st.dataframe(df, use_container_width=True, hide_index=True)

# Simple stacked bar chart: where did the water go?
# Parts: Tanks, Hügel intercepted, Permeable direct infiltration, Vetiver extra infiltration, Remaining runoff
bars = {
    "Tanks (rooftop)": captured_roof_m3,
    "Hügel intercepted": hugel_intercepted_m3,
    "Permeable infiltration": pp_direct_infiltration_m3,
    "Vetiver added infiltration": vetiver_extra_infiltration_m3,
    "Remaining runoff": effective_runoff_m3,
}
st.subheader("Distribution of Water (m³)")
fig, ax = plt.subplots(figsize=(8, 3))
left = 0.0
total = sum(bars.values()) + 1e-9
for label, val in bars.items():
    ax.barh([0], [val], left=left)
    ax.text(left + val/2, 0, f"{label}\n{val:,.1f} m³", ha="center", va="center", fontsize=9)
    left += val
ax.set_yticks([])
ax.set_xlabel("Volume (m³)")
ax.set_xlim(0, left * 1.05)
ax.set_title("Water Distribution Across Solutions")
st.pyplot(fig)

# Narrative
st.divider()
st.markdown("### What to try")
st.markdown(
    """
- Increase **# of tanks** or **unit size** to see how rooftop capture reduces ground overflow.
- Add **Hügelkultur** beds (more beds / longer / deeper / higher porosity) to intercept more road runoff.
- Turn on **Permeable Pavements** and increase the **permeable fraction** to boost direct infiltration.
- Increase **Vetiver CN reduction** and **extra infiltration** to represent denser, well-maintained hedgerows.
- Try a higher **storm depth (mm)**—see how combined solutions help keep the Road Protection Score high.
"""
)

st.caption("Illustrative tool based on feasible measures described in HOPE Rwanda technical materials.")
