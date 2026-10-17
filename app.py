import os
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
import streamlit as st

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Wind Turbine Thermal & Reactive Power Derating Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to read files robustly
def safe_read_file(filepath, **kwargs):
    try:
        return pd.read_excel(filepath, engine="openpyxl", **kwargs)
    except Exception:
        try:
            return pd.read_excel(filepath, engine="calamine", **kwargs)
        except Exception:
            try:
                return pd.read_excel(filepath, engine="xlrd", **kwargs)
            except Exception:
                for enc in ["utf-8", "latin1", "cp1252"]:
                    try:
                        return pd.read_csv(filepath, encoding=enc, **kwargs)
                    except Exception:
                        continue
    raise RuntimeError(f"Could not read file at {filepath}")

# ==========================================
# DATA QUALITY & CLEANING FUNCTION
# ==========================================
def validate_and_clean_telemetry(df):
    """
    Applies data quality checks and collects diagnostic stats:
    1. Removes missing timestamps, sorts chronologically, and drops duplicates.
    2. Clips sensor values to realistic physical bounds.
    3. Cross-checks recorded air density against Ideal Gas Law calculation.
    """
    audit = {}
    audit["raw_row_count"] = len(df)
    
    # 1. Timestamp Integrity
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    audit["valid_timestamp_count"] = len(df)
    
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["Timestamp"], keep="first")
    audit["duplicate_timestamps"] = pre_dedup - len(df)
    
    # 2. Physical Range & Bounding Checks
    audit["temp_clipped_high"] = (df["Temperature_C"] > 50.0).sum()
    audit["temp_clipped_low"] = (df["Temperature_C"] < -10.0).sum()
    df["Temperature_C"] = df["Temperature_C"].clip(-10.0, 50.0)
    
    audit["wind_clipped_high"] = (df["Free_Wind_Speed_m_s"] > 45.0).sum()
    audit["wind_clipped_low"] = (df["Free_Wind_Speed_m_s"] < 0.0).sum()
    df["Free_Wind_Speed_m_s"] = df["Free_Wind_Speed_m_s"].clip(0.0, 45.0)
    
    df["Air_Density_kg_m3"] = df["Air_Density_kg_m3"].clip(0.80, 1.40)
    if "Total_Power_kW" in df.columns:
        df["Total_Power_kW"] = df["Total_Power_kW"].clip(0.0, 7500.0)

    # 3. Relational Consistency: Ideal Gas Law Check (R = 287.058 J/(kg·K))
    if "Pressure_hPa" in df.columns:
        p_pa = df["Pressure_hPa"] * 100.0
        t_kelvin = df["Temperature_C"] + 273.15
        rho_calc = p_pa / (287.058 * t_kelvin)
        density_dev = np.abs(df["Air_Density_kg_m3"] - rho_calc) / rho_calc
        df["Density_Sensor_Flag"] = density_dev > 0.05
        audit["density_flag_count"] = df["Density_Sensor_Flag"].sum()
    else:
        df["Density_Sensor_Flag"] = False
        audit["density_flag_count"] = 0

    return df, audit

# ==========================================
# DATA LOADING & EVALUATION PIPELINE
# ==========================================
@st.cache_data
def load_and_process_data(folder_path="/content/drive/MyDrive/Data/"):
    pc_file = os.path.join(folder_path, "kW_Wind-speed_Air density.xlsx")
    derate_file = os.path.join(folder_path, "active_power_derating.xlsx")
    dataset_file = os.path.join(folder_path, "multi-year_data-set_2.xlsx")
    dataset_csv = os.path.join(folder_path, "multi-year_data-set_2.csv")

    # 1. Load 2D Power Curve Matrix
    pc_raw = safe_read_file(pc_file, header=None)
    densities = pc_raw.iloc[1, 1:].values.astype(float)
    wind_speeds = (
        pc_raw.iloc[2:, 0]
        .astype(str)
        .str.replace("*", "", regex=False)
        .values.astype(float)
    )
    power_matrix = pc_raw.iloc[2:, 1:].values.astype(float)

    p_exp_interp = RegularGridInterpolator(
        (wind_speeds, densities),
        power_matrix,
        bounds_error=False,
        fill_value=None
    )

    # 2. Load 1D Thermal Derating Schedule
    df_derate_full = safe_read_file(derate_file)
    derate_data = df_derate_full.iloc[5:].copy()
    temps = np.array([
        22.0 if (isinstance(t, str) and "Up to" in t) else float(t)
        for t in derate_data.iloc[:, 1].values
    ])
    p_cap_q0 = derate_data.iloc[:, 2].values.astype(float)

    if derate_data.shape[1] > 3 and not pd.isna(derate_data.iloc[0, 3]):
        p_cap_qmid = derate_data.iloc[:, 3].values.astype(float)
    else:
        p_cap_qmax_tmp = derate_data.iloc[:, 4].values.astype(float)
        p_cap_qmid = p_cap_q0 - (p_cap_q0 - p_cap_qmax_tmp) * (2235.0 / 3670.0)

    p_cap_qmax = derate_data.iloc[:, 4].values.astype(float)

    # 3. Load Telemetry Dataset
    if os.path.exists(dataset_file):
        df = safe_read_file(dataset_file, sheet_name="DATA", skiprows=3)
    else:
        df = safe_read_file(dataset_csv, skiprows=3)

    cols = [
        "Timestamp", "Total_Power_kW", "Time_h", "Free_Wind_Speed_m_s",
        "Reduced_Wind_Speed_m_s", "Wind_Direction_deg", "Temperature_C",
        "Pressure_hPa", "Air_Density_kg_m3", "WDC", "All_Corrections", "Air_Density_Correction"
    ]
    df = df.iloc[:, :12]
    df.columns = cols
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Run Data Quality Checks
    df, audit = validate_and_clean_telemetry(df)

    # 4. Interpolate Expected Power & Clip Non-Physical Negative Values
    query_pts = df[["Free_Wind_Speed_m_s", "Air_Density_kg_m3"]].values
    raw_p_exp = p_exp_interp(query_pts)
    df["P_exp"] = np.maximum(0.0, raw_p_exp)

    # 5. Interpolate Thermal Caps
    df["P_cap_Q0"] = np.interp(df["Temperature_C"], temps, p_cap_q0, left=6800.0, right=4700.0)
    df["P_cap_Qmid"] = np.interp(df["Temperature_C"], temps, p_cap_qmid, left=6800.0, right=3850.0)
    df["P_cap_Qmax"] = np.interp(df["Temperature_C"], temps, p_cap_qmax, left=6800.0, right=3000.0)

    # 6. Apply Net Power Capping
    df["P_net_Q0"] = np.minimum(df["P_exp"], df["P_cap_Q0"])
    df["P_net_Qmid"] = np.minimum(df["P_exp"], df["P_cap_Qmid"])
    df["P_net_Qmax"] = np.minimum(df["P_exp"], df["P_cap_Qmax"])

    # 7. Derive Active Power Curtailment Losses (kW)
    df["Loss_Q0_kW"] = np.maximum(0.0, df["P_exp"] - df["P_net_Q0"])
    df["Loss_Qmid_kW"] = np.maximum(0.0, df["P_exp"] - df["P_net_Qmid"])
    df["Loss_Qmax_kW"] = np.maximum(0.0, df["P_exp"] - df["P_net_Qmax"])

    return df, temps, p_cap_q0, p_cap_qmid, p_cap_qmax, audit

# Load Processed Telemetry Data & Audit Summary
data_dir = "./"
df, temps, p_cap_q0, p_cap_qmid, p_cap_qmax, audit = load_and_process_data(data_dir)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("Control Panel")
q_mode = st.sidebar.radio(
    "Operating Regime:",
    [
        "Unity Power Factor (Q = 0 kVar)",
        "Mid-Range Reactive Power (Q = 2235 kVar)",
        "Max Reactive Power (Q = 3670 kVar)"
    ]
)
temp_cutoff = st.sidebar.slider("Temperature Filter Threshold (°C):", 15.0, 40.0, 22.0, 1.0)

if "2235" in q_mode:
    active_loss_col, active_pnet_col = "Loss_Qmid_kW", "P_net_Qmid"
elif "3670" in q_mode:
    active_loss_col, active_pnet_col = "Loss_Qmax_kW", "P_net_Qmax"
else:
    active_loss_col, active_pnet_col = "Loss_Q0_kW", "P_net_Q0"

# Data Quality Flag Summary in Sidebar
if audit["density_flag_count"] > 0:
    st.sidebar.warning(f"Data Quality Alert: {audit['density_flag_count']} timestamps (~{audit['density_flag_count']*10/60:.1f} hrs) flagged for density-temperature mismatch (>5%).")

# ==========================================
# DASHBOARD HEADER & KPI METRICS
# ==========================================
st.title("Wind Turbine Thermal & Reactive Power Derating Analytics")
st.caption("Multi-year wind telemetry audit & active power yield curtailment evaluation.")

mwh_loss = df[active_loss_col].sum() * (10 / 60) / 1000
derate_records = (df[active_loss_col] > 0).sum()
derate_hours = derate_records * (10 / 60)
max_temp = df["Temperature_C"].max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Cumulative Yield Loss", f"{mwh_loss:.2f} MWh")
col2.metric("Curtailment Duration", f"{derate_hours:.1f} Hours")
col3.metric("Curtailment Events", f"{derate_records:,}")
col4.metric("Peak Ambient Temp", f"{max_temp:.1f} °C")

# ==========================================
# DATA QUALITY AUDIT EXPANDER
# ==========================================
with st.expander("🛠️ Data Quality & CRISP-DM Audit Summary"):
    q_col1, q_col2, q_col3 = st.columns(3)
    
    q_col1.markdown("#### Timestamp & Row Integrity")
    q_col1.write(f"- **Raw Rows Ingested:** {audit['raw_row_count']:,}")
    q_col1.write(f"- **Valid Timestamps:** {audit['valid_timestamp_count']:,}")
    q_col1.write(f"- **Duplicate Rows Dropped:** {audit['duplicate_timestamps']}")
    
    q_col2.markdown("#### Sensor Physical Bounds")
    q_col2.write(f"- **Temperature Bounds (-10 to 50°C):** {audit['temp_clipped_high'] + audit['temp_clipped_low']} clipped")
    q_col2.write(f"- **Wind Speed Bounds (0 to 45 m/s):** {audit['wind_clipped_high'] + audit['wind_clipped_low']} clipped")
    q_col2.write(f"- **Non-Negative Power Floor:** Enforced on interpolated matrix")

    q_col3.markdown("#### Cross-Sensor Realism")
    q_col3.write(f"- **Ideal Gas Law Mismatches (>5%):** {audit['density_flag_count']:,} timestamps")
    q_col3.write(f"- **Flagged Duration:** {audit['density_flag_count'] * (10 / 60):.1f} hours")
    q_col3.write(f"- **Status:** Flagged in telemetry for audit tracking")

st.markdown("---")

# ==========================================
# VISUALIZATIONS
# ==========================================
st.subheader("1. Inverter/Generator Thermal Capability Envelopes")
fig_cap = go.Figure()
fig_cap.add_trace(go.Scatter(x=temps, y=p_cap_q0, mode="lines+markers", name="Q = 0 kVar (Unity PF)"))
fig_cap.add_trace(go.Scatter(x=temps, y=p_cap_qmid, mode="lines+markers", name="Q = Mid (2235 kVar)", line=dict(dash="dot")))
fig_cap.add_trace(go.Scatter(x=temps, y=p_cap_qmax, mode="lines+markers", name="Q = Max (3670 kVar)", line=dict(dash="dash")))
fig_cap.update_layout(xaxis_title="Ambient Temperature (°C)", yaxis_title="Active Power Cap (kW)", template="plotly_white")
st.plotly_chart(fig_cap, use_container_width=True)

st.subheader("2. Power-Curve Active Power Output with Thermal Color-Mapping")
filtered_df = df[df["Temperature_C"] >= temp_cutoff]
fig_pc = go.Figure()
fig_pc.add_trace(go.Scatter(
    x=filtered_df["Free_Wind_Speed_m_s"],
    y=filtered_df[active_pnet_col],
    mode="markers",
    marker=dict(size=2, opacity=0.55, color=filtered_df["Temperature_C"], colorscale="RdBu_r", showscale=True, colorbar=dict(title="Air temp (°C)")),
    text=filtered_df["Timestamp"],
    hovertemplate="Timestamp: %{text}<br>Wind Speed: %{x:.2f} m/s<br>Active Power: %{y:.1f} kW<br>Air Temp: %{marker.color:.1f} °C<extra></extra>"
))
fig_pc.update_layout(xaxis_title="Wind speed (m/s)", yaxis_title="Calculated Active Power (kW)", template="plotly_white")
st.plotly_chart(fig_pc, use_container_width=True)

st.subheader("3. Thermal Tail Analysis at Rated Wind Speed")
df_rated = df[df["P_exp"] >= 6800.0]
if not df_rated.empty:
    p95, p96, p97, p98, p99 = [df_rated["Temperature_C"].quantile(q) for q in [0.95, 0.96, 0.97, 0.98, 0.99]]
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    p_col1.metric("95th Percentile", f"{p95:.2f} °C")
    p_col2.metric("96th Percentile", f"{p96:.2f} °C")
    p_col3.metric("97th Percentile", f"{p97:.2f} °C")
    p_col4.metric("98th Percentile", f"{p98:.2f} °C")
    p_col5.metric("99th Percentile", f"{p99:.2f} °C")

    fig_temp_tail = go.Figure()
    fig_temp_tail.add_trace(go.Histogram(
        x=df_rated["Temperature_C"], 
        nbinsx=30, 
        name="Rated Operating Hours",
        marker_color="#4C78A8"
    ))
    
    percentiles = [p95, p96, p97, p98, p99]
    names = ["P95", "P96", "P97", "P98", "P99"]
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    y_levels = [0.95, 0.82, 0.69, 0.56, 0.43]

    for val, name, color, y_lvl in zip(percentiles, names, colors, y_levels):
        fig_temp_tail.add_vline(
            x=val, 
            line_dash="dash", 
            line_color=color, 
            line_width=2
        )
        fig_temp_tail.add_annotation(
            x=val,
            y=y_lvl,
            yref="paper",
            text=f"<b>{name}</b> ({val:.1f}°C)",
            showarrow=False,
            font=dict(color=color, size=12),
            xanchor="left",
            bgcolor="rgba(255, 255, 255, 0.8)"
        )

    fig_temp_tail.update_layout(
        xaxis_title="Ambient Temperature (°C) during Rated Power", 
        yaxis_title="Count (10-min)", 
        template="plotly_white"
    )
    st.plotly_chart(fig_temp_tail, use_container_width=True)

st.subheader("4. Instantaneous Active Power Curtailment vs. Temperature")
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=filtered_df["Temperature_C"],
    y=filtered_df[active_loss_col],
    mode="markers",
    marker=dict(size=2.5, opacity=0.6, color=filtered_df[active_loss_col], colorscale="Reds", showscale=True, colorbar=dict(title="Loss (kW)")),
    text=filtered_df["Timestamp"],
    hovertemplate="Timestamp: %{text}<br>Temp: %{x:.1f} °C<br>Loss: %{y:.1f} kW<extra></extra>"
))
fig_scatter.update_layout(xaxis_title="Ambient Temperature (°C)", yaxis_title="Curtailment Loss (kW)", template="plotly_white")
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("5. Monthly Derating & Curtailment Loss Breakdown")
df_valid_ts = df.dropna(subset=["Timestamp"]).copy()
if not df_valid_ts.empty:
    df_valid_ts["YearMonth"] = df_valid_ts["Timestamp"].dt.strftime("%Y-%m")
    df_valid_ts["Year"] = df_valid_ts["Timestamp"].dt.year
    df_valid_ts["Month"] = df_valid_ts["Timestamp"].dt.month
    df_valid_ts["Month_Name"] = df_valid_ts["Timestamp"].dt.strftime("%b")

    monthly_summary = df_valid_ts.groupby(["Year", "Month", "Month_Name", "YearMonth"], as_index=False).agg(
        Yield_Loss_MWh=(active_loss_col, lambda x: (x.sum() * (10 / 60)) / 1000.0),
        Derating_Hours=(active_loss_col, lambda x: (x > 0).sum() * (10 / 60))
    ).sort_values("YearMonth")

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Bar(x=monthly_summary["YearMonth"], y=monthly_summary["Yield_Loss_MWh"], name="Yield Loss (MWh)", yaxis="y"))
    fig_monthly.add_trace(go.Scatter(x=monthly_summary["YearMonth"], y=monthly_summary["Derating_Hours"], name="Derating Duration (Hours)", mode="lines+markers", yaxis="y2"))
    fig_monthly.update_layout(
        xaxis_title="Month (YYYY-MM)",
        yaxis=dict(title="Monthly Yield Loss (MWh)", side="left"),
        yaxis2=dict(title="Monthly Derating Duration (Hours)", overlaying="y", side="right", showgrid=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50),
        template="plotly_white"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.subheader("6. Multi-Year Average Monthly Derating & Curtailment Profile")
    st.caption("Averages monthly yield losses and derating hours across all available years in the dataset.")

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    avg_monthly_summary = monthly_summary.groupby(["Month", "Month_Name"], as_index=False).agg(
        Avg_Yield_Loss_MWh=("Yield_Loss_MWh", "mean"),
        Avg_Derating_Hours=("Derating_Hours", "mean"),
        Years_Evaluated=("Year", "nunique")
    ).sort_values("Month")

    fig_avg_monthly = go.Figure()
    fig_avg_monthly.add_trace(go.Bar(
        x=avg_monthly_summary["Month_Name"],
        y=avg_monthly_summary["Avg_Yield_Loss_MWh"],
        name="Average Yield Loss (MWh)",
        marker_color="crimson",
        yaxis="y"
    ))
    fig_avg_monthly.add_trace(go.Scatter(
        x=avg_monthly_summary["Month_Name"],
        y=avg_monthly_summary["Avg_Derating_Hours"],
        name="Average Derating (Hours)",
        mode="lines+markers",
        line=dict(color="darkorange", width=2.5),
        yaxis="y2"
    ))
    fig_avg_monthly.update_layout(
        xaxis=dict(title="Month of Year", categoryorder="array", categoryarray=month_order),
        yaxis=dict(title="Average Monthly Yield Loss (MWh)", side="left"),
        yaxis2=dict(title="Average Monthly Derating Duration (Hours)", overlaying="y", side="right", showgrid=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50),
        template="plotly_white"
    )
    st.plotly_chart(fig_avg_monthly, use_container_width=True)

    with st.expander("View Multi-Year Average Monthly Data Table"):
        display_avg = avg_monthly_summary.copy()
        display_avg.columns = [
            "Month Number", "Month", "Mean Yield Loss (MWh)", "Mean Derating Duration (hrs)", "Years Included"
        ]
        st.dataframe(display_avg, use_container_width=True)
