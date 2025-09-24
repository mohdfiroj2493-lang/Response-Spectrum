# app.py
import os
import re
import importlib.util
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import streamlit as st
from scipy import integrate
import plotly.graph_objects as go

st.set_page_config(page_title="REQPY Spectral Matching", layout="wide")
st.title("REQPY Spectral Matching (interactive)")

# -----------------------------------------------------------------------------
# Robust import of REQPY_Module (standard import, env var, common paths, /mnt/data)
# -----------------------------------------------------------------------------
def import_reqpy_module():
    try:
        from REQPY_Module import REQPYrotdnn, REQPY_single  # type: ignore
        return REQPYrotdnn, REQPY_single
    except Exception:
        pass

    candidate_paths = [
        os.environ.get("APP_REQPY_PATH", "").strip(),
        "REQPY_Module.py",
        "./REQPY_Module.py",
        "/app/REQPY_Module.py",
        "/mount/src/response-spectrum/REQPY_Module.py",
        "/home/appuser/REQPY_Module.py",
        "/mnt/data/REQPY_Module.py",
    ]
    candidate_paths = [p for p in candidate_paths if p]

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("REQPY_Module", path)
                mod = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(mod)  # type: ignore
                return mod.REQPYrotdnn, mod.REQPY_single
            except Exception:
                continue

    raise ImportError(
        "Could not import REQPY_Module. Place REQPY_Module.py next to app.py, "
        "or set APP_REQPY_PATH to its absolute path."
    )

try:
    REQPYrotdnn, REQPY_single = import_reqpy_module()
except ImportError as e:
    st.error(str(e))
    st.stop()

st.markdown(
    "Upload a seed motion (one or two components) and a target spectrum, set parameters, "
    "then run spectral matching. You’ll get interactive plots **and** the underlying "
    "values as tables you can download."
)

# -----------------------------------------------------------------------------
# Sidebar — Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Inputs")

# Accept TXT, AT2, CSV for seed components
seed_file1 = st.sidebar.file_uploader("Seed Record 1", type=["txt", "AT2", "csv"], key="seed1")
seed_file2 = st.sidebar.file_uploader("Seed Record 2 (optional for RotDnn)", type=["txt", "AT2", "csv"], key="seed2")

# Accept TXT/CSV for target spectrum
target_file = st.sidebar.file_uploader("Target Spectrum (2 cols: T[s], PSA[g])", type=["txt", "csv"], key="target")

st.sidebar.markdown("**If your seed files are single-column (acc only), set dt here:**")
dt_single = st.sidebar.number_input(
    "dt for single-column seed files [s]", min_value=1e-4, max_value=1.0, value=0.02, step=0.001, format="%.4f"
)

dampratio = st.sidebar.number_input("Damping ratio (ζ)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
T1 = st.sidebar.number_input("Match range start T1 [s]", min_value=0.0, max_value=20.0, value=0.05, step=0.01)
T2 = st.sidebar.number_input("Match range end T2 [s]", min_value=0.0, max_value=20.0, value=6.0, step=0.01)
nit = st.sidebar.number_input("Max iterations", min_value=1, max_value=100, value=15, step=1)
NS = st.sidebar.number_input("Number of wavelet scales (NS)", min_value=50, max_value=800, value=100, step=10)
percentile = st.sidebar.selectbox("RotD percentile (for 2-component)", options=[100, 90, 75, 50], index=0)

run_btn = st.sidebar.button("Run Matching")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def series_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def load_target(file):
    """
    Reads a target spectrum as two numeric columns:
      col1 = Period T [s], col2 = PSA [g]
    Supports CSV (commas) or TXT (whitespace). Skips non-numeric header lines (e.g., 'T,PSA').
    """
    raw = file.read()
    text = raw.decode("utf-8", errors="ignore").strip()

    # Keep only lines that start with a number; drop headers like "T,PSA"
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        token = s.split(",")[0].split()[0]
        try:
            float(token)
            lines.append(s)
        except Exception:
            continue

    cleaned = "\n".join(lines)

    # Try comma-delimited first; fallback to whitespace
    try:
        arr = np.loadtxt(StringIO(cleaned), delimiter=",")
    except Exception:
        arr = np.loadtxt(StringIO(cleaned))

    if arr.ndim == 1 and arr.size >= 2:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Target spectrum must have two numeric columns: T[s], PSA[g].")

    To = arr[:, 0].astype(float)
    dso = arr[:, 1].astype(float)

    order = np.argsort(To)
    return To[order], dso[order]

def _read_numeric_table(text):
    """Read a numeric table from text that may be CSV or whitespace separated."""
    try:
        df = pd.read_csv(StringIO(text))
        return df.values
    except Exception:
        pass
    try:
        arr = np.loadtxt(StringIO(text), delimiter=",")
        return arr
    except Exception:
        pass
    return np.loadtxt(StringIO(text))

def load_seed(file, default_dt):
    """
    Supports seed formats:
      - PEER/NGA .AT2 files with headers (PEER, NPTS=..., DT=...) and wrapped columns
      - 2-col TXT/CSV: [time(s), acc(g)]
      - 1-col TXT/CSV: [acc(g)]  -> uses default_dt from sidebar
    Returns: s (acc in g), dt (s), fs (Hz)
    """
    raw = file.read()
    text = raw.decode("latin-1", errors="ignore")
    head = text[:500].upper()

    # AT2 / PEER header path
    if (file.name and file.name.upper().endswith(".AT2")) or ("PEER" in head) or ("NPTS" in head and "DT" in head):
        m = re.search(r"DT\s*=\s*([0-9.+\-Ee]+)", text)
        if not m:
            raise ValueError("AT2 header found but DT= missing.")
        dt = float(m.group(1))

        # Start numeric data after the line containing NPTS
        lines = text.splitlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if "NPTS" in line.upper():
                start_idx = i + 1
                break
        data_str = "\n".join(lines[start_idx:])

        # Robust read for wrapped columns (whitespace-separated numbers)
        s = np.fromstring(data_str, sep=" ", dtype=float).astype(float)
        fs = 1.0 / dt
        return s, dt, fs

    # CSV/TXT numeric path (1-col or 2-col)
    arr = _read_numeric_table(text)
    if np.ndim(arr) == 1:
        arr = np.asarray(arr).reshape(-1, 1)

    if arr.shape[1] == 1:  # 1-col acceleration
        s = arr[:, 0].astype(float)
        dt = float(default_dt)
    else:  # assume first two columns are [time, acc]
        t = arr[:, 0].astype(float)
        s = arr[:, 1].astype(float)
        dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    return s, dt, fs

def plot_spectrum_plotly(To, target, T, curves: dict, title="Response Spectrum"):
    """curves: dict name->y"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=To, y=target, mode="lines", name="Target"))
    for name, y in curves.items():
        fig.add_trace(go.Scatter(x=T, y=y, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Period T [s]",
        yaxis_title="PSA [g]",
        xaxis_type="log",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

def plot_timeseries_plotly(t, series: dict, title, yaxis_title):
    """series: dict name->y"""
    fig = go.Figure()
    for name, y in series.items():
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name))
    fig.update_layout(
        title=title,
        xaxis_title="Time [s]",
        yaxis_title=yaxis_title,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig

# -----------------------------------------------------------------------------
# Optional diagnostics
# -----------------------------------------------------------------------------
with st.expander("Environment diagnostics (optional)"):
    st.code(
        "CWD: " + os.getcwd() + "\n"
        "APP_REQPY_PATH: " + str(os.environ.get("APP_REQPY_PATH", "")) + "\n"
        "REQPY present next to app.py: " + str(os.path.exists('REQPY_Module.py')) + "\n"
        "REQPY present in /mnt/data: " + str(os.path.exists('/mnt/data/REQPY_Module.py')),
        language="bash",
    )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if run_btn:
    if (seed_file1 is None) or (target_file is None):
        st.error("Please upload at least Seed Record 1 and a Target Spectrum.")
        st.stop()

    # Load target
    try:
        To, dso = load_target(target_file)
    except Exception as e:
        st.error(f"Failed to read target spectrum: {e}")
        st.stop()

    # Load seed 1
    try:
        s1, dt1, fs1 = load_seed(seed_file1, dt_single)
    except Exception as e:
        st.error(f"Failed to read Seed Record 1: {e}")
        st.stop()

    # Two-component branch
    if seed_file2 is not None:
        # Load seed 2
        try:
            s2, dt2, fs2 = load_seed(seed_file2, dt1)
        except Exception as e:
            st.error(f"Failed to read Seed Record 2: {e}")
            st.stop()

        # Enforce same length and dt
        n = min(len(s1), len(s2))
        s1, s2 = s1[:n], s2[:n]
        dt = float(dt1)
        fs = 1.0 / dt

        # Run RotDnn matching
        try:
            (scc1, scc2, cvel1, cvel2, cdisp1, cdisp2,
             PSArotnn, PSArotnnor, T, meanefin, rmsefin) = REQPYrotdnn(
                s1, s2, fs, dso, To, int(percentile),
                T1=T1, T2=T2, zi=dampratio,
                nit=int(nit), NS=int(NS),
                baseline=1, porder=-1,
                plots=0, nameOut="streamlit"
            )
        except Exception as e:
            st.error(f"REQPYrotdnn failed: {e}")
            st.stop()

        st.success(f"Two-Component match complete — RMSE: {rmsefin:.2f}%, Misfit: {meanefin:.2f}%")

        # ---------- Responsive spectrum ----------
        st.subheader("RotD Response Spectrum (interactive)")
        fig_spec = plot_spectrum_plotly(To, dso, T, {
            "Original RotD": PSArotnnor,
            "Matched RotD": PSArotnn
        }, title=f"RotD{int(percentile)} PSA (ζ={dampratio:.2f})")
        st.plotly_chart(fig_spec, use_container_width=True)

        # ---------- Spectrum values table & download ----------
        df_spec = pd.DataFrame({
            "T[s]": T,
            "PSA_RotD_original[g]": PSArotnnor,
            "PSA_RotD_matched[g]": PSArotnn
        })
        df_target = pd.DataFrame({"T[s]": To, "PSA_target[g]": dso})
        st.markdown("**Values — Target and RotD spectra**")
        st.dataframe(df_target.merge(df_spec, how="outer", on="T[s]").sort_values("T[s]"), use_container_width=True)
        st.download_button("Download spectra values (CSV)",
                           data=series_to_csv_bytes(df_target.merge(df_spec, how="outer", on="T[s]").sort_values("T[s]")),
                           file_name="spectra_values.csv")

        # ---------- ORIGINAL + MATCHED time histories ----------
        # Build time axis
        t = np.linspace(0, (len(s1) - 1) * dt, len(s1))
        # Originals
        v1 = integrate.cumulative_trapezoid(s1, dx=dt, initial=0)
        d1 = integrate.cumulative_trapezoid(v1, dx=dt, initial=0)
        v2 = integrate.cumulative_trapezoid(s2, dx=dt, initial=0)
        d2 = integrate.cumulative_trapezoid(v2, dx=dt, initial=0)
        # Scale factors so original overlays visually
        sf1 = (np.linalg.norm(cvel1) / max(np.linalg.norm(v1), 1e-12))
        sf2 = (np.linalg.norm(cvel2) / max(np.linalg.norm(v2), 1e-12))

        # Component 1
        st.markdown("---")
        st.subheader("Component 1 — Time Histories (original vs. matched)")
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) acc [g]": sf1*s1, "Matched acc [g]": scc1},
                                   "Acceleration (Comp 1)", "acc [g]"),
            use_container_width=True
        )
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) vel [g·s]": sf1*v1, "Matched vel [g·s]": cvel1},
                                   "Velocity (Comp 1)", "vel [g·s]"),
            use_container_width=True
        )
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) disp [g·s²]": sf1*d1, "Matched disp [g·s²]": cdisp1},
                                   "Displacement (Comp 1)", "disp [g·s²]"),
            use_container_width=True
        )
        df_c1 = pd.DataFrame({
            "t[s]": t,
            "acc_original_scaled[g]": sf1*s1, "acc_matched[g]": scc1,
            "vel_original_scaled[g·s]": sf1*v1, "vel_matched[g·s]": cvel1,
            "disp_original_scaled[g·s²]": sf1*d1, "disp_matched[g·s²]": cdisp1,
        })
        st.dataframe(df_c1, use_container_width=True, height=280)
        st.download_button("Download Component 1 time history (CSV)", data=series_to_csv_bytes(df_c1),
                           file_name="component1_time_history.csv")

        # Component 2
        st.markdown("---")
        st.subheader("Component 2 — Time Histories (original vs. matched)")
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) acc [g]": sf2*s2, "Matched acc [g]": scc2},
                                   "Acceleration (Comp 2)", "acc [g]"),
            use_container_width=True
        )
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) vel [g·s]": sf2*v2, "Matched vel [g·s]": cvel2},
                                   "Velocity (Comp 2)", "vel [g·s]"),
            use_container_width=True
        )
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) disp [g·s²]": sf2*d2, "Matched disp [g·s²]": cdisp2},
                                   "Displacement (Comp 2)", "disp [g·s²]"),
            use_container_width=True
        )
        df_c2 = pd.DataFrame({
            "t[s]": t,
            "acc_original_scaled[g]": sf2*s2, "acc_matched[g]": scc2,
            "vel_original_scaled[g·s]": sf2*v2, "vel_matched[g·s]": cvel2,
            "disp_original_scaled[g·s²]": sf2*d2, "disp_matched[g·s²]": cdisp2,
        })
        st.dataframe(df_c2, use_container_width=True, height=280)
        st.download_button("Download Component 2 time history (CSV)", data=series_to_csv_bytes(df_c2),
                           file_name="component2_time_history.csv")

        # TXT downloads (compat)
        st.markdown("---")
        st.subheader("Downloads — Matched Series (TXT)")
        st.download_button("Matched comp1 (TXT)",
                           data=pd.Series(scc1).to_csv(index=False, header=False).encode("utf-8"),
                           file_name="matched_comp1.txt")
        st.download_button("Matched comp2 (TXT)",
                           data=pd.Series(scc2).to_csv(index=False, header=False).encode("utf-8"),
                           file_name="matched_comp2.txt")

    # Single-component branch
    else:
        dt = float(dt1)
        fs = float(fs1)

        # Run single-component matching
        try:
            (ccs, rmse, misfit, cvel, cdisp,
             PSAccs, PSAs, T, sf) = REQPY_single(
                s1, fs, dso, To,
                T1=T1, T2=T2, zi=dampratio,
                nit=int(nit), NS=int(NS),
                baseline=1, porder=-1,
                plots=0, nameOut="streamlit"
            )
        except Exception as e:
            st.error(f"REQPY_single failed: {e}")
            st.stop()

        st.success(f"Single-Component match complete — RMSE: {rmse:.2f}%, Misfit: {misfit:.2f}%")

        # ---------- Responsive spectrum ----------
        st.subheader("Response Spectrum (interactive)")
        fig_spec = plot_spectrum_plotly(To, dso, T, {
            "Original": PSAs,
            "Matched": PSAccs
        }, title=f"PSA (ζ={dampratio:.2f})")
        st.plotly_chart(fig_spec, use_container_width=True)

        # ---------- Spectrum values table & download ----------
        df_spec = pd.DataFrame({"T[s]": T, "PSA_original[g]": PSAs, "PSA_matched[g]": PSAccs})
        df_target = pd.DataFrame({"T[s]": To, "PSA_target[g]": dso})
        st.markdown("**Values — Target and PSA (original vs. matched)**")
        df_merged = df_target.merge(df_spec, how="outer", on="T[s]").sort_values("T[s]")
        st.dataframe(df_merged, use_container_width=True)
        st.download_button("Download spectra values (CSV)", data=series_to_csv_bytes(df_merged),
                           file_name="spectra_values.csv")

        # ---------- ORIGINAL + MATCHED time histories ----------
        t = np.linspace(0, (len(s1) - 1) * dt, len(s1))
        v_orig = integrate.cumulative_trapezoid(s1, dx=dt, initial=0)
        d_orig = integrate.cumulative_trapezoid(v_orig, dx=dt, initial=0)

        st.markdown("---")
        st.subheader("Time Histories (original vs. matched)")
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) acc [g]": sf*s1, "Matched acc [g]": ccs},
                                   "Acceleration", "acc [g]"),
            use_container_width=True
        )
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) vel [g·s]": sf*v_orig, "Matched vel [g·s]": cvel},
                                   "Velocity", "vel [g·s]"),
            use_container_width=True
        )
        st.plotly_chart(
            plot_timeseries_plotly(t, {"Original (scaled) disp [g·s²]": sf*d_orig, "Matched disp [g·s²]": cdisp},
                                   "Displacement", "disp [g·s²]"),
            use_container_width=True
        )

        df_th = pd.DataFrame({
            "t[s]": t,
            "acc_original_scaled[g]": sf*s1, "acc_matched[g]": ccs,
            "vel_original_scaled[g·s]": sf*v_orig, "vel_matched[g·s]": cvel,
            "disp_original_scaled[g·s²]": sf*d_orig, "disp_matched[g·s²]": cdisp,
        })
        st.dataframe(df_th, use_container_width=True, height=280)
        st.download_button("Download matched time history (CSV)", data=series_to_csv_bytes(df_th),
                           file_name="matched_time_history.csv")

# Footer
st.caption(
    "Notes: Place REQPY_Module.py next to app.py (recommended) or set APP_REQPY_PATH to its absolute path. "
    "AT2 files are auto-detected (DT read from header). Seed upload accepts AT2/TXT/CSV; "
    "targets accept TXT/CSV with optional 'T,PSA' header. Plots are Plotly-based and responsive."
)
