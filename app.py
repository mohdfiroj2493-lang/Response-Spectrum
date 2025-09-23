# app.py
import os
import importlib.util
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from scipy import integrate

st.set_page_config(page_title="REQPY Spectral Matching", layout="wide")
st.title("REQPY Spectral Matching")

# -----------------------------------------------------------------------------
# Robust import of REQPY_Module (works on Streamlit Cloud, local, or /mnt/data)
# -----------------------------------------------------------------------------
def import_reqpy_module():
    """
    Try to import REQPY_Module by:
      1) standard import,
      2) loading from common absolute/relative paths.

    If found, returns (REQPYrotdnn, REQPY_single). Otherwise raises ImportError.
    """
    # 1) Standard import
    try:
        from REQPY_Module import REQPYrotdnn, REQPY_single
        return REQPYrotdnn, REQPY_single
    except Exception:
        pass

    # 2) Try file-based import from common paths
    candidate_paths = [
        os.environ.get("APP_REQPY_PATH", "").strip(), # optional env var
        "REQPY_Module.py",
        "./REQPY_Module.py",
        "/app/REQPY_Module.py",         # some Streamlit Cloud layouts
        "/mount/src/response-spectrum/REQPY_Module.py",
        "/home/appuser/REQPY_Module.py",
        "/mnt/data/REQPY_Module.py",    # path used in your upload
    ]
    candidate_paths = [p for p in candidate_paths if p]  # drop empties

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                spec = importlib.util.spec_from_file_location("REQPY_Module", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                return mod.REQPYrotdnn, mod.REQPY_single
            except Exception as e:
                # keep trying others
                continue

    # If we get here, we failed all attempts
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
    """
Upload a seed motion (one or two components) and a target spectrum, set parameters,
then run spectral matching. The app plots spectra and time histories and lets
you download matched series and figures.
"""
)

# ----------------------------
# Sidebar — Inputs
# ----------------------------
st.sidebar.header("Inputs")

seed_file1 = st.sidebar.file_uploader("Seed Record 1", type=["txt", "AT2"], key="seed1")
seed_file2 = st.sidebar.file_uploader("Seed Record 2 (optional for RotDnn)", type=["txt", "AT2"], key="seed2")
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

# ----------------------------
# Helpers
# ----------------------------
def load_seed(file, default_dt):
    """
    Accepts:
      - 2-col (t, acc[g]) or multi-col (col0=time, col1=acc)
      - 1-col acc[g] only (uses default_dt)
    Returns: s (acc[g]), dt (s), fs (Hz)
    """
    arr = np.loadtxt(file)
    if arr.ndim == 1:  # single column: acceleration only
        s = arr.astype(float)
        dt = float(default_dt)
    else:
        t = arr[:, 0].astype(float)
        s = arr[:, 1].astype(float)
        dt = float(np.median(np.diff(t)))  # robust to tiny irregularities
    fs = 1.0 / dt
    return s, dt, fs

def load_target(file):
    """
    Target spectrum: 2 columns (T[s], PSA[g])
    """
    tso = np.loadtxt(file)
    if tso.ndim == 1 or tso.shape[1] < 2:
        raise ValueError("Target spectrum must have 2 columns: T[s], PSA[g].")
    To = tso[:, 0].astype(float)
    dso = tso[:, 1].astype(float)
    return To, dso

def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    buf.seek(0)
    return buf

def series_to_txt_bytes(series, dt):
    sio = StringIO()
    header = f"accelerations in g, dt = {dt}"
    np.savetxt(sio, series, header=header)
    return sio.getvalue().encode("utf-8")

# ----------------------------
# Main
# ----------------------------
if run_btn:
    if (seed_file1 is None) or (target_file is None):
        st.error("Please upload at least Seed Record 1 and a Target Spectrum.")
        st.stop()

    try:
        To, dso = load_target(target_file)
    except Exception as e:
        st.error(f"Failed to read target spectrum: {e}")
        st.stop()

    # Load Seed 1
    try:
        s1, dt1, fs1 = load_seed(seed_file1, dt_single)
    except Exception as e:
        st.error(f"Failed to read Seed Record 1: {e}")
        st.stop()

    # Two-component (RotDnn) branch
    if seed_file2 is not None:
        try:
            s2, dt2, fs2 = load_seed(seed_file2, dt1)  # align dt to seed1 if needed
        except Exception as e:
            st.error(f"Failed to read Seed Record 2: {e}")
            st.stop()

        # enforce same length / dt
        n = min(len(s1), len(s2))
        s1 = s1[:n]
        s2 = s2[:n]
        dt = float(dt1)
        fs = 1.0 / dt

        # ===== Run RotDnn matching =====
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

        # ===== Spectra plot =====
        colS1, colS2 = st.columns([1, 1])
        with colS1:
            st.subheader("RotD Spectra")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.semilogx(To, dso, lw=2, label="Target")
            ax.semilogx(T, PSArotnnor, lw=1, label="Original RotD")
            ax.semilogx(T, PSArotnn, lw=1, label="Matched RotD")
            ax.set_xlabel("T [s]"); ax.set_ylabel("PSA [g]")
            ax.legend(frameon=False)
            ax.grid(True, which="both", ls=":")
            st.pyplot(fig)
            st.download_button("Download spectra plot (PNG)", data=fig_to_png_bytes(fig),
                               file_name="spectra_rotd.png")

        # ===== Time histories (acc, vel, disp) for both components =====
        v1 = integrate.cumulative_trapezoid(s1, dx=dt, initial=0)
        d1 = integrate.cumulative_trapezoid(v1, dx=dt, initial=0)
        v2 = integrate.cumulative_trapezoid(s2, dx=dt, initial=0)
        d2 = integrate.cumulative_trapezoid(v2, dx=dt, initial=0)

        # scale originals to similar norms
        sf1 = (np.linalg.norm(cvel1) / max(np.linalg.norm(v1), 1e-12))
        sf2 = (np.linalg.norm(cvel2) / max(np.linalg.norm(v2), 1e-12))

        t = np.linspace(0, (len(s1) - 1) * dt, len(s1))

        def plot_component(title, s_orig, v_orig, d_orig, s_mat, v_mat, d_mat, sf, comp_tag):
            st.subheader(title)

            # acc
            fig_a, ax_a = plt.subplots(figsize=(7, 3))
            ax_a.plot(t, sf * s_orig, lw=0.9, label="Original (scaled)")
            ax_a.plot(t, s_mat, lw=0.9, label="Matched")
            ax_a.set_ylabel("acc [g]"); ax_a.set_xlabel("t [s]")
            ax_a.legend(frameon=False); ax_a.grid(True, ls=":")
            st.pyplot(fig_a)
            st.download_button(f"Download {comp_tag} acc plot (PNG)", data=fig_to_png_bytes(fig_a),
                               file_name=f"timehistory_{comp_tag}_acc.png")

            # vel
            fig_v, ax_v = plt.subplots(figsize=(7, 3))
            ax_v.plot(t, sf * v_orig, lw=0.9, label="Original (scaled)")
            ax_v.plot(t, v_mat, lw=0.9, label="Matched")
            ax_v.set_ylabel("vel/g"); ax_v.set_xlabel("t [s]")
            ax_v.legend(frameon=False); ax_v.grid(True, ls=":")
            st.pyplot(fig_v)
            st.download_button(f"Download {comp_tag} vel plot (PNG)", data=fig_to_png_bytes(fig_v),
                               file_name=f"timehistory_{comp_tag}_vel.png")

            # disp
            fig_d, ax_d = plt.subplots(figsize=(7, 3))
            ax_d.plot(t, sf * d_orig, lw=0.9, label="Original (scaled)")
            ax_d.plot(t, d_mat, lw=0.9, label="Matched")
            ax_d.set_ylabel("disp/g"); ax_d.set_xlabel("t [s]")
            ax_d.legend(frameon=False); ax_d.grid(True, ls=":")
            st.pyplot(fig_d)
            st.download_button(f"Download {comp_tag} disp plot (PNG)", data=fig_to_png_bytes(fig_d),
                               file_name=f"timehistory_{comp_tag}_disp.png")

        with colS2:
            st.subheader("Downloads — Matched Series")
            st.download_button("Download matched comp1 (TXT)", data=series_to_txt_bytes(scc1, dt),
                               file_name="matched_comp1.txt")
            st.download_button("Download matched comp2 (TXT)", data=series_to_txt_bytes(scc2, dt),
                               file_name="matched_comp2.txt")

        st.markdown("---")
        plot_component("Component 1 — Time Histories", s1, v1, d1, scc1, cvel1, cdisp1, sf1, "comp1")
        plot_component("Component 2 — Time Histories", s2, v2, d2, scc2, cvel2, cdisp2, sf2, "comp2")

    # Single-component branch
    else:
        dt = float(dt1)
        fs = float(fs1)

        # ===== Run single-component matching =====
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

        # ===== Spectra plot =====
        colS1, colS2 = st.columns([1, 1])
        with colS1:
            st.subheader("Spectra (PSA)")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.semilogx(To, dso, lw=2, label="Target")
            ax.semilogx(T, PSAs, lw=1, label="Original")
            ax.semilogx(T, PSAccs, lw=1, label="Matched")
            ax.set_xlabel("T [s]"); ax.set_ylabel("PSA [g]")
            ax.legend(frameon=False)
            ax.grid(True, which="both", ls=":")
            st.pyplot(fig)
            st.download_button("Download spectra plot (PNG)", data=fig_to_png_bytes(fig),
                               file_name="spectra_single.png")

        with colS2:
            st.subheader("Download — Matched Series")
            st.download_button("Download matched record (TXT)", data=series_to_txt_bytes(ccs, dt),
                               file_name="matched_single.txt")

        # ===== Time histories =====
        t = np.linspace(0, (len(s1) - 1) * dt, len(s1))
        v_orig = integrate.cumulative_trapezoid(s1, dx=dt, initial=0)
        d_orig = integrate.cumulative_trapezoid(v_orig, dx=dt, initial=0)

        st.markdown("---")
        st.subheader("Time Histories")

        # Acc
        fig_a, ax_a = plt.subplots(figsize=(9, 3))
        ax_a.plot(t, sf * s1, lw=0.9, label="Original (scaled)")
        ax_a.plot(t, ccs, lw=0.9, label="Matched")
        ax_a.set_ylabel("acc [g]"); ax_a.set_xlabel("t [s]")
        ax_a.legend(frameon=False); ax_a.grid(True, ls=":")
        st.pyplot(fig_a)
        st.download_button("Download acc plot (PNG)", data=fig_to_png_bytes(fig_a),
                           file_name="timehistory_acc.png")

        # Vel
        fig_v, ax_v = plt.subplots(figsize=(9, 3))
        ax_v.plot(t, sf * v_orig, lw=0.9, label="Original (scaled)")
        ax_v.plot(t, cvel, lw=0.9, label="Matched")
        ax_v.set_ylabel("vel/g"); ax_v.set_xlabel("t [s]")
        ax_v.legend(frameon=False); ax_v.grid(True, ls=":")
        st.pyplot(fig_v)
        st.download_button("Download vel plot (PNG)", data=fig_to_png_bytes(fig_v),
                           file_name="timehistory_vel.png")

        # Disp
        fig_d, ax_d = plt.subplots(figsize=(9, 3))
        ax_d.plot(t, sf * d_orig, lw=0.9, label="Original (scaled)")
        ax_d.plot(t, cdisp, lw=0.9, label="Matched")
        ax_d.set_ylabel("disp/g"); ax_d.set_xlabel("t [s]")
        ax_d.legend(frameon=False); ax_d.grid(True, ls=":")
        st.pyplot(fig_d)
        st.download_button("Download disp plot (PNG)", data=fig_to_png_bytes(fig_d),
                           file_name="timehistory_disp.png")

# ----------------------------
# Footer note
# ----------------------------
st.caption(
    "Notes: Place REQPY_Module.py next to app.py (recommended), or set environment "
    "variable APP_REQPY_PATH to its absolute path. For 1-column seed files, set dt in the sidebar."
)
