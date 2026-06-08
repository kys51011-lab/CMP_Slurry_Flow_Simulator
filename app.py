import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="CMP Slurry Flow Simulator",
    layout="wide"
)

# =========================================================
# Reynolds lubrication solver
# =========================================================

def solve_reynolds(
    U=0.5,
    mu0=0.01,
    h0_um=20.0,
    tilt_x_um=8.0,
    tilt_y_um=0.0,
    Lx_mm=50.0,
    Ly_mm=50.0,
    N=55,
    n_power=1.0,
    iterations=1200,
    relaxation=0.75
):
    """
    Simplified 2D Reynolds lubrication equation:

    d/dx(h^3 dp/dx) + d/dy(h^3 dp/dy) = 6 mu U dh/dx

    Boundary condition:
    p = 0 at all outer boundaries.
    """

    # Unit conversion
    Lx = Lx_mm * 1e-3
    Ly = Ly_mm * 1e-3
    h0 = h0_um * 1e-6
    tilt_x = tilt_x_um * 1e-6
    tilt_y = tilt_y_um * 1e-6

    x = np.linspace(-Lx / 2, Lx / 2, N)
    y = np.linspace(-Ly / 2, Ly / 2, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Gap profile
    h = h0 - tilt_x * (X / Lx) + tilt_y * (Y / Ly)
    h = np.maximum(h, 1.0e-6)

    # Shear-thinning effective viscosity
    gamma_dot = np.maximum(U / h, 1e-9)
    gamma_ref = 1.0e4
    mu_local = mu0 * (gamma_dot / gamma_ref) ** (n_power - 1.0)
    mu_eff = float(np.mean(mu_local))

    h3 = h ** 3
    dhdx = np.gradient(h, dx, axis=0)

    rhs = 6.0 * mu_eff * U * dhdx

    # Coefficients for finite-difference equation
    aE = 0.5 * (h3[1:-1, 1:-1] + h3[2:, 1:-1]) / dx**2
    aW = 0.5 * (h3[1:-1, 1:-1] + h3[:-2, 1:-1]) / dx**2
    aN = 0.5 * (h3[1:-1, 1:-1] + h3[1:-1, 2:]) / dy**2
    aS = 0.5 * (h3[1:-1, 1:-1] + h3[1:-1, :-2]) / dy**2
    denom = aE + aW + aN + aS

    p = np.zeros_like(h)

    # Vectorized weighted Jacobi iteration
    for _ in range(iterations):
        p_old = p.copy()

        p_new_inner = (
            aE * p[2:, 1:-1]
            + aW * p[:-2, 1:-1]
            + aN * p[1:-1, 2:]
            + aS * p[1:-1, :-2]
            - rhs[1:-1, 1:-1]
        ) / denom

        p[1:-1, 1:-1] = (
            (1.0 - relaxation) * p[1:-1, 1:-1]
            + relaxation * p_new_inner
        )

        # Ambient pressure boundary condition
        p[0, :] = 0.0
        p[-1, :] = 0.0
        p[:, 0] = 0.0
        p[:, -1] = 0.0

        if np.max(np.abs(p - p_old)) < 1e-5:
            break

    # Only positive hydrodynamic pressure is used for supporting pressure/removal tendency
    p_pos = np.maximum(p, 0.0)

    # Shear stress estimate
    tau = mu_local * U / h

    # Preston equation: RR = kP * p * U
    # Here, kP is not experimentally calibrated, so RR is treated as relative trend.
    kP = 1.0e-14
    removal_rate = kP * p_pos * U

    Re = 1000.0 * U * np.mean(h) / mu_eff
    aspect_ratio = np.mean(h) / Lx

    return {
        "x_mm": x * 1e3,
        "y_mm": y * 1e3,
        "X_mm": X * 1e3,
        "Y_mm": Y * 1e3,
        "h_um": h * 1e6,
        "p": p,
        "p_pos": p_pos,
        "tau": tau,
        "removal_rate": removal_rate,
        "mu_eff": mu_eff,
        "Re": Re,
        "aspect_ratio": aspect_ratio,
        "max_pressure": float(np.max(p_pos)),
        "avg_pressure": float(np.mean(p_pos)),
        "max_tau": float(np.max(tau)),
        "avg_tau": float(np.mean(tau)),
        "avg_rr": float(np.mean(removal_rate)),
        "rr_nonuniformity": float(
            np.std(removal_rate) / (np.mean(removal_rate) + 1e-30) * 100.0
        )
    }


def heatmap_fig(z, x, y, title, colorbar_title):
    fig = go.Figure(
        data=go.Heatmap(
            z=z.T,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar=dict(title=colorbar_title)
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="x position [mm]",
        yaxis_title="y position [mm]",
        height=500
    )
    return fig


def surface_fig(z, x, y, title, ztitle):
    fig = go.Figure(
        data=[
            go.Surface(
                z=z.T,
                x=x,
                y=y,
                colorscale="Viridis",
                colorbar=dict(title=ztitle)
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [mm]",
            yaxis_title="y [mm]",
            zaxis_title=ztitle
        ),
        height=550
    )
    return fig


# =========================================================
# App title
# =========================================================

st.title("CMP Slurry Flow Simulator")

st.markdown(
    """
    This web simulator analyzes slurry flow in the thin gap between a polishing pad
    and a wafer during chemical mechanical polishing. The model uses Reynolds
    lubrication theory to estimate pressure, shear stress, and removal-rate trends.
    """
)

# =========================================================
# Sidebar input
# =========================================================

st.sidebar.header("Input Parameters")

U = st.sidebar.slider("Pad velocity U [m/s]", 0.05, 2.00, 0.30, 0.05)
mu0 = st.sidebar.slider("Base slurry viscosity μ₀ [Pa·s]", 0.001, 0.050, 0.010, 0.001)
h0_um = st.sidebar.slider("Average gap h₀ [μm]", 5.0, 100.0, 20.0, 1.0)

tilt_x_um = st.sidebar.slider("x-direction gap variation Δhx [μm]", -30.0, 30.0, 8.0, 1.0)
tilt_y_um = st.sidebar.slider("y-direction gap variation Δhy [μm]", -30.0, 30.0, 0.0, 1.0)

Lx_mm = st.sidebar.slider("Domain length Lx [mm]", 10.0, 100.0, 50.0, 5.0)
Ly_mm = st.sidebar.slider("Domain length Ly [mm]", 10.0, 100.0, 50.0, 5.0)

n_power = st.sidebar.slider(
    "Power-law index n",
    0.50,
    1.00,
    1.00,
    0.05,
    help="n = 1: Newtonian slurry, n < 1: shear-thinning slurry"
)

N = st.sidebar.select_slider("Grid size", options=[35, 45, 55, 65], value=55)

results = solve_reynolds(
    U=U,
    mu0=mu0,
    h0_um=h0_um,
    tilt_x_um=tilt_x_um,
    tilt_y_um=tilt_y_um,
    Lx_mm=Lx_mm,
    Ly_mm=Ly_mm,
    N=N,
    n_power=n_power
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Main Simulation", "Validation", "Design Exploration", "Model Description"]
)

# =========================================================
# Tab 1
# =========================================================

with tab1:
    st.header("Main Simulation Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max pressure [Pa]", f"{results['max_pressure']:.3e}")
    c2.metric("Average pressure [Pa]", f"{results['avg_pressure']:.3e}")
    c3.metric("Average shear stress [Pa]", f"{results['avg_tau']:.3e}")
    c4.metric("Reynolds number", f"{results['Re']:.3e}")

    st.subheader("Gap profile")
    st.plotly_chart(
        surface_fig(
            results["h_um"],
            results["x_mm"],
            results["y_mm"],
            "Wafer-pad gap profile h(x,y)",
            "Gap [μm]"
        ),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            heatmap_fig(
                results["p_pos"],
                results["x_mm"],
                results["y_mm"],
                "Hydrodynamic pressure distribution",
                "Pressure [Pa]"
            ),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            heatmap_fig(
                results["tau"],
                results["x_mm"],
                results["y_mm"],
                "Shear stress distribution",
                "Shear stress [Pa]"
            ),
            use_container_width=True
        )

    st.plotly_chart(
        heatmap_fig(
            results["removal_rate"],
            results["x_mm"],
            results["y_mm"],
            "Predicted removal-rate distribution based on Preston equation",
            "Removal rate [a.u.]"
        ),
        use_container_width=True
    )

    st.info(
        "Interpretation: when the moving pad drags slurry into a spatially varying gap, "
        "hydrodynamic pressure is generated. Regions with larger pressure and shear stress "
        "are expected to have a higher material-removal tendency."
    )

# =========================================================
# Tab 2
# =========================================================

with tab2:
    st.header("Validation and Limit Checks")

    st.markdown(
        """
        The simulator is checked using basic physical limits expected from
        Reynolds lubrication theory.
        """
    )

    st.subheader("1. Uniform-gap test")

    uniform = solve_reynolds(
        U=U,
        mu0=mu0,
        h0_um=h0_um,
        tilt_x_um=0.0,
        tilt_y_um=0.0,
        Lx_mm=Lx_mm,
        Ly_mm=Ly_mm,
        N=35,
        n_power=n_power,
        iterations=700
    )

    st.metric(
        "Maximum pressure for uniform gap [Pa]",
        f"{uniform['max_pressure']:.3e}"
    )

    st.caption(
        "Expected result: when the gap is uniform, dh/dx is zero, so the hydrodynamic pressure should be nearly zero."
    )

    st.subheader("2. Velocity scaling test")

    U_values = np.array([0.1, 0.3, 0.5, 0.8, 1.2, 1.6])
    pmax_values = []

    for U_test in U_values:
        r = solve_reynolds(
            U=float(U_test),
            mu0=mu0,
            h0_um=h0_um,
            tilt_x_um=tilt_x_um,
            tilt_y_um=tilt_y_um,
            Lx_mm=Lx_mm,
            Ly_mm=Ly_mm,
            N=35,
            n_power=n_power,
            iterations=700
        )
        pmax_values.append(r["max_pressure"])

    fig_v = px.line(
        x=U_values,
        y=pmax_values,
        markers=True,
        labels={
            "x": "Pad velocity U [m/s]",
            "y": "Maximum pressure [Pa]"
        },
        title="Validation: pressure increases with pad velocity"
    )
    st.plotly_chart(fig_v, use_container_width=True)

    st.subheader("3. Gap-size effect")

    h_values = np.array([10, 15, 20, 30, 50, 80])
    p_gap = []

    for h_test in h_values:
        r = solve_reynolds(
            U=U,
            mu0=mu0,
            h0_um=float(h_test),
            tilt_x_um=tilt_x_um,
            tilt_y_um=tilt_y_um,
            Lx_mm=Lx_mm,
            Ly_mm=Ly_mm,
            N=35,
            n_power=n_power,
            iterations=700
        )
        p_gap.append(r["max_pressure"])

    fig_h = px.line(
        x=h_values,
        y=p_gap,
        markers=True,
        labels={
            "x": "Average gap h₀ [μm]",
            "y": "Maximum pressure [Pa]"
        },
        title="Validation: smaller gap produces larger pressure"
    )
    fig_h.update_xaxes(autorange="reversed")
    st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("4. Viscosity scaling test")

    mu_values = np.array([0.002, 0.005, 0.010, 0.020, 0.035, 0.050])
    p_mu = []

    for mu_test in mu_values:
        r = solve_reynolds(
            U=U,
            mu0=float(mu_test),
            h0_um=h0_um,
            tilt_x_um=tilt_x_um,
            tilt_y_um=tilt_y_um,
            Lx_mm=Lx_mm,
            Ly_mm=Ly_mm,
            N=35,
            n_power=n_power,
            iterations=700
        )
        p_mu.append(r["max_pressure"])

    fig_mu = px.line(
        x=mu_values,
        y=p_mu,
        markers=True,
        labels={
            "x": "Slurry viscosity μ [Pa·s]",
            "y": "Maximum pressure [Pa]"
        },
        title="Validation: pressure increases with viscosity"
    )
    st.plotly_chart(fig_mu, use_container_width=True)

# =========================================================
# Tab 3
# =========================================================

with tab3:
    st.header("Design Exploration")

    st.markdown(
        """
        This section translates the simulation output into practical CMP process-design insights.
        """
    )

    rr_non = results["rr_nonuniformity"]
    max_p = results["max_pressure"]
    avg_tau = results["avg_tau"]
    re = results["Re"]
    aspect = results["aspect_ratio"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Removal-rate nonuniformity [%]", f"{rr_non:.2f}")
    c2.metric("Maximum pressure [Pa]", f"{max_p:.3e}")
    c3.metric("Aspect ratio h/L", f"{aspect:.3e}")

    st.subheader("Automatic qualitative assessment")

    if rr_non < 20:
        st.success("Removal-rate uniformity is relatively good under this condition.")
    elif rr_non < 60:
        st.warning("Removal-rate nonuniformity is moderate. Process tuning may be needed.")
    else:
        st.error("Removal-rate nonuniformity is high. Excessive gap tilt or localized pressure may be present.")

    if h0_um < 10:
        st.warning("Very small average gap: pressure and shear stress may become excessive.")
    else:
        st.write("Average gap is within a moderate range for this simplified simulation.")

    if abs(tilt_x_um) + abs(tilt_y_um) > 30:
        st.warning("Large gap variation: strong wedge effect may increase removal-rate nonuniformity.")
    else:
        st.write("Gap variation is not excessively large in this simplified model.")

    if re < 1:
        st.success("Reynolds number is below 1, supporting the low-Re lubrication-flow assumption.")
    else:
        st.warning("Reynolds number is not very small. Inertial effects may not be negligible.")

    st.subheader("Process-design message")

    st.markdown(
        """
        - Increasing pad velocity can raise removal rate, but it also increases pressure and shear stress.
        - Reducing the gap increases hydrodynamic pressure and may increase defect risk.
        - Excessive pad or wafer tilt leads to nonuniform pressure and removal-rate distribution.
        - A moderate gap and controlled tilt are favorable for more uniform CMP performance.
        - In real CMP, asperity contact pressure and slurry chemistry should be combined with this hydrodynamic model.
        """
    )

# =========================================================
# Tab 4
# =========================================================

with tab4:
    st.header("Model Description")

    st.markdown(
        """
        ### Physical system

        The simulator represents slurry flow in a thin gap between a moving polishing pad
        and a stationary wafer surface during chemical mechanical polishing.

        ### Governing equation

        The main model is the Reynolds lubrication equation:

        $$
        \\frac{\\partial}{\\partial x}
        \\left(h^3 \\frac{\\partial p}{\\partial x}\\right)
        +
        \\frac{\\partial}{\\partial y}
        \\left(h^3 \\frac{\\partial p}{\\partial y}\\right)
        =
        6 \\mu U \\frac{\\partial h}{\\partial x}
        $$

        where:

        - $h(x,y)$ is the local wafer-pad gap.
        - $p(x,y)$ is the slurry pressure.
        - $\\mu$ is the effective slurry viscosity.
        - $U$ is the pad velocity.

        ### Gap profile

        $$
        h(x,y) = h_0 - \\Delta h_x \\frac{x}{L_x}
        + \\Delta h_y \\frac{y}{L_y}
        $$

        ### Shear stress estimate

        $$
        \\tau \\approx \\frac{\\mu U}{h}
        $$

        ### Removal-rate estimate

        The material removal tendency is estimated using the Preston-type relation:

        $$
        RR = k_P p U
        $$

        This simulator is intended for qualitative process-design analysis rather than
        experimentally calibrated prediction.
        """
    )
