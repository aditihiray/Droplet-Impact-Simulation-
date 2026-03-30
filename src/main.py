# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import matplotlib
#from matplotlib.patches import Circle
#oimport os
matplotlib.use('Qt5Agg')
# %%
# USER / MODEL ASSUMPTIONS
D0 = 0.002         # fixed initial droplet diameter (m)
R0 = D0 / 2.0       # initial droplet radius (m)
g = 9.81            # gravity (m/s^2)

# Release height is measured from the ground to the bottom of the droplet.

# %%
# FLUID DATABASE
excel_file = "fluids.xlsx"

try:
    df = pd.read_excel(excel_file)
    df["fluid"] = df["fluid"].astype(str).str.lower().str.strip()
except Exception as e:
    raise FileNotFoundError(
        f"Could not load '{excel_file}'. Make sure it is in the same folder as this script."
    ) from e

available_fluids = df["fluid"].tolist()

# %%
# Helping FUNCTIONS
def get_fluid_properties(fluid_name: str):
    fluid_name = fluid_name.lower().strip()
    if fluid_name not in available_fluids:
        return None
    row = df[df["fluid"] == fluid_name].iloc[0]
    return float(row["rho"]), float(row["mu"]), float(row["sigma"])

def early_time_wet_radius(tau, R0, Vimp):
    """
    Reduced-order physical wet-radius law based on the short-time analytical impact solution.
    tau = time since first contact (s)
    """
    tau = np.maximum(tau, 0.0)
    return np.sqrt(3.0 * R0 * Vimp * tau)

def build_spreading_history(t, t_impact, D0, R0, Vimp, Dmax_pred,
                            recoil_fraction=0.03, tau_recoil=0.0015):

    D_hist = np.zeros_like(t)
    H_hist = np.zeros_like(t)

    tau_spread = 0.0015

    for i, ti in enumerate(t):

        if ti <= t_impact:
            D = D0
            h = H - 0.5 * g * ti**2

        else:
            tau = ti - t_impact

            # spreading
            D = D0 + (Dmax_pred - D0) * (1 - np.exp(-tau / tau_spread))

            # recoil
            if tau > tau_spread:
                D -= recoil_fraction * (Dmax_pred - D0) * (
                    1 - np.exp(-(tau - tau_spread) / tau_recoil)
                )

            D = np.clip(D, D0, Dmax_pred)
            h = 0

        D_hist[i] = D
        H_hist[i] = max(h, 0)
    return D_hist, H_hist

# %%
# USER INPUTS
while True:
    fluid = input("Enter fluid name: ").lower().strip()
    props = get_fluid_properties(fluid)
    if props is None:
        print("Fluid not found. Available fluids:", available_fluids)
        continue
    rho, mu, sigma = props
    break

while True:
    try:
        H_mm = float(input("Enter droplet release height (mm): "))
        if H_mm <= 0:
            print("Height must be positive.")
            continue
        H = H_mm / 1000.0
        break
    except ValueError:
        print("Please enter a numeric value.")

# %%
# BASIC PHYSICS
Vimp = np.sqrt(2.0 * g * H)  # impact speed from free fall
We = rho * Vimp**2 * D0 / sigma
Re = rho * Vimp * D0 / mu

# Classical reduced-order maximum spread estimate
Dmax_pred = D0 * (We ** 0.25)

# Time to first impact
t_impact = np.sqrt(2.0 * H / g)

# Simulation window long enough to show impact and post-impact evolution
#t_end = t_impact + 0.015
t_end = t_impact + 0.01
dt = 5e-5
t = np.arange(0.0, t_end + dt, dt)

# Build histories
D_hist, bottom_h_hist = build_spreading_history(
    t=t,
    t_impact=t_impact,
    D0=D0,
    R0=R0,
    Vimp=Vimp,
    Dmax_pred=Dmax_pred,
    recoil_fraction=0.03,
    tau_recoil=0.0015
)

simulated_Dmax = float(np.max(D_hist))
contact_velocity = float(Vimp)

# %%
# RESULTS
print(f"\nFluid: {fluid}")
print(f"Density: {rho:.3f} kg/m^3")
print(f"Viscosity: {mu:.6f} Pa·s")
print(f"Surface tension: {sigma:.6f} N/m")
print(f"Release height: {H_mm:.1f} mm")
print(f"Initial droplet diameter: {D0*1000:.2f} mm")
print(f"Impact velocity: {contact_velocity:.3f} m/s")
print(f"Weber number: {We:.2f}")
print(f"Reynolds number: {Re:.2f}")
print(f"Predicted Dmax: {Dmax_pred*1000:.5f} mm")
print(f"Simulated Dmax: {simulated_Dmax*1000:.5f} mm")

# %%
# PLOT 1: DIAMETER VS TIME
plt.figure(figsize=(8, 4.8))
plt.plot(t, D_hist, label="Model diameter", color = 'orange')
plt.axvline(t_impact, linestyle="--", label="First impact", color = '#ba30ac' )
plt.axhline(Dmax_pred, linestyle="--", label="Predicted Dmax", color = '#ba30ac')
plt.xlabel("Time (s)")
plt.ylabel("Droplet diameter (m)")
plt.title(f"Diameter vs Time - {fluid}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("Diameter_vs_Time_plot.png")

# %%
# PLOT 2: PARAMETRIC STUDY: WEBER vs Dmax
start_H = H_mm

heights_mm = [start_H + i * 5 for i in range(10)]

We_list = []
Dmax_list = []

for H_mm_i in heights_mm:

    H_i = H_mm_i / 1000
    V = np.sqrt(2 * g * H_i) # impact velocity
    We = rho * V**2 * D0 / sigma  # Weber number
    Dmax = D0 * (We ** 0.25) # predicted max spreading

    We_list.append(We)
    Dmax_list.append(Dmax)

# PLOT: Weber number vs Dmax
plt.figure()

plt.plot(We_list, Dmax_list, marker='o' , color ='#ba306c')

plt.xlabel("Weber number (-)")
plt.ylabel("Maximum spreading diameter (m)")
plt.title(f"Weber vs Dmax - {fluid}")

plt.grid()

plt.show()
plt.savefig("Webber_No_vs_Dmax_plot.png")

#%%
# ANIMATION DATA

step_skip = 5

t_anim = t[::step_skip]
D_anim = (D_hist * 1000)[::step_skip]

Dmax_mm = np.max(D_anim)
Rmax_mm = Dmax_mm /2

# height (center-based, mm)
z_anim = []

for ti in t_anim:
    if ti <= t_impact:
        z_center = (H - 0.5 * g * ti**2) + R0   # FIXED (center!)
        z_val = max(z_center * 1000, 0)
    else:
        z_val = 0

    z_anim.append(z_val)

z_anim = np.array(z_anim)

# DYNAMIC AXIS SCALING
max_D = np.max(D_anim)
max_z = np.max(z_anim)

x_limit = 0.8 * max_D
y_limit = 1.2 * max_z

# balance aspect ratio
if y_limit > 2 * x_limit:
    x_limit = y_limit / 2

# margin
margin = 1.1
x_limit *= margin
y_limit *= margin

# --
# ANIMATION FUNCTION

fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()

    time = t_anim[frame]
    D = D_anim[frame]
    R = D / 2
    z = z_anim[frame]

    if time <= t_impact:  
        #falling droplet (circle)
       
        theta = np.linspace(0, 2*np.pi, 100)
        x = R * np.cos(theta)
        y = R * np.sin(theta) + z

        ax.plot(x, y, linewidth=2)
        ax.fill(x, y, alpha=0.4, color='#208ad6')

    else:
        # spreading droplet
        thickness = max(0.2, (R0**3 / (R**2)) * 1000)

        x = np.linspace(-R, R, 200)
        y = thickness * np.exp(-(x / R)**2)

        ax.plot(x, y, linewidth=2)
        ax.fill_between(x, 0, y, alpha=0.4, color='#1be3d6')


    ax.axhline(0, color='black')

    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(0, y_limit)

    ax.set_aspect('equal')

    ax.set_title(f"{fluid.capitalize()} Impact")

    ax.text(
        0.95, 0.95,
        f"t = {time*1000:.2f} ms",
        transform=ax.transAxes,
        ha='right',
        va='top'
    )
    
    if t_impact < time < t_impact + 0.002:
         alpha = (time - t_impact) / 0.002
         
    else:
         alpha = 1 if time > t_impact else 0

    if time > t_impact:
        ax.text(
            0.95, 0.85, f"Dₘₐₓ = {Dmax_mm:.2f} mm\nRₘₐₓ = {Rmax_mm:.2f} mm", transform=ax.transAxes, ha='right',va='top',alpha=alpha, color='darkred'
        )
        
    ax.grid()

    # limits
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(0, y_limit)

    # labels
    ax.set_xlabel("Radial direction (mm)")
    ax.set_ylabel("Height (mm)")

    # title
    ax.set_title(f"{fluid.capitalize()} Impact")

    # time display
    ax.text(
        0.95, 0.95,
        f"t = {time*1000:.2f} ms",
        transform=ax.transAxes,
        ha='right',
        va='top'
    )

    ax.set_aspect('equal')
    ax.grid()
    
# --
# RUN ANIMATION

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(t_anim),
    interval=40
)
writer = PillowWriter(fps=20)
ani.save("Droplet.gif", writer=writer, dpi=120)

plt.show()

