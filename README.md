\# CMP Slurry Flow Simulator



\## Project Overview



This project presents a simplified web-based simulator for slurry flow in chemical mechanical polishing (CMP). The simulator models the thin fluid film between a moving polishing pad and a stationary wafer surface. The main objective is to estimate the hydrodynamic pressure distribution, shear stress distribution, and relative material-removal tendency using lubrication theory.



CMP is an important semiconductor manufacturing process used for wafer planarization. During CMP, slurry containing abrasive particles and chemical species flows through the narrow gap between the wafer and polishing pad. The local pressure and shear stress in this gap influence the material removal rate and its spatial uniformity.



\## Physical Model



The simulator is based on the Reynolds lubrication equation for thin-gap viscous flow:



```text

d/dx(h^3 dp/dx) + d/dy(h^3 dp/dy) = 6 μ U dh/dx

```



where:



\* `h(x,y)` is the local wafer-pad gap.

\* `p(x,y)` is the slurry pressure.

\* `μ` is the effective slurry viscosity.

\* `U` is the pad velocity.



The gap profile is modeled as:



```text

h(x,y) = h0 - Δhx(x/Lx) + Δhy(y/Ly)

```



The wall shear stress is estimated as:



```text

τ ≈ μU/h

```



The material-removal tendency is estimated using a Preston-type relation:



```text

RR = kP p U

```



The simulator is intended for qualitative process-design analysis rather than experimentally calibrated prediction.



\## Main Features



The web simulator includes four sections:



1\. Main Simulation



&#x20;  \* Gap profile

&#x20;  \* Hydrodynamic pressure distribution

&#x20;  \* Shear stress distribution

&#x20;  \* Relative removal-rate distribution



2\. Validation



&#x20;  \* Uniform-gap limit check

&#x20;  \* Velocity scaling test

&#x20;  \* Gap-size effect

&#x20;  \* Viscosity scaling test



3\. Design Exploration



&#x20;  \* Removal-rate nonuniformity

&#x20;  \* Maximum pressure

&#x20;  \* Aspect ratio

&#x20;  \* Qualitative process-design assessment



4\. Model Description



&#x20;  \* Governing equation

&#x20;  \* Gap profile

&#x20;  \* Shear stress estimate

&#x20;  \* Removal-rate estimate



\## Input Parameters



The user can control:



\* Pad velocity

\* Slurry viscosity

\* Average gap height

\* Gap variation in the x-direction

\* Gap variation in the y-direction

\* Domain size

\* Power-law index for shear-thinning behavior

\* Grid size



\## Validation Concept



The simulator was checked using basic physical trends expected from Reynolds lubrication theory.



\* When the gap is uniform, the pressure remains nearly zero.

\* When pad velocity increases, the generated pressure increases.

\* When the average gap decreases, the pressure increases strongly.

\* When slurry viscosity increases, the pressure increases.



These checks confirm that the simulator follows the expected qualitative behavior of thin-gap viscous flow.



\## How to Run



Install the required packages:



```bash

pip install -r requirements.txt

```



Run the Streamlit app:



```bash

python -m streamlit run app.py

```



Then open the local URL shown in the terminal.



\## Files



```text

CMP\_Slurry\_Flow\_Simulator

├── app.py

├── requirements.txt

├── README.md

└── figures

&#x20;   ├── fig1\_main\_gap\_profile.png

&#x20;   ├── fig2\_pressure\_shear\_removal.png

&#x20;   ├── fig3\_validation\_velocity.png

&#x20;   ├── fig4\_validation\_gap\_viscosity.png

&#x20;   ├── fig5\_design\_exploration.png

&#x20;   └── fig6\_model\_description.png

```



\## Main Design Insight



The simulation shows that the CMP removal tendency is strongly affected by pad velocity, slurry viscosity, gap size, and gap tilt. A smaller gap or higher pad velocity can increase the removal rate, but it may also increase pressure, shear stress, and spatial nonuniformity. Therefore, moderate gap control and limited tilt are important for achieving more uniform CMP performance.



