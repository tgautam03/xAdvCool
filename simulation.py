import warp as wp
import numpy as np
from tqdm import tqdm
import os
import shutil

from src.fluid_engine3d import (BC_SOLID, BC_FLUID, BC_INLET, BC_OUTLET, 
                                compute_equilibrium_kernel, compute_macroscopic_kernel,
                                bgk_collision_kernel, streaming_kernel, compute_stats_kernel)


def main():
    # Delete all saved results safely inside main
    import glob
    for file in glob.glob("results/*.npy"):
        try:
            os.remove(file)
        except OSError:
            pass

    print("Initializing Warp...")
    wp.init()
    
    mask_file = "designed_mask.npy"
    if not os.path.exists(mask_file):
        print(f"Error: {mask_file} not found.")
        print("Please use the geometry.py script to export a mask first.")
        return
        
    print(f"Loading {mask_file}...")
    mask = np.load(mask_file)
    mask_vals = np.unique(mask)
    print(f"Mask values found: {mask_vals}")
    
    nx, ny, nz = mask.shape
    print(f"Simulation Domain: {nx}x{ny}x{nz}")

    # Calculate precise geometric properties for Reynolds number
    buf = 10
    core = mask[buf:-buf, 1:-1, 4:-2] 
    fluid_mask = (core != BC_SOLID)
    solid_mask = (core == BC_SOLID)
    
    V_total = core.size
    V_fluid = np.sum(fluid_mask)
    porosity = V_fluid / V_total if V_total > 0 else 1.0
    
    faces_x = np.sum(fluid_mask[1:,:,:] & solid_mask[:-1,:,:]) + np.sum(solid_mask[1:,:,:] & fluid_mask[:-1,:,:])
    faces_y = np.sum(fluid_mask[:,1:,:] & solid_mask[:,:-1,:]) + np.sum(solid_mask[:,1:,:] & fluid_mask[:,:-1,:])
    faces_z = np.sum(fluid_mask[:,:,1:] & solid_mask[:,:,:-1]) + np.sum(solid_mask[:,:,1:] & fluid_mask[:,:,:-1])
    A_wetted = faces_x + faces_y + faces_z
    
    Dh = 4.0 * V_fluid / A_wetted if A_wetted > 0 else float(ny)
    print(f"Geometry -> Porosity: {porosity:.2f}, Hydraulic Diameter (Dh): {Dh:.2f} LU")

    # Simulation Parameters
    max_steps = 40000
    stats_every = 100 # Check convergence/Update TQDM (Decoupled from disk I/O)
    
    # Convergence Criteria
    tol_ke = 1e-7
    tol_T = 1e-7
    
    # Allocations
    device = "cuda"
    
    # Fluid Properties
    tau_fluid = 0.52 # More viscous fluid
    rho_outlet = 1.0
    
    # --- PUMP CONTROL SETUP ---
    u_inlet_val = 0.03   # Initial guess for velocity
    u_inlet_array = wp.array([u_inlet_val], dtype=float, device=device)
    
    # Initial Conditions (Rho=1, U=0, T=0)
    rho_np = np.ones((nx, ny, nz), dtype=np.float32)
    rho_warp = wp.array(rho_np, dtype=wp.float32, device=device)
    u = wp.zeros((nx, ny, nz), dtype=wp.vec3f, device=device)
    
    
    domain_mask = wp.array(mask, dtype=wp.int32, device=device)
    
    # Fluid Distributions
    f_old = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=device)
    f_new = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=device)
    f_eq = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=device)
    
    # --- THERMAL SETUP ---
    from src.thermal_engine3d import (compute_thermal_equilibrium_kernel, thermal_bgk_collision_kernel, 
                                    thermal_streaming_kernel, compute_thermal_macroscopic_kernel,
                                    compute_thermal_stats_kernel)
    
    # Thermal Properties
    # 1. Water (Pr ≈ 7) 
    tau_th_fluid = 0.5 + ((tau_fluid - 0.5) / 7.0)   
    # 2. Silicon (Thermal Diffusivity ~628x greater than Water)
    tau_th_solid = 0.5 + 628.0 * (tau_th_fluid - 0.5)
    
    # Heat Source
    # Power input Q. Dimensions: Temperature / Time.
    heat_power = 0.1
    
    T = wp.full((nx, ny, nz), value=25.0, dtype=float, device=device)
    g_old = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    g_new = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    g_eq = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    
    # Create Heat Source Mask (z=0, buffer excluded)
    # Buffer in geometry.py was 15. We don't have that var here easily unless we guess or pass it.
    # Looking at geometry.py, buffer_size=15 is default.
    # Let's approximate or just use the mask structure.
    # The mask has BC_INLET/OUTLET at ends.
    # Let's check where the solid pins start.
    # Actually, simpler: Apply to the entire z=0 floor EXCEPT x < 5 and x > nx-5 (small buffer)
    # or match the fluid inlet/outlet buffers.
    # The user said "excluding the buffer zones".
    # Mask logic:
    # Heat Source Definition
    heat_mask_np = np.zeros((nx, ny, nz), dtype=np.float32)
    
    # "Dual-Core Processor" Scenario
    print("Generating Processor Heat Map (Dual Core + Cache)...")
    
    # Use shared utility from geometry.py
    import geometry
    heat_map_2d = geometry.generate_processor_heatmap(nx, ny)
    
    # Apply to base layer
    heat_mask_np[:, :, 0] = heat_map_2d
    
    # Normalize map to [0, 1] range to work with existing intensity scaling logic
    if heat_mask_np.max() > 0:
        heat_mask_np /= heat_mask_np.max()
    
    heat_source_mask = wp.array(heat_mask_np, dtype=float, device=device)
    
    # Thermal Init (T=0 everywhere)
    wp.launch(compute_thermal_equilibrium_kernel, dim=(nx, ny, nz), 
              inputs=[rho_warp, u, T, g_eq, domain_mask], device=device)
    wp.copy(g_old, g_eq)

    
    # Initialization (Equilibrium)
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, f_eq], device=device)
    wp.copy(f_old, f_eq)
    
    print(f"Running Simulation until Equilibrium or {max_steps} steps...")
    
    # Capture Graph for speed
    wp.capture_begin(device=device)
    
    # --- FLUID STEP ---
    wp.launch(bgk_collision_kernel, dim=(nx, ny, nz), inputs=[f_old, f_eq, f_new, domain_mask, tau_fluid], device=device)
    wp.launch(streaming_kernel, dim=(nx, ny, nz), inputs=[f_new, f_old, domain_mask, u_inlet_array, rho_outlet, nx, ny, nz], device=device)
    wp.launch(compute_macroscopic_kernel, dim=(nx, ny, nz), inputs=[f_old, rho_warp, u], device=device)
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, f_eq], device=device)
    
    # --- THERMAL STEP ---
    # 1. Equilibrium (using new u)
    wp.launch(compute_thermal_equilibrium_kernel, dim=(nx, ny, nz), 
              inputs=[rho_warp, u, T, g_eq, domain_mask], device=device)
    
    # 2. Collision (with source)
    wp.launch(thermal_bgk_collision_kernel, dim=(nx, ny, nz), 
              inputs=[g_old, g_eq, g_new, domain_mask, tau_th_fluid, tau_th_solid, heat_source_mask, heat_power], device=device)
              
    # 3. Streaming
    # Inlet T = 25.0 (Ambient liquid entering)
    t_inlet = 25.0
    wp.launch(thermal_streaming_kernel, dim=(nx, ny, nz), 
              inputs=[g_new, g_old, domain_mask, t_inlet, nx, ny, nz], device=device)
              
    # 4. Macroscopic T
    wp.launch(compute_thermal_macroscopic_kernel, dim=(nx, ny, nz), inputs=[g_old, T], device=device)
    
    graph = wp.capture_end(device=device)
    
    # Results Dir
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results", exist_ok=True)
    
    pbar = tqdm(total=max_steps)
    
    prev_ke = 0.0

    # Stats buffers
    # [max_v, sum_v, sum_ke, count, sum_rho, sum_rho_in, sum_rho_out, sum_ux_out, c_in, c_out, c_outlet_nodes]
    stats_warp = wp.zeros(11, dtype=float, device=device)
    stats_thermal_warp = wp.zeros(3, dtype=float, device=device)

    step = 0
    prev_max_T = 0.0
    
    while step < max_steps:
        wp.capture_launch(graph)
        step += 1

        # Check stats regularly
        if step % stats_every == 0:
            
            # 1. Compute on GPU
            stats_warp.zero_()
            stats_thermal_warp.zero_()
            
            wp.launch(compute_stats_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, domain_mask, stats_warp, nx, ny, nz], device=device)
            wp.launch(compute_thermal_stats_kernel, dim=(nx, ny, nz), inputs=[T, stats_thermal_warp], device=device)
            
            # 2. Fetch Scalar Result (Tiny copy)
            stats_cpu = stats_warp.numpy()
            t_stats_cpu = stats_thermal_warp.numpy()
            
            # Fluid Stats
            max_v = float(stats_cpu[0])
            sum_v = float(stats_cpu[1])
            ke    = float(stats_cpu[2])
            count = float(stats_cpu[3])
            sum_rho = float(stats_cpu[4])
            
            avg_v = sum_v / count if count > 0 else 0.0
            avg_rho = sum_rho / count if count > 0 else 0.0
            
            # Hydraulic Stats for Pump Control
            rho_in_avg = stats_cpu[5] / stats_cpu[8] if stats_cpu[8] > 0 else 1.0
            rho_out_avg = stats_cpu[6] / stats_cpu[9] if stats_cpu[9] > 0 else 1.0
            
            # Use INLET Flow for the control loop feedback. 
            # Using Outlet Flow (Q_measured) causes the pump to turn to max during startup
            # because it thinks it's doing 0 work until the fluid reaches the exit.
            inlet_area = stats_cpu[8]
            Q_inlet_est = u_inlet_val * inlet_area
            
            dP_measured = (rho_in_avg - rho_out_avg) / 3.0
            power_measured = dP_measured * Q_inlet_est
            
            # Thermal Stats
            max_T = float(t_stats_cpu[0])
            avg_T = float(t_stats_cpu[1]) / float(t_stats_cpu[2]) if float(t_stats_cpu[2]) > 0 else 0.0
            
            # --- PUMP CONTROL LOOP ---
            # Soft Start Ramp (0 to 2000 steps)
            ramp_steps = 2000
            
            # --- PUMP MODE ---
            # "constant_power": adjusts flow to maintain fixed hydraulic power (dP * Q)
            #     Fair energy comparison, but complex geometries get less flow.
            # "constant_flow": fixes inlet velocity after ramp, all geometries get same flow.
            #     Isolates geometry effect on heat transfer.
            PUMP_MODE = "constant_flow"  # Change to "constant_flow" for equal-flow comparison
            
            # Constant Power Target (only used in constant_power mode)
            TARGET_POWER = 1 
            
            if step < ramp_steps:
                # Linear ramp to initial guess
                progress = step / float(ramp_steps)
                current_u = u_inlet_val * progress
                u_inlet_array.assign([max(1e-6, current_u)])
                
            # Active Control (After ramp)
            elif step > ramp_steps:
                if PUMP_MODE == "constant_power":
                    # Target: Constant Hydraulic Power
                    # Power = dP * Q
                    err_power = TARGET_POWER - power_measured
                    gain = 0.005 
                    u_inlet_val = u_inlet_val + gain * err_power
                    
                    # Safety Clamp
                    u_inlet_val = np.clip(u_inlet_val, 0.0001, 0.15)
                    u_inlet_array.assign([u_inlet_val])
                    
                # else: "constant_flow" — do nothing, inlet velocity stays at ramp final value
            
            # Blow-up detection
            if np.isnan(max_v) or max_v > 0.25:
                print(f"\n⚠️ BLOW-UP DETECTED at step {step}! max_v={max_v}")
                print("Simulation unstable - reduce inlet velocity or increase tau")
                break
            
            # Convergence Metrics
            d_ke = 0.0
            dT = 0.0
            
            if step > 1:
                d_ke = abs(ke - prev_ke) / (prev_ke + 1e-9) / stats_every
                d_max_T = abs(max_T - prev_max_T) / (prev_max_T + 1e-9) / stats_every
                
            prev_ke = ke
            prev_max_T = max_T
            
            # Calculate Accurate Instantaneous Reynolds Number
            nu = (tau_fluid - 0.5) / 3.0
            u_mean = u_inlet_val / porosity
            Re = (u_mean * Dh) / nu

            # Update TQDM
            pbar.set_postfix({
                "Re": f"{Re:.1f}",
                "U_in": f"{u_inlet_val:.4f}",
                "Power": f"{power_measured:.4f}",
                "MaxV": f"{max_v:.4f}",
                "MaxT": f"{max_T:.2f}",
                "dKE": f"{d_ke:.1e}",
                "dMaxT": f"{d_max_T:.1e}"
            })
            
            # Check Convergence
            # Warmup for 1000 steps to let flow develop
            if step > 1000 and d_ke < tol_ke and d_max_T < tol_T:
                print(f"\n\u2713 Converged at step {step}")
                print(f"Final dKE: {d_ke:.2e}, dMaxT: {d_max_T:.2e}")
                break
        
        pbar.update(1)

    u_np = u.numpy() # ONLY copy heavy data here
    fname = f"results/final_velocity.npy"
    np.save(fname, u_np)
    
    rho_np = rho_warp.numpy()
    np.save("results/final_density.npy", rho_np)
    
    T_np = T.numpy()
    np.save("results/final_temperature.npy", T_np)
            
    # --- Post-Simulation Analysis ---
    print("\n" + "="*30)
    print("   HYDRAULIC PERFORMANCE")
    print("="*30)
    
    # 1. Flow Rate (Q) - Sum of u_x at outlet (taking a slice near outlet)
    # Using slice at nx-2 to avoid boundary artifacts
    u_outlet = u_np[-2, :, :, 0] # shape (ny, nz)
    # Filter only fluid nodes at outlet
    mask_outlet = mask[-2, :, :]
    fluid_area = np.sum(mask_outlet != BC_SOLID)
    total_area = ny * nz
    
    Q = np.sum(u_outlet[mask_outlet != BC_SOLID])
    
    # 2. Pressure Drop (dP) in Lattice Units
    # 2. Pressure Drop (dP) in Lattice Units
    # Calculate actual pressure drop from simulation data
    rho_in_act = rho_np[1, :, :].mean()
    rho_out_act = rho_np[-2, :, :].mean()
    dP = (rho_in_act - rho_out_act) / 3.0
    
    # 3. Hydraulic Power (Power needed to drive flow)
    power = dP * Q
    
    # 4. Global Permeability (k)
    # Darcy: Q = (k * A * dP) / (nu * L)  =>  k = (Q * nu * L) / (A * dP)
    # L = nx (length of domain)
    # A = ny * nz (cross-sectional area)
    nu = (tau_fluid - 0.5) / 3.0
    k = (Q * nu * nx) / (total_area * dP) if dP > 0 else 0
    
    print(f"Flow Rate (Q):      {Q:.2e} [LU^3/t]")
    print(f"Pressure Drop (dP): {dP:.2e} [LU]")
    print(f"Hydraulic Power:    {power:.2e} [Energy/t]")
    print(f"Permeability (k):   {k:.2e} [LU^2]")
    print("-"*30)
    print("Higher Q/Power ratio = More Efficient")
    print("="*30)

    print("Done.")

if __name__ == "__main__":
    main()
