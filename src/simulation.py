import warp as wp
import numpy as np
from tqdm import tqdm

from src.fluid_engine3d import (BC_SOLID, compute_equilibrium_kernel, compute_macroscopic_kernel,
                                fused_equilibrium_collision_kernel,
                                streaming_kernel, compute_stats_kernel)
from src.thermal_engine3d import (compute_thermal_equilibrium_kernel,
                                  fused_thermal_equilibrium_collision_kernel,
                                  thermal_streaming_kernel, compute_thermal_macroscopic_kernel,
                                  compute_thermal_stats_kernel)


def run_simulation(mask,
                   max_steps=40000,
                   tau_fluid=0.52,
                   heat_power=0.1,
                   t_inlet=25.0,
                   u_inlet_val=0.03,
                   stats_every=100,
                   tol_ke=1e-7,
                   tol_T=1e-7,
                   pump_mode="constant_flow",
                   target_power=1.0,
                   device="cuda",
                   silent=False,
                   heat_source=None,
                   Pr=7.0,
                   k_ratio=628.0):
    """
    Runs a single LBM simulation and returns the final fields.

    Args:
        heat_source: Optional 3D numpy array (nx, ny, nz) of heat source
                     intensities (0.0-1.0). If None, uses the default
                     processor heatmap applied at the z=0 floor.
        Pr: Prandtl number. Controls thermal diffusivity ratio (default 7.0 = water).
        k_ratio: Solid/fluid thermal conductivity ratio (default 628.0 = copper/water).
    """
    wp.init()

    nx, ny, nz = mask.shape
    if not silent:
        print(f"Simulation Domain: {nx}x{ny}x{nz}")

    # Dh is only needed for the progress bar Re display
    Dh = 0.0
    if not silent:
        buf = 10
        core = mask[buf:-buf, 1:-1, 4:-2]
        fluid_mask = (core != BC_SOLID)
        solid_mask = (core == BC_SOLID)
        V_fluid = np.sum(fluid_mask)
        faces_x = np.sum(fluid_mask[1:,:,:] & solid_mask[:-1,:,:]) + np.sum(solid_mask[1:,:,:] & fluid_mask[:-1,:,:])
        faces_y = np.sum(fluid_mask[:,1:,:] & solid_mask[:,:-1,:]) + np.sum(solid_mask[:,1:,:] & fluid_mask[:,:-1,:])
        faces_z = np.sum(fluid_mask[:,:,1:] & solid_mask[:,:,:-1]) + np.sum(solid_mask[:,:,1:] & fluid_mask[:,:,:-1])
        A_wetted = faces_x + faces_y + faces_z
        Dh = 4.0 * V_fluid / A_wetted if A_wetted > 0 else float(ny)

    # Allocations
    rho_outlet = 1.0
    u_inlet_array = wp.array([u_inlet_val], dtype=float, device=device)
    rho_warp = wp.from_numpy(np.ones((nx, ny, nz), dtype=np.float32), dtype=wp.float32, device=device)
    u = wp.zeros((nx, ny, nz), dtype=wp.vec3f, device=device)
    domain_mask = wp.array(mask, dtype=wp.int32, device=device)

    f_old = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=device)
    f_new = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=device)

    tau_th_fluid = 0.5 + ((tau_fluid - 0.5) / Pr)
    tau_th_solid = 0.5 + k_ratio * (tau_th_fluid - 0.5)

    T = wp.full((nx, ny, nz), value=t_inlet, dtype=float, device=device)
    g_old = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    g_new = wp.zeros((19, nx, ny, nz), dtype=float, device=device)

    if heat_source is not None:
        heat_mask_np = np.array(heat_source, dtype=np.float32)
        if heat_mask_np.shape != (nx, ny, nz):
            raise ValueError(f"heat_source shape {heat_mask_np.shape} != domain shape ({nx}, {ny}, {nz})")
    else:
        heat_mask_np = np.zeros((nx, ny, nz), dtype=np.float32)
        heat_mask_np[:, :, 0] = 1.0
    if heat_mask_np.max() > 0:
        heat_mask_np /= heat_mask_np.max()
    heat_source_mask = wp.array(heat_mask_np, dtype=float, device=device)

    # Initialize directly into f_old / g_old (no intermediate f_eq / g_eq)
    wp.launch(compute_thermal_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, T, g_old, domain_mask], device=device)
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, f_old], device=device)

    # Capture CUDA graph for the time-stepping kernels
    wp.capture_begin(device=device)
    wp.launch(fused_equilibrium_collision_kernel, dim=(nx, ny, nz), inputs=[f_old, rho_warp, u, f_new, domain_mask, tau_fluid], device=device)
    wp.launch(streaming_kernel, dim=(nx, ny, nz), inputs=[f_new, f_old, domain_mask, u_inlet_array, rho_outlet, nx, ny, nz], device=device)
    wp.launch(compute_macroscopic_kernel, dim=(nx, ny, nz), inputs=[f_old, rho_warp, u], device=device)
    wp.launch(fused_thermal_equilibrium_collision_kernel, dim=(nx, ny, nz), inputs=[g_old, u, T, g_new, domain_mask, tau_th_fluid, tau_th_solid, heat_source_mask, heat_power], device=device)
    wp.launch(thermal_streaming_kernel, dim=(nx, ny, nz), inputs=[g_new, g_old, domain_mask, t_inlet, nx, ny, nz], device=device)
    wp.launch(compute_thermal_macroscopic_kernel, dim=(nx, ny, nz), inputs=[g_old, T], device=device)
    graph = wp.capture_end(device=device)

    # Minimum steps before convergence checks: at least 2 flow-throughs,
    # but capped at 60% of max_steps so there's room to actually converge
    min_steps = min(int(0.6 * max_steps), max(5000, int(2.0 * nx / max(u_inlet_val, 1e-6))))

    # Main loop
    pbar = tqdm(total=max_steps, disable=silent)
    prev_ke = 0.0
    prev_max_T = 0.0
    d_ke = 0.0
    dT_conv = 0.0
    max_v = 0.0
    u_in_eff = u_inlet_val
    stats_warp = wp.zeros(11, dtype=float, device=device)
    stats_thermal_warp = wp.zeros(5, dtype=float, device=device)
    prev_Q_out = 0.0
    dQ_conv = 1.0

    step = 0
    while step < max_steps:
        wp.capture_launch(graph)
        step += 1
        if step % stats_every == 0:
            stats_warp.zero_()
            stats_thermal_warp.zero_()
            wp.launch(compute_stats_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, domain_mask, stats_warp, nx, ny, nz], device=device)
            wp.launch(compute_thermal_stats_kernel, dim=(nx, ny, nz), inputs=[T, stats_thermal_warp, u, domain_mask], device=device)
            stats_cpu = stats_warp.numpy()
            t_stats_cpu = stats_thermal_warp.numpy()
            max_v = float(stats_cpu[0])
            ke = float(stats_cpu[2])
            max_T = float(t_stats_cpu[0])
            Q_out = float(t_stats_cpu[3])
            d_ke = abs(ke - prev_ke) / (prev_ke + 1e-9) / stats_every if step > stats_every else 0.0
            dT_conv = abs(max_T - prev_max_T) / (prev_max_T + 1e-9) / stats_every if step > stats_every else 0.0
            dQ_conv = abs(Q_out - prev_Q_out) / (abs(prev_Q_out) + 1e-12) / stats_every if step > stats_every else 1.0
            prev_ke, prev_max_T, prev_Q_out = ke, max_T, Q_out

            if pump_mode == "constant_power" and step > 2000:
                u_in_eff = u_inlet_array.numpy()[0]
                rho_in_avg = stats_cpu[5] / stats_cpu[8] if stats_cpu[8] > 0 else 1.0
                rho_out_avg = stats_cpu[6] / stats_cpu[9] if stats_cpu[9] > 0 else 1.0
                dP_measured = (rho_in_avg - rho_out_avg) / 3.0
                power_measured = dP_measured * (u_in_eff * stats_cpu[8])
                u_in_eff = np.clip(u_in_eff + 0.005 * (target_power - power_measured), 0.0001, 0.15)
                u_inlet_array.assign([u_in_eff])

            if not silent:
                pbar.set_postfix({"Re": f"{(u_in_eff*Dh)/((tau_fluid-0.5)/3.0):.1f}", "MaxV": f"{max_v:.4f}", "MaxT": f"{max_T:.2f}", "dKE": f"{d_ke:.1e}", "dQ": f"{dQ_conv:.1e}"})
            if step > min_steps and d_ke < tol_ke and dT_conv < tol_T and dQ_conv < tol_T:
                break
            if np.isnan(max_v) or max_v > 0.17 or np.isnan(max_T) or max_T > 1e4:
                if not silent:
                    Ma = max_v * (3.0 ** 0.5)
                    print(f"\n⚠ Simulation diverged at step {step}: MaxV={max_v:.2e}, Ma={Ma:.2f}, MaxT={max_T:.2e}")
                break
        pbar.update(1)
    pbar.close()

    converged = step > min_steps and d_ke < tol_ke and dT_conv < tol_T and dQ_conv < tol_T
    return {"u": u.numpy(), "rho": rho_warp.numpy(), "T": T.numpy(), "mask": mask,
            "steps": step, "d_ke": d_ke, "dT_conv": dT_conv, "dQ_conv": dQ_conv, "converged": converged}
