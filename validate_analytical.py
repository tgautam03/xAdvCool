import warp as wp
import numpy as np
from scipy.special import erfc
import sys

# Import core LBM kernels from xAdvCool
from src.fluid_engine3d import (BC_SOLID, BC_FLUID, BC_INLET, BC_OUTLET, 
                                E_CONST, W_CONST, OPPOSITE_CONST,
                                compute_equilibrium_kernel, compute_macroscopic_kernel,
                                bgk_collision_kernel, streaming_kernel)
from src.thermal_engine3d import (compute_thermal_equilibrium_kernel, thermal_bgk_collision_kernel, 
                                thermal_streaming_kernel, compute_thermal_macroscopic_kernel)

def test_1d_semi_infinite_conduction(device="cuda"):
    """
    Validates the thermal TLBM against the exact analytical solution for 1D Transient Heat Conduction 
    in a semi-infinite solid with a step change in surface temperature.
    Analytical Solution: T(x,t) = T_surface * erfc( x / (2 * sqrt(alpha * t)) )
    """
    print("\n" + "="*50)
    print("Running 1D Semi-Infinite Thermal Conduction Test")
    print("="*50)
    
    nx, ny, nz = 100, 3, 3
    # Pure solid domain
    mask = np.full((nx, ny, nz), BC_SOLID, dtype=np.int32)
    mask[0, :, :] = BC_INLET # The surface at x=0 is the heated inlet
    # We leave the rest as solid. The boundary at x=nx is handled via bounce-back or out-of-bounds, 
    # but we only simulate until before the thermal wave hits the end.
    
    domain_mask = wp.array(mask, dtype=wp.int32, device=device)
    
    # Thermal TLBM parameters
    tau_th_solid = 0.8
    # alpha = cs^2 * (tau - 0.5) = (1/3) * (0.8 - 0.5) = 0.1
    c_s_sq = 1.0 / 3.0
    alpha = c_s_sq * (tau_th_solid - 0.5)
    
    T_initial = 0.0
    T_surface = 100.0
    
    # Allocations
    T = wp.full((nx, ny, nz), value=T_initial, dtype=float, device=device)
    # The inlet relies on the t_inlet parameter passed to streaming
    
    # Need dummy u and rho for equilibrium (velocity is perfectly 0)
    u = wp.zeros((nx, ny, nz), dtype=wp.vec3f, device=device)
    rho_warp = wp.full((nx, ny, nz), value=1.0, dtype=float, device=device)
    
    g_old = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    g_new = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    g_eq = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    heat_source_mask = wp.zeros((nx, ny, nz), dtype=float, device=device)
    
    # Initialize distributions to equilibrium with T_initial
    wp.launch(compute_thermal_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, T, g_eq, domain_mask], device=device)
    wp.copy(g_old, g_eq)
    
    steps = 400
    
    # Run loop
    for step in range(steps):
        wp.launch(compute_thermal_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, T, g_eq, domain_mask], device=device)
        wp.launch(thermal_bgk_collision_kernel, dim=(nx, ny, nz), inputs=[g_old, g_eq, g_new, domain_mask, tau_th_solid, tau_th_solid, heat_source_mask, 0.0], device=device)
        wp.launch(thermal_streaming_kernel, dim=(nx, ny, nz), inputs=[g_new, g_old, domain_mask, T_surface, nx, ny, nz], device=device)
        wp.launch(compute_thermal_macroscopic_kernel, dim=(nx, ny, nz), inputs=[g_old, T], device=device)
        
    # Validation against Analytical Solution
    T_np = T.numpy()
    
    # Sample along x at center of y, z
    x_positions = np.arange(1, 50) # Avoid exact boundary cell (0) for cleaner metric
    T_sim = T_np[x_positions, 1, 1]
    
    # Analytical Erfc
    # Time t = steps. 
    T_analytical = T_surface * erfc(x_positions / (2.0 * np.sqrt(alpha * steps)))
    
    # Calculate errors
    abs_errors = np.abs(T_sim - T_analytical)
    rel_errors = abs_errors / T_surface # Normalize by max delta T
    max_error = np.max(rel_errors) * 100.0 # Percentage
    
    print(f"Simulation Alpha: {alpha:.4f}")
    print(f"Time Steps Evaluated: {steps}")
    print(f"Max Validation Error: {max_error:.3f}%")
    
    if max_error <= 3.0:
        print("✓ VALIDATION PASSED (<3% Error)")
    else:
        print("✗ VALIDATION FAILED (>3% Error)")
        print(f"Sim Dump: {T_sim[:10]}")
        print(f"Ana Dump: {T_analytical[:10]}")
        
    return max_error


def test_3d_rectangular_poiseuille(device="cuda"):
    """
    Validates the fluid TLBM against the exact analytical solution for fully developed Poiseuille flow
    in a rectangular duct (width 2a, height 2b) using purely stationary bounce-back walls.
    """
    print("\n" + "="*50)
    print("Running 3D Rectangular Poiseuille Flow Test")
    print("="*50)
    
    nx = 60 # Length (x)
    ny = 21 # Width (y) -> effectively 19 fluid cells
    nz = 11 # Height (z) -> effectively 9 fluid cells
    
    mask = np.full((nx, ny, nz), BC_FLUID, dtype=np.int32)
    # Set Walls
    mask[:, 0, :] = BC_SOLID
    mask[:, -1, :] = BC_SOLID
    mask[:, :, 0] = BC_SOLID
    mask[:, :, -1] = BC_SOLID
    
    # Set Inlet / Outlet
    mask[0, 1:-1, 1:-1] = BC_INLET
    mask[-1, 1:-1, 1:-1] = BC_OUTLET
    
    domain_mask = wp.array(mask, dtype=wp.int32, device=device)
    
    tau_fluid = 0.6
    u_inlet_val = 0.01
    u_inlet_array = wp.array([u_inlet_val], dtype=float, device=device)
    rho_outlet = 1.0
    
    rho_np = np.ones((nx, ny, nz), dtype=np.float32)
    rho_warp = wp.array(rho_np, dtype=wp.float32, device=device)
    u = wp.zeros((nx, ny, nz), dtype=wp.vec3f, device=device)
    f_old = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=device)
    f_new = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=device)
    f_eq = wp.zeros((19, nx, ny, nz), dtype=wp.float32, device=device)
    
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, f_eq], device=device)
    wp.copy(f_old, f_eq)
    
    # Define capture graph for speed
    wp.capture_begin(device=device)
    wp.launch(bgk_collision_kernel, dim=(nx, ny, nz), inputs=[f_old, f_eq, f_new, domain_mask, tau_fluid], device=device)
    wp.launch(streaming_kernel, dim=(nx, ny, nz), inputs=[f_new, f_old, domain_mask, u_inlet_array, rho_outlet, nx, ny, nz], device=device)
    wp.launch(compute_macroscopic_kernel, dim=(nx, ny, nz), inputs=[f_old, rho_warp, u], device=device)
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u, f_eq], device=device)
    graph = wp.capture_end(device=device)
    
    steps = 4000 # Wait for flow to develop
    
    sys.stdout.write("Simulating...")
    sys.stdout.flush()
    for s in range(steps):
        wp.capture_launch(graph)
        if s % 1000 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
    print(" Done.")
            
    u_np = u.numpy()
    
    # Cross-section at fully developed region
    slice_x = nx - 10
    u_slice = u_np[slice_x, :, :, 0] # Extract UX component
    
    # Analytical calculation
    # Domain sizes: ny has ny-2 fluid cells, nz has nz-2 fluid cells
    # a = half width, b = half height
    w_cells = ny - 2
    h_cells = nz - 2
    a = w_cells / 2.0
    b = h_cells / 2.0
    
    # Instead of deriving theoretical dP/dx and integrating, we normalize against 
    # the exact maximum centerline velocity from the simulation. 
    # This specifically validates the *shape* of the 2D cross-section and the momentum transport 
    # mechanics (viscosity/wall friction ratios), which is standard practice for LBM profile validation.
    
    u_centerline_sim = np.max(u_slice)
    
    y = np.linspace(-w_cells/2 + 0.5, w_cells/2 - 0.5, w_cells)
    z = np.linspace(-h_cells/2 + 0.5, h_cells/2 - 0.5, h_cells)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    
    # Analytical series shape function (unnormalized)
    def poiseuille_shape(Y, Z, a, b, num_terms=30):
        val = np.zeros_like(Y)
        for i in range(1, num_terms*2, 2): # Odd terms: 1, 3, 5...
            term1 = ((-1)**((i-1)//2)) / (i**3)
            term2 = 1.0 - np.cosh(i * np.pi * Z / (2*a)) / np.cosh(i * np.pi * b / (2*a))
            term3 = np.cos(i * np.pi * Y / (2*a))
            val += term1 * term2 * term3
        return val
        
    shape_field = poiseuille_shape(Y, Z, a, b)
    shape_max = np.max(shape_field)
    
    # Normalized analytical profile scaled to the same center velocity
    u_analytical = (shape_field / shape_max) * u_centerline_sim
    
    # Extract just the fluid cells
    u_sim_fluid = u_slice[1:-1, 1:-1]
    
    abs_errors = np.abs(u_sim_fluid - u_analytical)
    rel_errors = abs_errors / u_centerline_sim
    max_error = np.max(rel_errors) * 100.0
    
    print(f"Centerline Velocity: {u_centerline_sim:.5f}")
    print(f"Max Validation Error: {max_error:.3f}%")
    
    if max_error <= 3.0:
        print("✓ VALIDATION PASSED (<3% Error)")
    else:
        print("✗ VALIDATION FAILED (>3% Error)")
        
    return max_error


@wp.kernel
def periodic_streaming_kernel(f_post_collision: wp.array4d(dtype=float),
                              f_streamed: wp.array4d(dtype=float),
                              nx: int, ny: int, nz: int):
    x, y, z = wp.tid()
    for i in range(19):
        # We need to use E_CONST from fluid_engine3d
        # We can't access it natively inside wp.kernel without importing it directly in warp symbol scope,
        # but E_CONST is already warp-accessible. 
        # Warp kernels resolve outer scope constants.
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        
        src_x = (x - ex + nx) % nx
        src_y = (y - ey + ny) % ny
        src_z = (z - ez + nz) % nz
        
        f_streamed[i, x, y, z] = f_post_collision[i, src_x, src_y, src_z]

def test_taylor_green_vortex(device="cuda"):
    """
    Validates the pure fluid kinematic viscosity by simulating the 2D Taylor-Green Vortex decay 
    in a 3D periodic domain. The kinetic energy should decay exactly exponentially.
    """
    print("\n" + "="*50)
    print("Running Taylor-Green Vortex Decay Test")
    print("="*50)
    
    L = 32
    nx, ny, nz = L, L, 2
    
    tau_fluid = 0.55
    nu = (tau_fluid - 0.5) / 3.0
    
    U0 = 0.05
    rho0 = 1.0
    k = 2.0 * np.pi / float(L)
    
    # Initialize Analytical Fields
    x_coords = np.arange(nx)
    y_coords = np.arange(ny)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    
    ux = U0 * np.sin(k * X) * np.cos(k * Y)
    uy = -U0 * np.cos(k * X) * np.sin(k * Y)
    uz = np.zeros_like(X)
    
    u_np = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    rho_np = np.zeros((nx, ny, nz), dtype=np.float32)
    
    for z in range(nz):
        u_np[:, :, z, 0] = ux
        u_np[:, :, z, 1] = uy
        u_np[:, :, z, 2] = uz
        rho_np[:, :, z] = rho0 - (rho0 * U0**2 / (4.0 * (1.0/3.0))) * (np.cos(2*k*X) + np.cos(2*k*Y))
        
    u_warp = wp.array(u_np, dtype=wp.vec3f, device=device)
    rho_warp = wp.array(rho_np, dtype=float, device=device)
    
    f_old = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    f_new = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    f_eq  = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    
    # Dummy mask to satisfy collision kernel signature, though periodic relies on periodic_streaming_kernel
    mask = wp.full((nx, ny, nz), value=BC_FLUID, dtype=int, device=device)
    
    # Initialize to equilibrium
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u_warp, f_eq], device=device)
    wp.copy(f_old, f_eq)
    
    steps = 1000
    kinetic_energies = []
    
    # Calculate initial kinetic energy
    ke0 = 0.5 * rho0 * np.sum(u_np[:,:,:,0]**2 + u_np[:,:,:,1]**2)
    kinetic_energies.append(ke0)
    
    wp.capture_begin(device=device)
    wp.launch(bgk_collision_kernel, dim=(nx, ny, nz), inputs=[f_old, f_eq, f_new, mask, tau_fluid], device=device)
    wp.launch(periodic_streaming_kernel, dim=(nx, ny, nz), inputs=[f_new, f_old, nx, ny, nz], device=device)
    wp.launch(compute_macroscopic_kernel, dim=(nx, ny, nz), inputs=[f_old, rho_warp, u_warp], device=device)
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u_warp, f_eq], device=device)
    graph = wp.capture_end(device=device)
    
    sys.stdout.write("Simulating...")
    sys.stdout.flush()
    for s in range(1, steps + 1):
        wp.capture_launch(graph)
        
        # Measure KE periodically
        if s % 100 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
            u_curr = u_warp.numpy()
            ke_curr = 0.5 * rho0 * np.sum(u_curr[:,:,:,0]**2 + u_curr[:,:,:,1]**2)
            kinetic_energies.append(ke_curr)
            
    print(" Done.")
    
    # Analytical KE Decay: E(t) = E(0) * exp(-4 * nu * k^2 * t)
    t_arr = np.arange(0, steps + 1, 100)
    ke_analytical = ke0 * np.exp(-4.0 * nu * (k**2) * t_arr)
    ke_simulated = np.array(kinetic_energies)
    
    abs_errors = np.abs(ke_simulated - ke_analytical)
    # Normalize error by E(0)
    rel_errors = abs_errors / ke0
    max_error = np.max(rel_errors) * 100.0
    
    print(f"Final Analytical KE: {ke_analytical[-1]:.6f}")
    print(f"Final Simulated KE:  {ke_simulated[-1]:.6f}")
    print(f"Max Validation Error: {max_error:.3f}%")
    
    if max_error <= 3.0:
        print("✓ VALIDATION PASSED (<3% Error)")
    else:
        print("✗ VALIDATION FAILED (>3% Error)")
        
    return max_error


@wp.kernel
def forced_bgk_collision_kernel(f_old: wp.array4d(dtype=float),
                                f_eq: wp.array4d(dtype=float),
                                f_new: wp.array4d(dtype=float),
                                mask: wp.array3d(dtype=int),
                                tau: float,
                                force_x: float):
    x, y, z = wp.tid()
    m = mask[x, y, z]
    omega = 1.0 / tau
    
    if m != BC_SOLID:
        for i in range(19):
            ex = float(E_CONST[i, 0])
            source = W_CONST[i] * 3.0 * ex * force_x
            f_new[i, x, y, z] = f_old[i, x, y, z] - omega * (f_old[i, x, y, z] - f_eq[i, x, y, z]) + source
    else:
        for i in range(19):
            f_new[i, x, y, z] = f_old[i, x, y, z]

@wp.kernel
def periodic_streaming_with_solids_kernel(f_post_collision: wp.array4d(dtype=float),
                                          f_streamed: wp.array4d(dtype=float),
                                          mask: wp.array3d(dtype=int),
                                          nx: int, ny: int, nz: int):
    x, y, z = wp.tid()
    m = mask[x, y, z]
    if m == BC_SOLID:
        return
        
    for i in range(19):
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        
        src_x = (x - ex + nx) % nx
        src_y = (y - ey + ny) % ny
        src_z = (z - ez + nz) % nz
        
        neighbor_type = mask[src_x, src_y, src_z]
        if neighbor_type == BC_SOLID:
            f_streamed[i, x, y, z] = f_post_collision[OPPOSITE_CONST[i], x, y, z]
        else:
            f_streamed[i, x, y, z] = f_post_collision[i, src_x, src_y, src_z]

def test_kuwabara_pin_fin(device="cuda"):
    print("\n" + "="*50)
    print("Running Kuwabara Periodic Pin-Fin Drag Test")
    print("="*50)
    
    L = 40
    R = 8.0
    nx, ny, nz = L, L, 3
    
    mask_np = np.full((nx, ny, nz), BC_FLUID, dtype=np.int32)
    x_c, y_c = L / 2.0, L / 2.0
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    dist_sq = (xx - x_c)**2 + (yy - y_c)**2
    
    cylinder_mask = dist_sq <= R**2
    for z in range(nz):
        mask_np[cylinder_mask, z] = BC_SOLID
        
    mask = wp.array(mask_np, dtype=wp.int32, device=device)
    
    tau_fluid = 0.8
    nu = (tau_fluid - 0.5) / 3.0
    force_x = 1e-5
    
    rho_np = np.ones((nx, ny, nz), dtype=np.float32)
    rho_warp = wp.array(rho_np, dtype=wp.float32, device=device)
    u_warp = wp.zeros((nx, ny, nz), dtype=wp.vec3f, device=device)
    
    f_old = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    f_new = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    f_eq  = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u_warp, f_eq], device=device)
    wp.copy(f_old, f_eq)
    
    steps = 4000
    wp.capture_begin(device=device)
    wp.launch(forced_bgk_collision_kernel, dim=(nx, ny, nz), inputs=[f_old, f_eq, f_new, mask, tau_fluid, force_x], device=device)
    wp.launch(periodic_streaming_with_solids_kernel, dim=(nx, ny, nz), inputs=[f_new, f_old, mask, nx, ny, nz], device=device)
    wp.launch(compute_macroscopic_kernel, dim=(nx, ny, nz), inputs=[f_old, rho_warp, u_warp], device=device)
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u_warp, f_eq], device=device)
    graph = wp.capture_end(device=device)
    
    sys.stdout.write("Simulating...")
    sys.stdout.flush()
    for s in range(steps):
        wp.capture_launch(graph)
        if s % 1000 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
    print(" Done.")
    
    u_np = u_warp.numpy()
    
    U_simulated = np.mean(u_np[:, :, 1, 0])
    
    N_solid = np.sum(cylinder_mask)
    N_total = nx * ny
    phi = float(N_solid) / float(N_total)
    
    K_phi = -0.5 * np.log(phi) - 0.75 + phi - 0.25 * (phi**2)
    U_analytical = (force_x * L**2 * K_phi) / (4.0 * np.pi * nu)
    
    abs_error = abs(U_simulated - U_analytical)
    rel_error = abs_error / U_analytical
    max_error = rel_error * 100.0
    
    print(f"Solid Volume Fraction (phi): {phi:.4f}")
    print(f"Analytical Superficial Velocity: {U_analytical:.6f}")
    print(f"Simulated Superficial Velocity:  {U_simulated:.6f}")
    print(f"Validation Error: {max_error:.3f}%")
    
    if max_error <= 5.0:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")
        
    return max_error


@wp.kernel
def couette_streaming_kernel(f_post_collision: wp.array4d(dtype=float),
                             f_streamed: wp.array4d(dtype=float),
                             mask: wp.array3d(dtype=int),
                             u_wall_x: float,
                             nx: int, ny: int, nz: int):
    x, y, z = wp.tid()
    m = mask[x, y, z]
    if m == BC_SOLID:
        return
        
    for i in range(19):
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        
        src_x = x - ex
        src_y = y - ey
        src_z = z - ez
        
        # Periodic in X and Y
        src_x = (src_x + nx) % nx
        src_y = (src_y + ny) % ny
        
        if src_z < 0:
            f_streamed[i, x, y, z] = f_post_collision[OPPOSITE_CONST[i], x, y, z]
        elif src_z >= nz:
            opp_i = OPPOSITE_CONST[i]
            # Momentum transfer from moving wall
            momentum = 6.0 * W_CONST[i] * float(ex) * u_wall_x
            f_streamed[i, x, y, z] = f_post_collision[opp_i, x, y, z] + momentum
        else:
            neighbor_type = mask[src_x, src_y, src_z]
            if neighbor_type == BC_SOLID:
                f_streamed[i, x, y, z] = f_post_collision[OPPOSITE_CONST[i], x, y, z]
            else:
                f_streamed[i, x, y, z] = f_post_collision[i, src_x, src_y, src_z]

def test_couette_flow(device="cuda"):
    """
    Validates the fluid TLBM boundary conditions by simulating a shear-driven Couette flow.
    The top boundary moves tangentially, driving the fluid to a linear analytical velocity profile.
    """
    print("\n" + "="*50)
    print("Running 3D Couette Flow (Shear-Driven) Test")
    print("="*50)
    
    nx, ny, nz = 5, 5, 21
    mask_np = np.full((nx, ny, nz), BC_FLUID, dtype=np.int32)
    mask = wp.array(mask_np, dtype=wp.int32, device=device)
    
    tau_fluid = 0.6
    u_wall_x = 0.05
    
    rho_np = np.ones((nx, ny, nz), dtype=np.float32)
    rho_warp = wp.array(rho_np, dtype=wp.float32, device=device)
    u_warp = wp.zeros((nx, ny, nz), dtype=wp.vec3f, device=device)
    
    f_old = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    f_new = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    f_eq  = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u_warp, f_eq], device=device)
    wp.copy(f_old, f_eq)
    
    steps = 20000
    wp.capture_begin(device=device)
    wp.launch(bgk_collision_kernel, dim=(nx, ny, nz), inputs=[f_old, f_eq, f_new, mask, tau_fluid], device=device)
    wp.launch(couette_streaming_kernel, dim=(nx, ny, nz), inputs=[f_new, f_old, mask, u_wall_x, nx, ny, nz], device=device)
    wp.launch(compute_macroscopic_kernel, dim=(nx, ny, nz), inputs=[f_old, rho_warp, u_warp], device=device)
    wp.launch(compute_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u_warp, f_eq], device=device)
    graph = wp.capture_end(device=device)
    
    sys.stdout.write("Simulating...")
    sys.stdout.flush()
    for s in range(steps):
        wp.capture_launch(graph)
        if s % 1000 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
    print(" Done.")
    
    u_np = u_warp.numpy()
    
    H = float(nz)
    u_sim_profile = u_np[2, 2, :, 0] # Z Profile at center
    z_coords = np.arange(nz, dtype=float)
    u_ana_profile = u_wall_x * (z_coords + 0.5) / H
    
    abs_errors = np.abs(u_sim_profile - u_ana_profile)
    rel_errors = abs_errors / u_wall_x
    max_error = np.max(rel_errors) * 100.0
    
    print(f"Analytical Top Velocity: {u_ana_profile[-1]:.6f}")
    print(f"Simulated Top Velocity:  {u_sim_profile[-1]:.6f}")
    print(f"Max Validation Error: {max_error:.3f}%")
    
    if max_error <= 3.0:
        print("✓ VALIDATION PASSED (<3% Error)")
    else:
        print("✗ VALIDATION FAILED (>3% Error)")
        
    return max_error


@wp.kernel
def periodic_thermal_streaming_kernel(g_post_collision: wp.array4d(dtype=float),
                                      g_streamed: wp.array4d(dtype=float),
                                      nx: int, ny: int, nz: int):
    x, y, z = wp.tid()
    for i in range(19):
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        
        src_x = (x - ex + nx) % nx
        src_y = (y - ey + ny) % ny
        src_z = (z - ez + nz) % nz
        
        g_streamed[i, x, y, z] = g_post_collision[i, src_x, src_y, src_z]

def test_advection_diffusion_gaussian(device="cuda"):
    print("\n" + "="*50)
    print("Running Advection-Diffusion Gaussian Hot Spot Test")
    print("="*50)
    
    nx, ny, nz = 200, 3, 3
    L = float(nx)
    
    tau_th = 0.8
    # alpha = 1 / 3 * (0.8 - 0.5) = 0.1
    c_s_sq = 1.0 / 3.0
    alpha = c_s_sq * (tau_th - 0.5)
    
    U0 = 0.05
    u_np = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    u_np[:, :, :, 0] = U0
    
    x0 = 50.0
    sigma0_sq = 10.0
    
    T_np = np.zeros((nx, ny, nz), dtype=float)
    x_coords = np.arange(nx, dtype=float)
    T_1d = np.exp(- ((x_coords - x0)**2) / (2.0 * sigma0_sq) )
    
    for y in range(ny):
        for z in range(nz):
            T_np[:, y, z] = T_1d
            
    rho_warp = wp.full((nx, ny, nz), value=1.0, dtype=float, device=device)
    u_warp = wp.array(u_np, dtype=wp.vec3f, device=device)
    T_warp = wp.array(T_np, dtype=float, device=device)
    
    g_old = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    g_new = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    g_eq = wp.zeros((19, nx, ny, nz), dtype=float, device=device)
    
    mask = wp.full((nx, ny, nz), value=BC_FLUID, dtype=int, device=device)
    heat_source_mask = wp.zeros((nx, ny, nz), dtype=float, device=device)
    
    wp.launch(compute_thermal_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u_warp, T_warp, g_eq, mask], device=device)
    wp.copy(g_old, g_eq)
    
    steps = 1000
    wp.capture_begin(device=device)
    wp.launch(thermal_bgk_collision_kernel, dim=(nx, ny, nz), inputs=[g_old, g_eq, g_new, mask, tau_th, tau_th, heat_source_mask, 0.0], device=device)
    wp.launch(periodic_thermal_streaming_kernel, dim=(nx, ny, nz), inputs=[g_new, g_old, nx, ny, nz], device=device)
    wp.launch(compute_thermal_macroscopic_kernel, dim=(nx, ny, nz), inputs=[g_old, T_warp], device=device)
    wp.launch(compute_thermal_equilibrium_kernel, dim=(nx, ny, nz), inputs=[rho_warp, u_warp, T_warp, g_eq, mask], device=device)
    graph = wp.capture_end(device=device)
    
    sys.stdout.write("Simulating...")
    sys.stdout.flush()
    for s in range(steps):
        wp.capture_launch(graph)
        if s % 200 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()
    print(" Done.")
    
    T_final = T_warp.numpy()[:, 1, 1]
    
    sum_T = np.sum(T_final)
    pdf = T_final / sum_T
    
    mean_sim = np.sum(x_coords * pdf)
    var_sim = np.sum(((x_coords - mean_sim)**2) * pdf)
    
    var_ana = sigma0_sq + 2.0 * alpha * steps
    mean_ana = x0 + U0 * steps
    
    err_mean = abs(mean_sim - mean_ana) / mean_ana * 100.0
    err_var = abs(var_sim - var_ana) / var_ana * 100.0
    max_error = max(err_mean, err_var)
    
    print(f"Analytical Mean Pos: {mean_ana:.4f}")
    print(f"Simulated Mean Pos:  {mean_sim:.4f}")
    print(f"Analytical Variance: {var_ana:.4f}")
    print(f"Simulated Variance:  {var_sim:.4f}")
    print(f"Max Validation Error: {max_error:.3f}%")
    
    if max_error <= 3.0:
        print("✓ VALIDATION PASSED (<3% Error)")
    else:
        print("✗ VALIDATION FAILED (>3% Error)")
        
    return max_error


if __name__ == "__main__":
    wp.init()
    test_1d_semi_infinite_conduction()
    test_3d_rectangular_poiseuille()
    test_taylor_green_vortex()
    test_kuwabara_pin_fin()
    test_couette_flow()
    test_advection_diffusion_gaussian()
