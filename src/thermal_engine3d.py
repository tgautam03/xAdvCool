import warp as wp
from src.fluid_engine3d import E_CONST, W_CONST, OPPOSITE_CONST, BC_SOLID, BC_FLUID, BC_INLET, BC_OUTLET

# Thermal Lattice Constants are the same as Fluid D3Q19
# We solve for Temperature (T) as a scalar passive scalar (with CHT)

@wp.kernel
def compute_thermal_equilibrium_kernel(rho: wp.array3d(dtype=float),      # Fluid rho (optional, can just use 1.0 approx) # type: ignore
                                       u: wp.array3d(dtype=wp.vec3f),     # Fluid velocity (0 in solid) # type: ignore
                                       T: wp.array3d(dtype=float),        # Temperature field # type: ignore
                                       g_eq: wp.array4d(dtype=float),     # Thermal distributions # type: ignore
                                       mask: wp.array3d(dtype=int)        # Domain mask # type: ignore
                                       ):
    x, y, z = wp.tid()
    
    # Get local macroscopic T
    local_T = T[x, y, z]
    
    # Get velocity
    # In CHT, velocity is 0 in solid nodes, actual velocity in fluid nodes
    # The input 'u' should already be 0 in solids from the fluid solver, 
    # but we can enforce it here to be safe if 'u' isn't cleared.
    # Typically fluid solver sets u=0 in solids.
    local_u = u[x, y, z]
    m = mask[x, y, z]
    if m == BC_SOLID:
        local_u = wp.vec3f(0.0, 0.0, 0.0)
        
    u_sq = wp.dot(local_u, local_u)
    
    # Compute Equilibrium (Advection-Diffusion)
    # g_eq_i = T * w_i * (1 + 3(e*u))  <- Zero-th and First order is often enough for thermal
    # Higher order: T * w_i * (1 + 3(e*u) + 4.5(e*u)^2 - 1.5u^2)
    # We'll use the higher order one matching the fluid eq for consistency/accuracy
    
    for i in range(19):
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        
        e_u = float(ex) * local_u.x + float(ey) * local_u.y + float(ez) * local_u.z
        
        val = W_CONST[i] * local_T * (
            1.0 + 3.0 * e_u + 4.5 * (e_u * e_u) - 1.5 * u_sq
        )
        
        g_eq[i, x, y, z] = val

@wp.kernel
def thermal_bgk_collision_kernel(g_old: wp.array4d(dtype=float),            # Input
                                 g_eq: wp.array4d(dtype=float),             # Input
                                 g_new: wp.array4d(dtype=float),            # Output
                                 mask: wp.array3d(dtype=int),               # Domain mask
                                 tau_fluid: float,
                                 tau_solid: float,
                                 heat_source_mask: wp.array3d(dtype=float), # 1.0 explicitly implies heating
                                 heat_source_power: float
                                 ):
    x, y, z = wp.tid()
    
    m = mask[x, y, z]
    
    # Determine relaxation time
    # This controls diffusivity: alpha = cs^2 * (tau - 0.5)
    tau = tau_fluid
    if m == BC_SOLID:
        tau = tau_solid
        
    omega = 1.0 / tau
    
    # Heat Source term
    # Q_term = w_i * Q_input
    # Apply as multiplier: Power = MaskValues * BasePower
    # This enables variable heat distribution (e.g. Cores=1.0, Cache=0.5)
    mask_val = heat_source_mask[x, y, z]
    q_val = mask_val * heat_source_power
        
    # BGK Collision step
    for i in range(19):
        val = g_old[i, x, y, z] - omega * (g_old[i, x, y, z] - g_eq[i, x, y, z])
        
        # Source is added as w_i * (1 - 0.5 * omega) * Q
        if q_val > 0.0:
            val += W_CONST[i] * (1.0 - 0.5 * omega) * q_val
            
        g_new[i, x, y, z] = val

#######################################################################
######## Fused Thermal Equilibrium + Collision (no g_eq array) ########
#######################################################################
@wp.kernel
def fused_thermal_equilibrium_collision_kernel(
    g_old: wp.array4d(dtype=float),            # Input: (19, nx, ny, nz) # type:ignore
    u: wp.array3d(dtype=wp.vec3f),             # Input: fluid velocity # type:ignore
    T: wp.array3d(dtype=float),                # Input: temperature field # type:ignore
    g_new: wp.array4d(dtype=float),            # Output: (19, nx, ny, nz) # type:ignore
    mask: wp.array3d(dtype=int),               # Domain mask # type:ignore
    tau_fluid: float,
    tau_solid: float,
    heat_source_mask: wp.array3d(dtype=float), # type:ignore
    heat_source_power: float
):
    """Computes thermal equilibrium in registers and applies BGK collision + heat source
    in one pass. Eliminates the separate g_eq array and its global memory round-trip."""
    x, y, z = wp.tid()

    m = mask[x, y, z]

    # Select relaxation time based on phase
    tau = tau_fluid
    if m == BC_SOLID:
        tau = tau_solid
    omega = 1.0 / tau

    # Heat source
    mask_val = heat_source_mask[x, y, z]
    q_val = mask_val * heat_source_power

    # Get local temperature
    local_T = T[x, y, z]

    # Get velocity (zero in solid for CHT)
    local_u = u[x, y, z]
    if m == BC_SOLID:
        local_u = wp.vec3f(0.0, 0.0, 0.0)
    u_sq = wp.dot(local_u, local_u)

    for i in range(19):
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        e_u = float(ex) * local_u.x + float(ey) * local_u.y + float(ez) * local_u.z

        # Equilibrium computed in register (never written to global memory)
        g_eq_i = W_CONST[i] * local_T * (
            1.0 + 3.0 * e_u + 4.5 * (e_u * e_u) - 1.5 * u_sq
        )

        val = g_old[i, x, y, z] - omega * (g_old[i, x, y, z] - g_eq_i)

        # Source term
        if q_val > 0.0:
            val += W_CONST[i] * (1.0 - 0.5 * omega) * q_val

        g_new[i, x, y, z] = val

@wp.kernel
def thermal_streaming_kernel(g_post_collision: wp.array4d(dtype=float), # Input
                             g_streamed: wp.array4d(dtype=float),       # Output
                             mask: wp.array3d(dtype=int),
                             T_inlet: float,
                             nx: int, ny: int, nz: int
                             ):
    x, y, z = wp.tid()
    
    m = mask[x, y, z]

    # --- 1. Explicit Inlet Handler (Dirichlet) ---
    # If the node is marked as Inlet in the mask, force it to T_inlet.
    if m == BC_INLET:
        for i in range(19):
            # Enforce equilibrium at T_inlet (assuming u=0 for simplicity, or negligible)
            # effectively g = w_i * T_inlet
            g_streamed[i, x, y, z] = W_CONST[i] * T_inlet
        return

    # --- 2. Explicit Outlet Handler (Neumann / Open) ---
    # If the node is marked as Outlet, copy state from upstream (x-1).
    # This creates a zero-gradient buffer that lets heat flow out.
    if m == BC_OUTLET:
        # Use upstream neighbor
        src_x_int = x - 1
        if src_x_int < 0: src_x_int = 0
        
        for i in range(19):
            g_streamed[i, x, y, z] = g_post_collision[i, src_x_int, y, z]
        return

    # --- 3. Standard Streaming (Fluid & Solid) ---
    for i in range(19):
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        
        src_x = x - ex
        src_y = y - ey
        src_z = z - ez
        
        # Check Domain Boundaries
        if src_x < 0 or src_x >= nx or src_y < 0 or src_y >= ny or src_z < 0 or src_z >= nz:
            # Out of bounds source
            
            # Inlet side (x < 0): Fixed Temperature T_inlet
            if src_x < 0:
                g_streamed[i, x, y, z] = W_CONST[i] * T_inlet
                
            # Outlet side (x >= nx): Zero Gradient Extrapolation
            elif src_x >= nx:
                # Copy from current node (Neumann approx) or slightly inside if possible.
                # Since we are at x=nx-1 (implied), we want x-1.
                # Using current node value g[x] effectively assumes dg/dx=0 locally
                g_streamed[i, x, y, z] = g_post_collision[i, x, y, z]
                
            # Other walls: Adiabatic (Bounce Back)
            else:
                opp_i = OPPOSITE_CONST[i]
                g_streamed[i, x, y, z] = g_post_collision[opp_i, x, y, z]
                
        else:
            # Source is inside domain - Normal streaming
            g_streamed[i, x, y, z] = g_post_collision[i, src_x, src_y, src_z]


@wp.kernel
def compute_thermal_macroscopic_kernel(g: wp.array4d(dtype=float),       # Input
                                       T: wp.array3d(dtype=float)        # Output
                                       ):
    x, y, z = wp.tid()
    
    sum_g = float(0.0)
    for i in range(19):
        sum_g += g[i, x, y, z]
        
    T[x, y, z] = sum_g

@wp.kernel
def compute_thermal_stats_kernel(T: wp.array3d(dtype=float),        # Input
                                 stats: wp.array(dtype=float)       # Output: [max_t, sum_t, count]
                                 ):
    x, y, z = wp.tid()
    
    val = T[x, y, z]
    
    wp.atomic_max(stats, 0, val)
    wp.atomic_add(stats, 1, val)
    wp.atomic_add(stats, 2, 1.0)
