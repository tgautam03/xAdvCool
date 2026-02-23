import warp as wp

"""    
3D LBM engine using D3Q19 lattice
- Left face (x=0): Inlet
- Right face (x=nx-1): Outlet  
- Other faces (y=0, y=ny-1, z=0, z=nz-1): Solid walls
"""

# Boundary conditions index
BC_FLUID = 0
BC_SOLID = 1
BC_INLET = 2
BC_OUTLET = 3


# D3Q19 Lattice vectors as 19x3 matrix (col 0 = ex, col 1 = ey, col 2 = ez)
# Order: Rest, 6 face-aligned, 12 edge-aligned
E_CONST = wp.constant(wp.mat((19, 3), dtype=wp.int32)(
    0, 0, 0,    # 0: Rest
    1, 0, 0,    # 1: +X
   -1, 0, 0,    # 2: -X
    0, 1, 0,    # 3: +Y
    0,-1, 0,    # 4: -Y
    0, 0, 1,    # 5: +Z
    0, 0,-1,    # 6: -Z
    1, 1, 0,    # 7: +X+Y
   -1,-1, 0,    # 8: -X-Y
    1,-1, 0,    # 9: +X-Y
   -1, 1, 0,    # 10: -X+Y
    1, 0, 1,    # 11: +X+Z
   -1, 0,-1,    # 12: -X-Z
    1, 0,-1,    # 13: +X-Z
   -1, 0, 1,    # 14: -X+Z
    0, 1, 1,    # 15: +Y+Z
    0,-1,-1,    # 16: -Y-Z
    0, 1,-1,    # 17: +Y-Z
    0,-1, 1     # 18: -Y+Z
))

# D3Q19 Weights: 1/3 (rest), 1/18 (face), 1/36 (edge)
W_CONST = wp.constant(wp.vec(19, dtype=wp.float32)(
    1.0/3.0,                                    # Rest
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,  # Face-aligned
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,  # Edge-aligned
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
))

# Opposite directions for bounce-back
# 0->0, 1<->2, 3<->4, 5<->6, 7<->8, 9<->10, 11<->12, 13<->14, 15<->16, 17<->18
OPPOSITE_CONST = wp.constant(wp.vec(19, dtype=wp.int32)(
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17
))

#######################################################################
##################### Computing equilibrium: f_eq #####################
#######################################################################
@wp.kernel
def compute_equilibrium_kernel(rho: wp.array3d(dtype=float),      # Input: (nx, ny, nz) # type: ignore
                               u: wp.array3d(dtype=wp.vec3f),     # Input: (nx, ny, nz) # type: ignore
                               f_eq: wp.array4d(dtype=float)      # Output: (19, nx, ny, nz) # type: ignore
                               ):
    x, y, z = wp.tid()
    
    local_rho = rho[x, y, z]
    local_u = u[x, y, z]
    
    u_sq = wp.dot(local_u, local_u)
    
    for i in range(19):
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        e_u = float(ex) * local_u.x + float(ey) * local_u.y + float(ez) * local_u.z
        
        val = W_CONST[i] * local_rho * (
            1.0 + 3.0 * e_u + 4.5 * (e_u * e_u) - 1.5 * u_sq
        )
        
        f_eq[i, x, y, z] = val

#######################################################################
################# Computing macroscopic values: rho, u ################
#######################################################################
@wp.kernel
def compute_macroscopic_kernel(f: wp.array4d(dtype=float),       # Input: (19, nx, ny, nz) # type:ignore
                               rho: wp.array3d(dtype=float),     # Output: (nx, ny, nz) # type:ignore
                               u: wp.array3d(dtype=wp.vec3f)     # Output: (nx, ny, nz) # type:ignore
                               ):
    x, y, z = wp.tid()
    
    local_rho = float(0.0)
    momentum_x = float(0.0)
    momentum_y = float(0.0)
    momentum_z = float(0.0)
    
    for i in range(19):
        val = f[i, x, y, z]
        local_rho += val
        
        ex = E_CONST[i, 0]
        ey = E_CONST[i, 1]
        ez = E_CONST[i, 2]
        momentum_x += val * float(ex)
        momentum_y += val * float(ey)
        momentum_z += val * float(ez)
    
    rho[x, y, z] = local_rho
    
    if local_rho > 1e-10:
        u[x, y, z] = wp.vec3f(momentum_x / local_rho, momentum_y / local_rho, momentum_z / local_rho)
    else:
        u[x, y, z] = wp.vec3f(0.0, 0.0, 0.0)

#######################################################################
################## Collision Operator: f_post_collision ###############
#######################################################################
@wp.kernel
def bgk_collision_kernel(f_old: wp.array4d(dtype=float),            # Input: (19, nx, ny, nz) # type:ignore
                         f_eq: wp.array4d(dtype=float),             # Input: (19, nx, ny, nz) # type:ignore
                         f_post_collision: wp.array4d(dtype=float), # Output: (19, nx, ny, nz) # type:ignore
                         domain_mask: wp.array3d(dtype=int),        # 3D mask # type:ignore
                         tau: float
                         ):
    x, y, z = wp.tid()
    
    mask = domain_mask[x, y, z]
    omega = 1.0 / tau
    
    if mask == BC_FLUID or mask == BC_INLET or mask == BC_OUTLET:
        for i in range(19):
            f_post_collision[i, x, y, z] = f_old[i, x, y, z] - omega * (f_old[i, x, y, z] - f_eq[i, x, y, z])
    else:
        # Solid -> Skip collision (or just copy)
        for i in range(19):
            f_post_collision[i, x, y, z] = f_old[i, x, y, z]

#######################################################################
#################### Streaming Operator: f_streamed ###################
#######################################################################
@wp.kernel
def streaming_kernel(f_post_collision: wp.array4d(dtype=float),
                     f_streamed: wp.array4d(dtype=float),
                     domain_mask: wp.array3d(dtype=int),
                     u_inlet: wp.array(dtype=float),         # Parameter: [ux_inlet]
                     rho_outlet: float,
                     nx: int, ny: int, nz: int):
    x, y, z = wp.tid()
    
    mask = domain_mask[x, y, z]
    
    # Skip solid nodes
    if mask == BC_SOLID:
        return

    # ZOU-HE VELOCITY INLET (x=0)
    if mask == BC_INLET:
        # Fetch dynamic ux from param array
        ux_in = u_inlet[0]
        
        # Known distributions
        f0 = f_post_collision[0, x, y, z]
        f2 = f_post_collision[2, x, y, z]   
        f3 = f_post_collision[3, x, y, z]
        f4 = f_post_collision[4, x, y, z]
        f5 = f_post_collision[5, x, y, z]
        f6 = f_post_collision[6, x, y, z]
        f8 = f_post_collision[8, x, y, z]   
        f10 = f_post_collision[10, x, y, z] 
        f12 = f_post_collision[12, x, y, z] 
        f14 = f_post_collision[14, x, y, z] 
        f15 = f_post_collision[15, x, y, z]
        f16 = f_post_collision[16, x, y, z]
        f17 = f_post_collision[17, x, y, z]
        f18 = f_post_collision[18, x, y, z]
        
        rho_new = (f0+f3+f4+f5+f6+f15+f16+f17+f18 + 2.0*(f2+f8+f10+f12+f14)) / (1.0 - ux_in)
        
        f1_new = f2 + (2.0/3.0) * rho_new * ux_in
        f7_new = f10 + (1.0/6.0) * rho_new * ux_in
        f9_new = f8 + (1.0/6.0) * rho_new * ux_in
        f11_new = f14 + (1.0/6.0) * rho_new * ux_in
        f13_new = f12 + (1.0/6.0) * rho_new * ux_in
        
        f_streamed[0, x, y, z]  = f0
        f_streamed[1, x, y, z]  = f1_new
        f_streamed[2, x, y, z]  = f2
        f_streamed[3, x, y, z]  = f3
        f_streamed[4, x, y, z]  = f4
        f_streamed[5, x, y, z]  = f5
        f_streamed[6, x, y, z]  = f6
        f_streamed[7, x, y, z]  = f7_new
        f_streamed[8, x, y, z]  = f8
        f_streamed[9, x, y, z]  = f9_new
        f_streamed[10, x, y, z] = f10
        f_streamed[11, x, y, z] = f11_new
        f_streamed[12, x, y, z] = f12
        f_streamed[13, x, y, z] = f13_new
        f_streamed[14, x, y, z] = f14
        f_streamed[15, x, y, z] = f15
        f_streamed[16, x, y, z] = f16
        f_streamed[17, x, y, z] = f17
        f_streamed[18, x, y, z] = f18
        return

    # EQUILIBRIUM PRESSURE OUTLET (x=nx-1)
    elif mask == BC_OUTLET:
        src_x = x - 1
        if src_x < 0: src_x = 0
        rho_sum = float(0.0)
        for i in range(19):
             f_val = f_post_collision[i, src_x, y, z]
             f_streamed[i, x, y, z] = f_val
             rho_sum += f_val
        if rho_sum > 1e-6:
             scale = rho_outlet / rho_sum
             for i in range(19):
                 f_streamed[i, x, y, z] *= scale
        return

    # Standard streaming with bounce-back for interior fluid
    for i in range(19):
        ex, ey, ez = E_CONST[i, 0], E_CONST[i, 1], E_CONST[i, 2]
        src_x, src_y, src_z = x - ex, y - ey, z - ez
        if src_x < 0 or src_x >= nx or src_y < 0 or src_y >= ny or src_z < 0 or src_z >= nz:
            f_streamed[i, x, y, z] = f_post_collision[OPPOSITE_CONST[i], x, y, z]
            continue
        neighbor_type = domain_mask[src_x, src_y, src_z]
        if neighbor_type == BC_SOLID:
            f_streamed[i, x, y, z] = f_post_collision[OPPOSITE_CONST[i], x, y, z]
        else:
            f_streamed[i, x, y, z] = f_post_collision[i, src_x, src_y, src_z]

#######################################################################
#################### Compute Stats: MaxV, KE ##########################
#######################################################################
@wp.kernel
def compute_stats_kernel(rho: wp.array3d(dtype=float),        # Input
                         u: wp.array3d(dtype=wp.vec3f),       # Input
                         domain_mask: wp.array3d(dtype=int),  # Input
                         stats: wp.array(dtype=float),        # Output buffer
                         nx: int, ny: int, nz: int):
    # Indices: [max_v, sum_v, sum_ke, count, sum_rho, sum_rho_in, sum_rho_out, sum_ux_out, c_in, c_out, c_outlet_nodes]
    x, y, z = wp.tid()
    mask = domain_mask[x, y, z]
    vel = u[x, y, z]
    val_rho = rho[x, y, z]
    v_sq = wp.dot(vel, vel)
    v_mag = wp.sqrt(v_sq)

    # 1. Global/Fluid Stats
    wp.atomic_add(stats, 2, v_sq)
    if mask != BC_SOLID:
        wp.atomic_max(stats, 0, v_mag)
        wp.atomic_add(stats, 1, v_mag)
        wp.atomic_add(stats, 3, 1.0)
        wp.atomic_add(stats, 4, val_rho)

    # 2. Hydraulic Stats (Specifically for Control Loop)
    # Using slices near boundaries for dP (x=1 and x=nx-2)
    # Using outlet (x=nx-1) for Q
    if x == 1:
        wp.atomic_add(stats, 5, val_rho)
        wp.atomic_add(stats, 8, 1.0) # Count Inlet Slice
    
    if x == nx - 2:
        wp.atomic_add(stats, 6, val_rho)
        wp.atomic_add(stats, 9, 1.0) # Count Outlet Slice
        
    if mask == BC_OUTLET:
        wp.atomic_add(stats, 7, vel.x) # Q (Sum of ux at outlet)
        wp.atomic_add(stats, 10, 1.0) # Count Outlet Nodes

