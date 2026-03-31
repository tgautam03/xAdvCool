"""
3D Geometry Designer with Interactive Sliders for Microchannel Cooling
Supported Designs: Empty Channel, Straight Channels, Pin-Fin Staggered
"""
import numpy as np
import pyvista as pv

# Boundary condition values
BC_FLUID = 0
BC_SOLID = 1
BC_INLET = 2
BC_OUTLET = 3

# === CONFIGURATION (edit these before running) ===
CORE_NX, CORE_NY, CORE_NZ = 128, 128, 32

# Current state
state = {
    'design_type': 0,     # 0=Empty, 1=Straight, 2=Pin-Fin
    'feature_size': 10.0, # Channel Width or Pin Diameter
    'spacing': 10.0,      # Wall Thickness or Pitch
    'buffer_size': 10,
    
    'clip_x': CORE_NX + 20, # Default no clip
    'clip_y': CORE_NY,
    'clip_z': CORE_NZ-1,
}

DESIGN_NAMES = ["Empty Channel", "Straight Channels", "Pin-Fin Staggered", "Gyroid TPMS", "Schwarz P TPMS", "Schwarz D TPMS"]

def draw_thick_line(grid, r0, c0, r1, c1, width):
    # ... (Keep existing implementation, but since I am replacing the block starting early, I need to be careful)
    # Actually, let's just target the DESIGN_NAMES line and the end of the file or use multiple edits.
    # The file is small enough.
    
    # Wait, replace_file_content requires contiguous block.
    # I will do it in 2 chunks.
    # Chunk 1: Update DESIGN_NAMES
    pass

# Start with chunk 1


def draw_thick_line(grid, r0, c0, r1, c1, width):
    """Draw a thick line on a 2D grid."""
    rr, cc = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing='ij')
    
    # Vector arithmetic to find distance from point to segment
    p0 = np.array([r0, c0])
    p1 = np.array([r1, c1])
    diff = p1 - p0
    l2 = np.sum(diff**2)
    
    if l2 == 0:
        dist = np.sqrt((rr - r0)**2 + (cc - c0)**2)
    else:
        # Projection t = dot(p - p0, p1 - p0) / |p1 - p0|^2
        t = ((rr - r0) * diff[0] + (cc - c0) * diff[1]) / l2
        t = np.clip(t, 0, 1)
        prox_r = r0 + t * diff[0]
        prox_c = c0 + t * diff[1]
        dist = np.sqrt((rr - prox_r)**2 + (cc - prox_c)**2)
        
    grid[dist <= width / 2.0] = True

    # ---------------------------
    # Processor Heat Map Utility
    # ---------------------------
def generate_dual_core_heatmap(nx, ny):
    """
    Generates a 2D normalized heat map (0.0 to 1.0) representing a Dual-Core Processor.
    Features two high-power CPU cores, an L3 cache strip, and background leakage.
    Pass the returned array as `heat_source[:, :, 0]` in run_simulation().
    """
    import scipy.ndimage as ndimage
    
    heat_map = np.zeros((nx, ny), dtype=np.float32)
    
    # 1. Background Leakage (Low Power)
    heat_map[:, :] = 0.1
    
    # 2. Define Regions (Side-by-side in Y, perpendicular to flow direction X)
    # Core 1: Top half (High Power)
    c1_x0, c1_x1 = int(nx * 0.25), int(nx * 0.75)
    c1_y0, c1_y1 = int(ny * 0.55), int(ny * 0.85)
    heat_map[c1_x0:c1_x1, c1_y0:c1_y1] = 1.0
    
    # Core 2: Bottom half (High Power)
    c2_x0, c2_x1 = int(nx * 0.25), int(nx * 0.75)
    c2_y0, c2_y1 = int(ny * 0.15), int(ny * 0.45)
    heat_map[c2_x0:c2_x1, c2_y0:c2_y1] = 1.0
    
    # L3 Cache: Center Strip between cores (Medium Power)
    l3_x0, l3_x1 = int(nx * 0.30), int(nx * 0.70)
    l3_y0, l3_y1 = int(ny * 0.45), int(ny * 0.55)
    heat_map[l3_x0:l3_x1, l3_y0:l3_y1] = 0.5
    
    # 3. Simulate Thermal Spreading (Gaussian Blur)
    sigma = nx / 80.0  # Approx 3.0 for nx=256
    heat_map = ndimage.gaussian_filter(heat_map, sigma=sigma)
    
    # Normalize
    if heat_map.max() > 0:
        heat_map /= heat_map.max()
        
    return heat_map


def generate_processor_heatmap(nx, ny):
    """Alias for generate_dual_core_heatmap — kept for backward compatibility."""
    return generate_dual_core_heatmap(nx, ny)


def generate_quad_core_heatmap(nx, ny):
    """2x2 quad-core CPU with shared L3 cache cross and background leakage."""
    import scipy.ndimage as ndimage

    hm = np.full((nx, ny), 0.08, dtype=np.float32)

    # Four cores in a 2x2 grid
    cores = [
        (0.15, 0.55, 0.45, 0.85),   # top-right
        (0.15, 0.15, 0.45, 0.45),   # top-left
        (0.55, 0.55, 0.85, 0.85),   # bottom-right
        (0.55, 0.15, 0.85, 0.45),   # bottom-left
    ]
    for x0f, y0f, x1f, y1f in cores:
        hm[int(nx*x0f):int(nx*x1f), int(ny*y0f):int(ny*y1f)] = 1.0

    # L3 cache cross between cores
    hm[int(nx*0.45):int(nx*0.55), int(ny*0.10):int(ny*0.90)] = 0.45
    hm[int(nx*0.10):int(nx*0.90), int(ny*0.45):int(ny*0.55)] = 0.45

    hm = ndimage.gaussian_filter(hm, sigma=nx / 80.0)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def generate_gpu_die_heatmap(nx, ny):
    """GPU die: large shader array block with HBM controller strips on two edges."""
    import scipy.ndimage as ndimage

    hm = np.full((nx, ny), 0.05, dtype=np.float32)

    # Shader / compute array — large central block
    hm[int(nx*0.15):int(nx*0.85), int(ny*0.15):int(ny*0.85)] = 0.75

    # HBM memory controller strips on left and right edges
    hm[int(nx*0.10):int(nx*0.90), int(ny*0.02):int(ny*0.12)] = 1.0
    hm[int(nx*0.10):int(nx*0.90), int(ny*0.88):int(ny*0.98)] = 1.0

    # Raster engine hotspots (small intense regions inside shader block)
    for cx_f in [0.30, 0.50, 0.70]:
        cx, cy = int(nx * cx_f), int(ny * 0.50)
        r = int(min(nx, ny) * 0.04)
        xx, yy = np.ogrid[:nx, :ny]
        dist2 = (xx - cx)**2 + (yy - cy)**2
        hm[dist2 < r**2] = 1.0

    hm = ndimage.gaussian_filter(hm, sigma=nx / 60.0)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def generate_chiplet_heatmap(nx, ny):
    """Multi-chiplet package: 4-6 discrete compute dies on a substrate with an I/O die."""
    import scipy.ndimage as ndimage

    hm = np.full((nx, ny), 0.03, dtype=np.float32)

    # Compute chiplets — 2x3 grid of small high-power dies
    chiplet_w, chiplet_h = 0.18, 0.22
    positions = [
        (0.12, 0.08), (0.12, 0.40), (0.12, 0.72),
        (0.58, 0.08), (0.58, 0.40), (0.58, 0.72),
    ]
    for x0f, y0f in positions:
        hm[int(nx*x0f):int(nx*(x0f+chiplet_w)),
           int(ny*y0f):int(ny*(y0f+chiplet_h))] = 1.0

    # I/O die — lower-power strip along center
    hm[int(nx*0.38):int(nx*0.52), int(ny*0.10):int(ny*0.90)] = 0.35

    hm = ndimage.gaussian_filter(hm, sigma=nx / 70.0)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def generate_igbt_heatmap(nx, ny):
    """IGBT half-bridge module: 3 IGBT dies interleaved with 3 diode dies in a row."""
    import scipy.ndimage as ndimage

    hm = np.full((nx, ny), 0.02, dtype=np.float32)

    # 3 IGBT + 3 diode pairs arranged linearly along Y
    die_h = 0.10
    igbt_w = 0.50
    diode_w = 0.25
    y_positions = [0.05, 0.20, 0.38, 0.55, 0.72, 0.88]

    for i, y0f in enumerate(y_positions):
        if i % 2 == 0:
            # IGBT die (high power)
            hm[int(nx*0.20):int(nx*(0.20+igbt_w)),
               int(ny*y0f):int(ny*(y0f+die_h))] = 1.0
        else:
            # Diode die (medium power)
            hm[int(nx*0.30):int(nx*(0.30+diode_w)),
               int(ny*y0f):int(ny*(y0f+die_h))] = 0.55

    hm = ndimage.gaussian_filter(hm, sigma=nx / 60.0)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def generate_soc_heatmap(nx, ny):
    """Heterogeneous SoC: big.LITTLE CPU clusters, GPU block, memory controller, NPU."""
    import scipy.ndimage as ndimage

    hm = np.full((nx, ny), 0.06, dtype=np.float32)

    # Big cores (2 high-power) — top-left
    hm[int(nx*0.10):int(nx*0.35), int(ny*0.60):int(ny*0.90)] = 1.0

    # Little cores (4 small low-power) — top-right
    for dy in [0.08, 0.22]:
        for dx in [0.10, 0.28]:
            hm[int(nx*(dx)):int(nx*(dx+0.12)),
               int(ny*(0.35+dy)):int(ny*(0.35+dy+0.10))] = 0.30

    # GPU block — bottom half, large area, moderate power
    hm[int(nx*0.50):int(nx*0.90), int(ny*0.35):int(ny*0.90)] = 0.65

    # Memory controller strip — left edge
    hm[int(nx*0.05):int(nx*0.95), int(ny*0.02):int(ny*0.12)] = 0.40

    # NPU / AI accelerator — small high-power block
    hm[int(nx*0.45):int(nx*0.60), int(ny*0.15):int(ny*0.30)] = 0.80

    hm = ndimage.gaussian_filter(hm, sigma=nx / 70.0)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def generate_gaussian_hotspots_heatmap(nx, ny, seed=0):
    """1-3 smooth Gaussian hotspots on a low background — analytical benchmark pattern."""
    rng = np.random.RandomState(seed)
    hm = np.full((nx, ny), 0.05, dtype=np.float32)

    n_spots = rng.randint(1, 4)
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    for _ in range(n_spots):
        cx = rng.uniform(0.2, 0.8) * nx
        cy = rng.uniform(0.2, 0.8) * ny
        sigma = rng.uniform(0.06, 0.15) * min(nx, ny)
        amplitude = rng.uniform(0.5, 1.0)
        hm += amplitude * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))

    if hm.max() > 0:
        hm /= hm.max()
    return hm


def generate_hotspot_on_background_heatmap(nx, ny):
    """Uniform low-power background with 1-2 small intense hotspots (ALU/FPU style)."""
    import scipy.ndimage as ndimage

    hm = np.full((nx, ny), 0.20, dtype=np.float32)

    # Hotspot 1 — small intense region
    cx1, cy1 = int(nx * 0.40), int(ny * 0.60)
    r1 = int(min(nx, ny) * 0.07)
    xx, yy = np.ogrid[:nx, :ny]
    hm[(xx - cx1)**2 + (yy - cy1)**2 < r1**2] = 1.0

    # Hotspot 2 — slightly weaker
    cx2, cy2 = int(nx * 0.65), int(ny * 0.35)
    r2 = int(min(nx, ny) * 0.05)
    hm[(xx - cx2)**2 + (yy - cy2)**2 < r2**2] = 0.85

    hm = ndimage.gaussian_filter(hm, sigma=nx / 100.0)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


def generate_peripheral_ring_heatmap(nx, ny):
    """High power along perimeter (I/O pads, SerDes), cool interior — FPGA-style."""
    import scipy.ndimage as ndimage

    hm = np.full((nx, ny), 0.05, dtype=np.float32)

    # Ring of I/O along all four edges
    ring_w = 0.10
    hm[:int(nx*ring_w), :] = 0.85          # top edge
    hm[int(nx*(1-ring_w)):, :] = 0.85      # bottom edge
    hm[:, :int(ny*ring_w)] = 0.85          # left edge
    hm[:, int(ny*(1-ring_w)):] = 0.85      # right edge

    # SerDes transceivers — high-power corners
    corner_r = int(min(nx, ny) * 0.12)
    xx, yy = np.ogrid[:nx, :ny]
    for cx_f, cy_f in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]:
        cx, cy = int(nx * cx_f), int(ny * cy_f)
        hm[(xx - cx)**2 + (yy - cy)**2 < corner_r**2] = 1.0

    # Cool FPGA fabric interior
    hm[int(nx*0.20):int(nx*0.80), int(ny*0.20):int(ny*0.80)] = 0.15

    hm = ndimage.gaussian_filter(hm, sigma=nx / 60.0)
    if hm.max() > 0:
        hm /= hm.max()
    return hm


# Registry of all heat source generators
HEAT_SOURCE_REGISTRY = {
    "uniform":              lambda nx, ny, **kw: np.ones((nx, ny), dtype=np.float32),
    "dual_core":            lambda nx, ny, **kw: generate_dual_core_heatmap(nx, ny),
    "quad_core":            lambda nx, ny, **kw: generate_quad_core_heatmap(nx, ny),
    "gpu_die":              lambda nx, ny, **kw: generate_gpu_die_heatmap(nx, ny),
    "chiplet":              lambda nx, ny, **kw: generate_chiplet_heatmap(nx, ny),
    "igbt_half_bridge":     lambda nx, ny, **kw: generate_igbt_heatmap(nx, ny),
    "soc_heterogeneous":    lambda nx, ny, **kw: generate_soc_heatmap(nx, ny),
    "gaussian_hotspots":    lambda nx, ny, **kw: generate_gaussian_hotspots_heatmap(nx, ny, seed=kw.get("seed", 0)),
    "hotspot_on_background": lambda nx, ny, **kw: generate_hotspot_on_background_heatmap(nx, ny),
    "peripheral_ring":      lambda nx, ny, **kw: generate_peripheral_ring_heatmap(nx, ny),
}

HEAT_SOURCE_NAMES = list(HEAT_SOURCE_REGISTRY.keys())


def generate_mask(state):
    """Generate the full mask with current state."""
    # Fixed seed for reproducibility in benchmark generation
    np.random.seed(42)
    
    nx, ny, nz = CORE_NX, CORE_NY, CORE_NZ
    design_idx = int(state['design_type'])
    if design_idx < 0: design_idx = 0
    if design_idx >= len(DESIGN_NAMES): design_idx = len(DESIGN_NAMES) - 1
    
    # Create base fluid domain
    buf = int(state['buffer_size'])
    total_nx = nx + 2 * buf
    mask = np.full((total_nx, ny, nz), BC_FLUID, dtype=np.int32)
    
    # Temporary solid mask for the core (to be carved)
    # Applied to mask[buf:buf+nx, :, :]
    is_solid = np.zeros((nx, ny, nz), dtype=bool)
    
    # ---------------------------
    # UNIFIED PARAMETERS (Re-mapped)
    # ---------------------------
    # Slider 1: Channel Width (Px) - Range ~ 2 to 20 px
    # state['feature_size'] is 1.0 to 10.0
    raw_width = state.get('feature_size', 3.0)
    channel_width = max(2, int(raw_width * 2.0))
    
    # Slider 2: Channel Count (N) - Range ~ 4 to 40 channels
    # state['spacing'] is 1.0 to 10.0
    raw_count = state.get('spacing', 3.0)
    target_channels = max(4, int(raw_count * 4.0))
    
    # DYNAMIC CONSTRAINT:
    # N * W + (N+1) * MinGap <= Ly
    # MinGap = 1.0
    # Max N = (Ly - 1) / (W + 1)
    max_n_possible = int((ny - 1) / (channel_width + 1))
    
    # Clamp N to be physically possible
    n_channels = min(target_channels, max_n_possible)
    n_channels = max(2, n_channels) # At least 2 channels

    if design_idx == 0: # Empty Channel
        # No solids inside
        pass 
        
    elif design_idx == 1: # Straight Channels
        # Calculate available space
        total_channel_width = n_channels * channel_width
        
        if total_channel_width >= ny:
            # Saturated (Channels wider than domain)
            pass # Leave as empty (all fluid) because is_solid initialized to False? 
                 # Wait, is_solid is initialized to False (lines 53/63)? 
                 # Actually `create_geometry` returns `is_solid`. 
                 # Let's assume is_solid is False by default?
                 # Looking at line 82: `is_solid = np.zeros((nx, ny, nz), dtype=bool)` -> All Fluid.
                 # So if saturated, we just do nothing (leave as fluid).
            pass
        else:
            # Calculate gap (Wall Thickness)
            # Gap * (N+1) + Channel * N = Ly
            # Gap = (Ly - Channel*N) / (N+1)
            total_gap_space = ny - total_channel_width
            gap = float(total_gap_space) / (n_channels + 1)
            
            # We need to set WALLS to True.
            # Initialize all to SOLID first? No, default is Fluid.
            # Let's simple set the WALL regions to True.
            
            current_y = 0.0
            
            # Draw N+1 Walls? 
            # Pattern: Wall - Channel - Wall - Channel ... - Wall
            
            # Wall 0
            y_start = 0
            y_end = int(gap)
            if y_end > 0: is_solid[:, :y_end, :] = True
            current_y = gap
            
            for i in range(n_channels):
                # Channel i (Skip, remains False/Fluid)
                current_y += channel_width
                
                # Wall i+1
                y_start = int(current_y)
                y_end = int(current_y + gap)
                # Correction for last wall to fill to end
                if i == n_channels - 1:
                    y_end = ny
                
                if y_start < ny:
                    is_solid[:, y_start:y_end, :] = True
                
                current_y += gap

    elif design_idx == 2: # Pin-Fin Staggered
        # Legacy mapping for Pin-Fin (keep distinct behavior)
        d_pin = max(2, int(raw_width * 2.0)) # Use same scale as channel width
        pitch = max(d_pin + 2, int(raw_count * 3.0)) # Approximate spacing scale
        radius = d_pin / 2.0
        
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Pattern A centers
        dx_a = (X % pitch) - pitch/2
        dy_a = (Y % pitch) - pitch/2
        dist_a = dx_a**2 + dy_a**2
        
        # Pattern B centers
        dx_b = ((X - pitch/2) % pitch) - pitch/2
        dy_b = ((Y - pitch/2) % pitch) - pitch/2
        dist_b = dx_b**2 + dy_b**2
        
        is_solid_2d = np.minimum(dist_a, dist_b) < radius**2
        is_solid[:, :, :] = is_solid_2d[:, :, np.newaxis]

    elif design_idx in [3, 4, 5]: # TPMS Structures
        # TPMS Parameter Mapping:
        # feature_size (Slider 1) controls Porosity (Isovalue C)
        # spacing (Slider 2) controls Unit Cell Size (Frequency)
        
        # 1. Porosity (Isovalue C)
        # raw_width 1.0 -> mostly solid (C = 0.8)
        # raw_width 10.0 -> mostly fluid (C = -0.8)
        # (Assuming equation < C is solid)
        C = 0.8 - ((raw_width - 1.0) / 9.0) * 1.6
        
        # 2. Unit Cell Size
        # raw_count 1.0 -> small dense cells (20px)
        # raw_count 10.0 -> large open cells (100px)
        # Note: raw_count is derived from state['spacing']
        cell_size = 20.0 + ((raw_count - 1.0) / 9.0) * 80.0
        
        # Angular frequency
        omega = 2.0 * np.pi / cell_size
        
        # 3. Evaluate 3D Equation
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        wx, wy, wz = X * omega, Y * omega, Z * omega
        
        if design_idx == 3: # Gyroid
            eq = np.sin(wx)*np.cos(wy) + np.sin(wy)*np.cos(wz) + np.sin(wz)*np.cos(wx)
        elif design_idx == 4: # Schwarz P
            eq = np.cos(wx) + np.cos(wy) + np.cos(wz)
        elif design_idx == 5: # Schwarz D
            eq = (np.sin(wx)*np.sin(wy)*np.sin(wz) + 
                  np.sin(wx)*np.cos(wy)*np.cos(wz) + 
                  np.cos(wx)*np.sin(wy)*np.cos(wz) + 
                  np.cos(wx)*np.cos(wy)*np.sin(wz))
        
        # 4. Define Solids
        is_solid[:, :, :] = eq < C
        
        # 5. Open Inlet/Outlet Manifolds
        # Carve out 5 pixels of pure fluid at the entry and exit to distribute flow evenly into the complex 3D TPMS structure
        is_solid[:5, :, :] = False
        is_solid[-5:, :, :] = False



    # Apply solids to mask
    mask[buf:buf+nx, :, :][is_solid] = BC_SOLID
    
    # Walls (Top/Bottom Y and Z)
    # Y-Walls (Side walls)
    mask[:, 0, :] = BC_SOLID
    mask[:, -1, :] = BC_SOLID
    
    # Z-Walls (Floor/Ceiling)
    base_thickness = 4
    top_thickness = 1
    mask[:, :, :base_thickness] = BC_SOLID
    mask[:, :, -top_thickness:] = BC_SOLID
    
    # Inlet/Outlet (X faces) - Inside the fluid channel
    # Prevent overwriting side walls and Z-walls
    mask[0, 1:-1, base_thickness:-top_thickness] = BC_INLET
    mask[-1, 1:-1, base_thickness:-top_thickness] = BC_OUTLET
    
    return mask

def create_meshes(mask):
    """Create PyVista meshes from mask."""
    # Use spacing=(1,1,1) for simplicity
    grid = pv.ImageData(dimensions=np.array(mask.shape)+1)
    grid.cell_data['mask'] = mask.flatten(order='F')
    
    solid = grid.threshold([0.9, 1.1], scalars='mask', preference='cell')
    inlet = grid.threshold([1.9, 2.1], scalars='mask', preference='cell')
    outlet = grid.threshold([2.9, 3.1], scalars='mask', preference='cell')

    return grid, solid, inlet, outlet


# ... (Previous code remains the same up to generate_mask and create_meshes) ...

if __name__ == "__main__":
    # Generate initial mask
    mask = generate_mask(state)
    grid, solid, inlet, outlet = create_meshes(mask)

    # Create plotter
    pl = pv.Plotter()
    actors = {}

    def refresh_view():
        """Refresh the 3D view with current state."""
        global mask, grid, solid, inlet, outlet
        
        mask = generate_mask(state)
        grid, solid, inlet, outlet = create_meshes(mask)
        
        # Remove old actors
        for key in list(actors.keys()):
            try:
                pl.remove_actor(actors[key])
            except:
                pass
            del actors[key]
        
        # Clip logic
        total_nx = CORE_NX + 2 * int(state['buffer_size'])
        clipped = solid
        
        # Robust Clipping
        if solid.n_cells > 0:
            # Clip X
            if state['clip_x'] < total_nx:
                 clipped = clipped.clip(normal='x', origin=(state['clip_x'], 0, 0), invert=True)
            # Clip Y
            if state['clip_y'] < CORE_NY and clipped.n_cells > 0:
                 clipped = clipped.clip(normal='y', origin=(0, state['clip_y'], 0), invert=True)
            # Clip Z
            if state['clip_z'] < CORE_NZ and clipped.n_cells > 0:
                 clipped = clipped.clip(normal='z', origin=(0, 0, state['clip_z']), invert=True)
        
        # Add meshes
        if clipped.n_cells > 0:
            actors['solid'] = pl.add_mesh(clipped, color='peru', opacity=1.0, show_edges=False, smooth_shading=True)
        if inlet.n_cells > 0:
            actors['inlet'] = pl.add_mesh(inlet, color='blue', opacity=0.6)
        if outlet.n_cells > 0:
            actors['outlet'] = pl.add_mesh(outlet, color='red', opacity=0.6)

        actors['outline'] = pl.add_mesh(grid.outline(), color='black', line_width=2)
        
        # Add Heat Map Overlay
        nx_m, ny_m, nz_m = mask.shape
        hm = generate_processor_heatmap(nx_m, ny_m)
        
        # Create a floor grid
        # Z-offset -2 to be below everything
        floor_grid = pv.ImageData(dimensions=(nx_m+1, ny_m+1, 2), spacing=(1,1,1), origin=(0, 0, -5))
        # We only care about the top surface of this floor
        # Actually a structured grid or just a plane.
        # Let's use the same ImageData trick but thin
        
        # Better: Just a slice.
        # Create a grid just for the heatmap
        hm_grid = pv.ImageData(dimensions=(nx_m, ny_m, 1), spacing=(1,1,1), origin=(0, 0, -2))
        hm_grid.point_data['heat'] = hm.flatten(order='F') # Point data for smooth interpolation
        
        actors['heatmap'] = pl.add_mesh(hm_grid, scalars='heat', cmap='inferno', 
                                        opacity=0.5, show_scalar_bar=False,
                                        clim=[0.0, 1.0])
        # actors['heatmap_label'] = pl.add_point_labels([hm_grid.center], ["Processor Heat Map"], 
        #                                               point_size=0, font_size=10, always_visible=True)
        
        # Update Title
        idx = int(state['design_type'])
        if idx < 0: idx = 0
        if idx >= len(DESIGN_NAMES): idx = len(DESIGN_NAMES) - 1
        t_name = DESIGN_NAMES[idx]
        pl.add_text(f"Geometry Designer: {t_name}", position='upper_left', font_size=10, name='title')

        # Calculate Porosity (Active Region)
        buf = int(state['buffer_size'])
        nx_core = CORE_NX
        # Ensure bounds
        if buf + nx_core <= mask.shape[0]:
            active_region = mask[buf:buf+nx_core, :, :]
            # Fluid is 0, Inlet/Outlet is 2/3 (which are also fluid-like but technically boundaries)
            # Usually we consider available fluid volume. 
            # BC_FLUID=0, BC_INLET=2, BC_OUTLET=3 are all fluid. BC_SOLID=1 is solid.
            # So Porosity = (Total - Solid) / Total
            n_solid = np.sum(active_region == BC_SOLID)
            total_cells = active_region.size
            if total_cells > 0:
                porosity = 1.0 - (n_solid / total_cells)
            else:
                porosity = 0.0
            
            pl.add_text(f"Active Porosity: {porosity:.1%}", position='upper_right', font_size=10, name='porosity')

    # Slider callbacks
    def cb_type(v): 
        # Clamp value safely
        idx = int(v + 0.5)
        if idx < 0: idx = 0
        if idx >= len(DESIGN_NAMES): idx = len(DESIGN_NAMES) - 1
        state['design_type'] = idx
        refresh_view()

    def cb_feat(v): state['feature_size'] = v; refresh_view()
    def cb_space(v): state['spacing'] = v; refresh_view()
    def cb_buf(v): state['buffer_size'] = int(v); refresh_view()

    def cb_cx(v): state['clip_x'] = int(v); refresh_view()
    def cb_cy(v): state['clip_y'] = int(v); refresh_view()
    def cb_cz(v): state['clip_z'] = int(v); refresh_view()

    # UI Layout
    min_dim = min(CORE_NX, CORE_NY, CORE_NZ)
    MAX_BUFFER = 20

    # TOP: Generation
    # Range 0 to 4.49
    # Define Wrappers (Capture state)
    # Define Wrappers (Capture state)
    def update_ui_labels():
        idx = int(state['design_type'])
        
        # Calculate raw values from sliders
        raw_width = state['feature_size']
        raw_count = state['spacing']
        
        test_w = max(2, int(raw_width * 2.0))
        test_n = max(4, int(raw_count * 4.0))
        
        # Domain Constraint
        H = CORE_NY
        max_n = int((H - 1) / (test_w + 1))
        effective_n = min(test_n, max_n)
        
        try:
            if idx == 1 or idx == 3 or idx == 4:
                s_feat.GetRepresentation().SetTitleText(f"Chan Width: {test_w} px")
                if test_n > max_n:
                     s_space.GetRepresentation().SetTitleText(f"Count: {effective_n} (Clamped from {test_n})")
                else:
                     s_space.GetRepresentation().SetTitleText(f"Chan Count: {test_n} (Max {max_n})")

            else:
                s_feat.GetRepresentation().SetTitleText("Feature Size")
                s_space.GetRepresentation().SetTitleText("Spacing")
        except NameError:
            # Sliders not yet initialized, safe to ignore
            pass
        except Exception as e:
            print(f"Warning updating labels: {e}")

    def cb_feat_wrapper(v):
        state['feature_size'] = v
        update_ui_labels()
        refresh_view()
        
    def cb_space_wrapper(v):
        state['spacing'] = v
        update_ui_labels()
        refresh_view()

    def cb_type_wrapper(v):
        cb_type(v)
        update_ui_labels()

    # Re-add sliders in order
    pl.clear_slider_widgets()
    
    # 1. Create Feature/Space sliders FIRST (so variables exist)
    s_feat = pl.add_slider_widget(cb_feat_wrapper, [1.0, 10.0], value=state['feature_size'], title="Feature Size",
                         pointa=(0.18, 0.92), pointb=(0.35, 0.92), title_height=0.015, fmt="%.1f")

    s_space = pl.add_slider_widget(cb_space_wrapper, [1.0, 10.0], value=state['spacing'], title="Spacing",
                         pointa=(0.38, 0.92), pointb=(0.55, 0.92), title_height=0.015, fmt="%.1f")
    
    # 2. Create Type slider LAST
    max_type_idx = len(DESIGN_NAMES) - 0.51
    pl.add_slider_widget(cb_type_wrapper, [0, max_type_idx], value=state['design_type'], title="Design Type", 
                         pointa=(0.02, 0.92), pointb=(0.15, 0.92), title_height=0.015, fmt="%.0f")
    
    # 3. Initial Label Update
    update_ui_labels()
                         
    pl.add_slider_widget(cb_buf, [5, MAX_BUFFER], value=state['buffer_size'], title="Buffer",
                         pointa=(0.58, 0.92), pointb=(0.70, 0.92), title_height=0.015, fmt="%.0f")


    # BOTTOM: Clipping
    total_x_max = CORE_NX + 2 * MAX_BUFFER
    pl.add_slider_widget(cb_cx, [0, total_x_max], value=state['clip_x'], title="ClipX",
                         pointa=(0.20, 0.08), pointb=(0.40, 0.08), title_height=0.015, fmt="%.0f")
    pl.add_slider_widget(cb_cy, [0, CORE_NY], value=state['clip_y'], title="ClipY",
                         pointa=(0.45, 0.08), pointb=(0.65, 0.08), title_height=0.015, fmt="%.0f")
    pl.add_slider_widget(cb_cz, [0, CORE_NZ], value=state['clip_z'], title="ClipZ",
                         pointa=(0.70, 0.08), pointb=(0.90, 0.08), title_height=0.015, fmt="%.0f")

    # Save function
    def save_mask():
        np.save("designed_mask.npy", mask)
        idx = int(state['design_type'])
        if idx < 0: idx = 0
        if idx >= len(DESIGN_NAMES): idx = len(DESIGN_NAMES) - 1
        t_name = DESIGN_NAMES[idx]
        print(f"\n✓ Saved designed_mask.npy | Shape: {mask.shape}")
        print(f"  Type: {t_name} | Feat: {state['feature_size']:.1f} | Space: {state['spacing']:.1f}")

    pl.add_key_event('s', save_mask)
    pl.add_text("Press 'S' to SAVE\nAdjust Sliders to change Geometry", position='lower_right', font_size=10)

    # Initial render
    refresh_view()
    pl.camera_position = 'iso'
    pl.show()
