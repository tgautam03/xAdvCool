"""
Combined Visualization
Launches two independent PyVista windows using multiprocessing:
1. Velocity Field (Slices)
2. Thermal Field (Volume + Slices)
"""
import multiprocessing
import numpy as np
import pyvista as pv
import os

def viz_velocity():
    """Velocity Field Visualization"""
    # Re-import inside process to avoid context issues
    import numpy as np
    import pyvista as pv
    import os
    
    import geometry
    
    # Load Data
    if not os.path.exists("designed_mask.npy"): return
    mask = np.load("designed_mask.npy")
    nx, ny, nz = mask.shape
    
    # Generate Heatmap for context
    try:
        heatmap = geometry.generate_processor_heatmap(nx, ny)
        # Create grid
        hm_grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(1,1,1), origin=(0, 0, -2))
        hm_grid.point_data['heat'] = heatmap.flatten(order='F')
    except Exception as e:
        print(f"Warning: Could not generate heatmap overlay: {e}")
        hm_grid = None
    
    import glob
    vel_files = sorted(glob.glob("results/time_stamps/velocity_*.npy"))
    if not vel_files:
        result_file = "results/final_velocity.npy"
        if os.path.exists(result_file):
            vel_files = [result_file]
        else:
            print(f"Error: {result_file} not found.")
            return

    print(f"[Velocity] Loaded {len(vel_files)} time steps. Loading {vel_files[-1]}...")
    u = np.load(vel_files[-1])
    v_mag = np.linalg.norm(u, axis=3)
    
    # Mask solids
    solidv = (mask == 1)
    v_mag[solidv] = np.nan
    
    # Calculate v_max excluding buffer zones
    # Find x-slices that contain internal solids (not just walls at y/z limits)
    internal_solids = np.any(mask[:, 1:-1, 1:-1] == 1, axis=(1, 2))
    core_indices = np.where(internal_solids)[0]
    if len(core_indices) > 0:
        x_start, x_end = core_indices[0], core_indices[-1]
        v_max = np.nanmax(v_mag[x_start:x_end+1, :, :])
    else:
        v_max = np.nanmax(v_mag)
    
    if np.isnan(v_max) or v_max == 0: v_max = 0.05 # Fallback
    
    # Setup Mesh
    grid = pv.ImageData(dimensions=(nx+1, ny+1, nz+1))
    grid.cell_data['velocity'] = v_mag.flatten(order='F')
    grid.cell_data['mask'] = mask.flatten(order='F')
    mesh = grid.cell_data_to_point_data()
    
    # Actors
    solids = mesh.threshold([0.9, 1.1], scalars='mask')
    
    # Plotter
    pl = pv.Plotter()
    pl.enable_depth_peeling()
    pl.add_text("Velocity Field (Volumetric + Slices)", font_size=12)
    
    # Add Heatmap if available
    if hm_grid:
         pl.add_mesh(hm_grid, scalars='heat', cmap='inferno', 
                     opacity=0.3, show_scalar_bar=False, clim=[0, 1])
         pl.add_text("Base: Heat Source Map", position=(10, 100), font_size=8, color='orange')
    
    # Static Actors
    solid_actor = pl.add_mesh(solids, color='tan', opacity=0.3)
    pl.add_mesh(grid.outline(), color='black')
    
    # Volumetric Rendering
    opacity = [0, 0.0, 0.1, 0.3, 0.6]
    vol_actor = pl.add_volume(grid, scalars='velocity', cmap='coolwarm', opacity=opacity, show_scalar_bar=True, clim=[0, v_max])
    
    # Dynamic State
    state = {'z': nz//2, 'y': ny//2, 't': len(vel_files) - 1, 'vol_vis': True}
    actors = {}

    def update_t():
        if not vel_files: return
        t_idx = state['t']
        u_t = np.load(vel_files[t_idx])
        v_mag_t = np.linalg.norm(u_t, axis=3)
        v_mag_t[solidv] = np.nan
        grid.cell_data['velocity'] = v_mag_t.flatten(order='F')
        
        # Rebuild point data so slices pick it up
        new_mesh = grid.cell_data_to_point_data()
        mesh.point_data['velocity'] = new_mesh.point_data['velocity']
        
        nonlocal vol_actor
        try:
            pl.remove_actor(vol_actor)
        except Exception:
            pass
        vol_actor = pl.add_volume(grid, scalars='velocity', cmap='coolwarm', opacity=opacity, show_scalar_bar=False, clim=[0, v_max])
        vol_actor.SetVisibility(state['vol_vis'])
        
        update()
    
    def update():
        for a in list(actors.keys()):
            try: pl.remove_actor(actors[a])
            except: pass
            del actors[a]
            
        # Horizontal Slice
        h_slice = mesh.slice(normal='z', origin=(nx//2, ny//2, state['z']))
        if h_slice.n_points > 0:
            actors['v_h'] = pl.add_mesh(h_slice, scalars='velocity', cmap='coolwarm', 
                                        clim=[0, v_max], nan_color='lightgray', show_scalar_bar=False)
                                        
        # Vertical Slice
        v_slice = mesh.slice(normal='y', origin=(nx//2, state['y'], nz//2))
        if v_slice.n_points > 0:
            actors['v_v'] = pl.add_mesh(v_slice, scalars='velocity', cmap='coolwarm',
                                        clim=[0, v_max], nan_color='lightgray', show_scalar_bar=False)
    
    def cb_z(v): state['z'] = int(v); update()
    def cb_y(v): state['y'] = int(v); update()
    def cb_t(v): state['t'] = int(v); update_t()
    
    pl.add_slider_widget(cb_z, [0, nz], value=state['z'], title="Z Slice", pointa=(0.05, 0.9), pointb=(0.35, 0.9), fmt="%.0f")
    pl.add_slider_widget(cb_y, [0, ny], value=state['y'], title="Y Slice", pointa=(0.4, 0.9), pointb=(0.7, 0.9), fmt="%.0f")
    if len(vel_files) > 1:
        pl.add_slider_widget(cb_t, [0, len(vel_files)-1], value=state['t'], title="Time Step", pointa=(0.75, 0.9), pointb=(0.95, 0.9), fmt="%.0f")
    
    def cb_solid_toggle(flag):
        solid_actor.SetVisibility(flag)
    
    pl.add_checkbox_button_widget(cb_solid_toggle, value=True, color_on='tan', color_off='grey',
                                  position=(10, 10), size=40, border_size=2)
    pl.add_text("Solids", position=(60, 20), font_size=10)

    def cb_volume_toggle(flag):
        state['vol_vis'] = flag
        if vol_actor:
            vol_actor.SetVisibility(flag)

    pl.add_checkbox_button_widget(cb_volume_toggle, value=True, color_on='orange', color_off='grey',
                                  position=(10, 60), size=40, border_size=2)
    pl.add_text("Volume", position=(60, 70), font_size=10)
    
    pl.camera_position = 'iso'
    update()
    pl.show()

def viz_thermal():
    """Thermal Field Visualization"""
    import numpy as np
    import pyvista as pv
    import os
    
    import geometry

    # Load Data
    if not os.path.exists("designed_mask.npy"): return
    mask = np.load("designed_mask.npy")
    nx, ny, nz = mask.shape
    
    # Generate Heatmap for context
    try:
        heatmap = geometry.generate_processor_heatmap(nx, ny)
        # Create grid
        hm_grid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(1,1,1), origin=(0, 0, -2))
        hm_grid.point_data['heat'] = heatmap.flatten(order='F')
    except Exception as e:
        print(f"Warning: Could not generate heatmap overlay: {e}")
        hm_grid = None
    
    import glob
    therm_files = sorted(glob.glob("results/time_stamps/temperature_*.npy"))
    if not therm_files:
        thermal_file = "results/final_temperature.npy"
        if os.path.exists(thermal_file):
            therm_files = [thermal_file]
        else:
            print(f"Error: {thermal_file} not found.")
            return
    
    print(f"[Thermal] Loaded {len(therm_files)} time steps. Loading {therm_files[-1]}...")
    T_field = np.load(therm_files[-1])
    t_max = np.max(T_field)
    
    # Setup Mesh
    grid = pv.ImageData(dimensions=(nx+1, ny+1, nz+1))
    grid.cell_data['temperature'] = T_field.flatten(order='F')
    grid.cell_data['mask'] = mask.flatten(order='F')
    mesh = grid.cell_data_to_point_data()
    
    # Actors
    solids = mesh.threshold([0.9, 1.1], scalars='mask')
    
    # Plotter
    pl = pv.Plotter()
    pl.enable_depth_peeling()
    pl.add_text("Temperature Field (Volumetric + Slices)", font_size=12)
    
    # Add Heatmap if available
    if hm_grid:
         # Use 'Oranges' to distinguish slightly from 'inferno' or just consistent
         pl.add_mesh(hm_grid, scalars='heat', cmap='Oranges', 
                     opacity=0.3, show_scalar_bar=False, clim=[0, 1])
         pl.add_text("Base: Heat Source Map", position=(10, 100), font_size=8, color='orange')
    
    # Static Actors
    solid_actor = pl.add_mesh(solids, color='tan', opacity=0.1) 
    pl.add_mesh(grid.outline(), color='black')
    
    # Volumetric Rendering
    opacity = [0, 0.0, 0.1, 0.3, 0.6]
    vol_actor = pl.add_volume(grid, scalars='temperature', cmap='inferno', opacity=opacity, show_scalar_bar=True)
    
    # Dynamic State
    state = {'z': nz//2, 'y': ny//2, 't': len(therm_files) - 1, 'vol_vis': True}
    actors = {}

    def update_t():
        if not therm_files: return
        t_idx = state['t']
        T_t = np.load(therm_files[t_idx])
        grid.cell_data['temperature'] = T_t.flatten(order='F')
        
        new_mesh = grid.cell_data_to_point_data()
        mesh.point_data['temperature'] = new_mesh.point_data['temperature']
        
        nonlocal vol_actor
        try:
            pl.remove_actor(vol_actor)
        except Exception:
            pass
        vol_actor = pl.add_volume(grid, scalars='temperature', cmap='inferno', opacity=opacity, show_scalar_bar=False)
        vol_actor.SetVisibility(state['vol_vis'])
        
        update()
    
    def update():
        for a in list(actors.keys()):
            try: pl.remove_actor(actors[a])
            except: pass
            del actors[a]
            
        # Horizontal Slice
        h_slice = mesh.slice(normal='z', origin=(nx//2, ny//2, state['z']))
        if h_slice.n_points > 0:
            actors['t_h'] = pl.add_mesh(h_slice, scalars='temperature', cmap='inferno', 
                                        clim=[0, t_max], nan_color='lightgray', show_scalar_bar=False)
                                        
        # Vertical Slice
        v_slice = mesh.slice(normal='y', origin=(nx//2, state['y'], nz//2))
        if v_slice.n_points > 0:
            actors['t_v'] = pl.add_mesh(v_slice, scalars='temperature', cmap='inferno',
                                        clim=[0, t_max], nan_color='lightgray', show_scalar_bar=False)
    
    def cb_z(v): state['z'] = int(v); update()
    def cb_y(v): state['y'] = int(v); update()
    def cb_t(v): state['t'] = int(v); update_t()
    
    pl.add_slider_widget(cb_z, [0, nz], value=state['z'], title="Z Slice", pointa=(0.05, 0.9), pointb=(0.35, 0.9), fmt="%.0f")
    pl.add_slider_widget(cb_y, [0, ny], value=state['y'], title="Y Slice", pointa=(0.4, 0.9), pointb=(0.7, 0.9), fmt="%.0f")
    if len(therm_files) > 1:
        pl.add_slider_widget(cb_t, [0, len(therm_files)-1], value=state['t'], title="Time Step", pointa=(0.75, 0.9), pointb=(0.95, 0.9), fmt="%.0f")
    
    def cb_solid_toggle(flag):
        solid_actor.SetVisibility(flag)
    
    pl.add_checkbox_button_widget(cb_solid_toggle, value=True, color_on='tan', color_off='grey',
                                  position=(10, 10), size=40, border_size=2)
    pl.add_text("Solids", position=(60, 20), font_size=10)

    def cb_volume_toggle(flag):
        state['vol_vis'] = flag
        if vol_actor:
            vol_actor.SetVisibility(flag)

    pl.add_checkbox_button_widget(cb_volume_toggle, value=True, color_on='orange', color_off='grey',
                                  position=(10, 60), size=40, border_size=2)
    pl.add_text("Volume", position=(60, 70), font_size=10)
    
    pl.camera_position = 'iso'
    update()
    pl.show()

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=viz_velocity)
    p2 = multiprocessing.Process(target=viz_thermal)
    
    p1.start()
    p2.start()
    
    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("\nExiting...")
        p1.terminate()
        p2.terminate()
        p1.join()
        p2.join()
