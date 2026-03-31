"""
Interactive HDF5 Dataset Viewer
Loads samples from the generated dataset and provides interactive 3D visualization
of velocity (volume) and temperature (slices) fields simultaneously, with sample
navigation and metadata display.

Usage:
    python dataset_viewer.py                          # default path: dataset/data.h5
    python dataset_viewer.py --data path/to/data.h5
    python dataset_viewer.py --data path/to/data.h5 --sample 5
"""
import argparse
import sys

import h5py
import numpy as np
import pyvista as pv


BC_SOLID = 1


def load_sample_list(h5_path):
    """Return sorted list of sample IDs and their attributes."""
    with h5py.File(h5_path, "r") as f:
        sample_ids = sorted(f.keys())
        sample_attrs = []
        for sid in sample_ids:
            attrs = dict(f[sid].attrs)
            attrs["sample_id"] = sid
            sample_attrs.append(attrs)
    return sample_ids, sample_attrs


def load_sample_data(h5_path, sample_id):
    """Load field arrays and attributes for a single sample."""
    with h5py.File(h5_path, "r") as f:
        grp = f[sample_id]
        data = {
            "mask": grp["mask"][()],
            "velocity": grp["velocity"][()],
            "temperature": grp["temperature"][()],
            "pressure": grp["pressure"][()],
            "heat_source": grp["heat_source"][()],
        }
        attrs = dict(grp.attrs)
    return data, attrs


def format_attrs_text(attrs, sample_idx, total):
    """Format sample attributes into a multiline string for display."""
    sid = attrs.get("sample_id", "?")
    lines = [
        f"Sample {sample_idx + 1} / {total}:  {sid}",
        "",
        f"Design: {attrs.get('design_name', '?')}",
        f"Grid:   {list(attrs.get('grid_shape', []))}",
        f"Re={attrs.get('Re', 0):.1f}   porosity={attrs.get('porosity', 0):.2%}   Dh={attrs.get('Dh', 0):.2f}",
        "",
        "--- Simulation Params ---",
        f"tau_fluid={attrs.get('tau_fluid', 0):.4f}   u_inlet={attrs.get('u_inlet_val', 0):.4f}   heat_power={attrs.get('heat_power', 0):.4f}",
        f"heat_source: {attrs.get('heat_source_type', '?')}",
        "",
        "--- Performance Metrics ---",
        f"R_th={attrs.get('R_th', 0):.4f}   Nu={attrs.get('Nu', 0):.2f}   f={attrs.get('f_friction', 0):.4f}",
        f"P_pump={attrs.get('P_pump', 0):.4e}   COP={attrs.get('COP', 0):.2f}   Q_vol={attrs.get('Q_vol', 0):.4f}",
        f"T_max_surf={attrs.get('T_max_surface', 0):.2f}   T_avg_surf={attrs.get('T_avg_surface', 0):.2f}",
        f"dT_max={attrs.get('delta_T_max', 0):.2f}   sigma_T={attrs.get('sigma_T', 0):.4f}   fluid_rise={attrs.get('fluid_temp_rise', 0):.4f}",
        "",
        f"max_v={attrs.get('max_velocity', 0):.4f}   max_T={attrs.get('max_temperature', 0):.2f}   mean_T={attrs.get('mean_temperature', 0):.2f}",
        f"elapsed={attrs.get('elapsed_s', 0):.1f}s",
    ]
    return "\n".join(lines)


class DatasetViewer:
    def __init__(self, h5_path, start_idx=0):
        self.h5_path = h5_path
        self.sample_ids, self.sample_attrs = load_sample_list(h5_path)
        self.total = len(self.sample_ids)
        if self.total == 0:
            print("No samples found in the dataset.")
            sys.exit(1)

        self.idx = min(start_idx, self.total - 1)

        # PyVista state
        self.pl = pv.Plotter()
        self.pl.enable_depth_peeling()
        self.grid = None
        self.mesh = None
        self.actors = {}
        self.solid_actor = None
        self.vol_actor = None
        self.info_actor = None
        self.slice_z = 0
        self.slice_y = 0
        self.show_vol = True
        self.show_solids = True
        self.slice_field_idx = 0  # 0=temperature, 1=pressure, 2=heat_source
        self.slice_fields = ["temperature", "pressure", "heat_source"]
        self.slice_cmaps = {"temperature": "inferno", "pressure": "viridis", "heat_source": "Oranges"}

        self._load_and_display()
        self._setup_widgets()
        self.pl.camera_position = "iso"
        self.pl.show()

    def _clear_actors(self):
        """Remove all dynamic actors."""
        for key in list(self.actors.keys()):
            try:
                self.pl.remove_actor(self.actors[key])
            except Exception:
                pass
        self.actors.clear()
        if self.solid_actor is not None:
            try:
                self.pl.remove_actor(self.solid_actor)
            except Exception:
                pass
            self.solid_actor = None
        if self.vol_actor is not None:
            try:
                self.pl.remove_actor(self.vol_actor)
            except Exception:
                pass
            self.vol_actor = None

    def _load_and_display(self):
        """Load current sample and rebuild the scene."""
        self._clear_actors()

        sid = self.sample_ids[self.idx]
        data, attrs = load_sample_data(self.h5_path, sid)
        self.data = data
        self.attrs = attrs

        mask = data["mask"]
        nx, ny, nz = mask.shape
        self.nx, self.ny, self.nz = nx, ny, nz
        self.slice_z = nz // 2
        self.slice_y = ny // 2

        # Compute velocity magnitude (NaN in solids for volume rendering)
        u = data["velocity"]
        v_mag = np.linalg.norm(u, axis=-1).astype(np.float32)
        solid = (mask == BC_SOLID)
        v_mag[solid] = np.nan
        self.v_mag = v_mag

        # Temperature (keep solids visible on slices)
        T = data["temperature"].copy()
        self.T_field = T

        # Pressure (NaN in solids)
        P = data["pressure"].copy()
        P[solid] = np.nan
        self.P_field = P

        # Heat source
        self.heat_source = data["heat_source"]

        # Color ranges
        self.v_max = float(np.nanmax(v_mag)) if not np.all(np.isnan(v_mag)) else 0.05
        self.clims = {
            "temperature": (float(np.nanmin(T)), float(np.nanmax(T))),
            "pressure": (float(np.nanmin(P[~np.isnan(P)])), float(np.nanmax(P[~np.isnan(P)]))) if np.any(~np.isnan(P)) else (0, 1),
            "heat_source": (0, max(float(np.max(self.heat_source)), 1e-6)),
        }

        # Build grid with all fields
        self.grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
        self.grid.cell_data["mask"] = mask.flatten(order="F")
        self.grid.cell_data["velocity"] = v_mag.flatten(order="F")
        self.grid.cell_data["temperature"] = T.flatten(order="F")
        self.grid.cell_data["pressure"] = P.flatten(order="F")
        self.grid.cell_data["heat_source"] = self.heat_source.flatten(order="F")

        self.mesh = self.grid.cell_data_to_point_data()

        # Solids
        solids_mesh = self.mesh.threshold([0.9, 1.1], scalars="mask")
        self.solid_actor = self.pl.add_mesh(solids_mesh, color="tan", opacity=0.3)
        self.solid_actor.SetVisibility(self.show_solids)

        # Outline
        if "outline" not in self.actors:
            self.actors["outline"] = self.pl.add_mesh(self.grid.outline(), color="black")

        # Velocity volume rendering
        self._add_volume()

        # Info text
        if self.info_actor is not None:
            try:
                self.pl.remove_actor(self.info_actor)
            except Exception:
                pass
        info_text = format_attrs_text({**attrs, "sample_id": sid}, self.idx, self.total)
        self.info_actor = self.pl.add_text(info_text, position="upper_left", font_size=8, color="white")

        # Temperature slices
        self._update_slices()

    def _add_volume(self):
        """Add velocity volume rendering."""
        if self.vol_actor is not None:
            try:
                self.pl.remove_actor(self.vol_actor)
            except Exception:
                pass
        opacity = [0, 0.0, 0.1, 0.3, 0.6]
        self.vol_actor = self.pl.add_volume(
            self.grid, scalars="velocity", cmap="coolwarm",
            opacity=opacity, show_scalar_bar=True, clim=[0, self.v_max],
            scalar_bar_args={"title": "Velocity"}
        )
        self.vol_actor.SetVisibility(self.show_vol)

    def _update_slices(self):
        """Update slices at current Z and Y positions with the active slice field."""
        for key in ["slice_z", "slice_y", "slice_sbar"]:
            if key in self.actors:
                try:
                    self.pl.remove_actor(self.actors[key])
                except Exception:
                    pass
                del self.actors[key]

        field = self.slice_fields[self.slice_field_idx]
        cmap = self.slice_cmaps[field]
        clim = list(self.clims[field])

        # Z slice (horizontal cross-section)
        h_slice = self.mesh.slice(normal="z", origin=(self.nx // 2, self.ny // 2, self.slice_z))
        if h_slice.n_points > 0:
            self.actors["slice_z"] = self.pl.add_mesh(
                h_slice, scalars=field, cmap=cmap, clim=clim,
                nan_color="lightgray", show_scalar_bar=True,
                scalar_bar_args={"title": field.capitalize(), "position_x": 0.85}
            )

        # Y slice (vertical cross-section)
        v_slice = self.mesh.slice(normal="y", origin=(self.nx // 2, self.slice_y, self.nz // 2))
        if v_slice.n_points > 0:
            self.actors["slice_y"] = self.pl.add_mesh(
                v_slice, scalars=field, cmap=cmap, clim=clim,
                nan_color="lightgray", show_scalar_bar=False
            )

    def _setup_widgets(self):
        # Sample navigation slider
        if self.total > 1:
            self.pl.add_slider_widget(
                self._cb_sample, [0, self.total - 1], value=self.idx,
                title="Sample", pointa=(0.05, 0.92), pointb=(0.45, 0.92), fmt="%.0f",
                style="modern",
            )

        # Z slice slider
        self.pl.add_slider_widget(
            self._cb_z, [0, max(self.nz, 1)], value=self.slice_z,
            title="Z Slice", pointa=(0.5, 0.92), pointb=(0.7, 0.92), fmt="%.0f",
        )

        # Y slice slider
        self.pl.add_slider_widget(
            self._cb_y, [0, max(self.ny, 1)], value=self.slice_y,
            title="Y Slice", pointa=(0.75, 0.92), pointb=(0.95, 0.92), fmt="%.0f",
        )

        # Slice field slider: 0=Temperature, 1=Pressure, 2=Heat Source
        self.pl.add_slider_widget(
            self._cb_field, [0, 2], value=self.slice_field_idx,
            title="Slice Field (T / P / Q)",
            pointa=(0.05, 0.05), pointb=(0.45, 0.05), fmt="%.0f",
        )

        # Toggle checkboxes
        self.pl.add_checkbox_button_widget(
            self._cb_solid_toggle, value=True, color_on="tan", color_off="grey",
            position=(10, 10), size=40, border_size=2,
        )
        self.pl.add_text("Solids", position=(60, 20), font_size=10)

        self.pl.add_checkbox_button_widget(
            self._cb_volume_toggle, value=True, color_on="orange", color_off="grey",
            position=(10, 60), size=40, border_size=2,
        )
        self.pl.add_text("Velocity Vol", position=(60, 70), font_size=10)

        self.pl.add_key_event("Right", self._next_sample)
        self.pl.add_key_event("Left", self._prev_sample)

        self.pl.add_text(
            "Volume: Velocity  |  Slices: use slider (0=Temp, 1=Pressure, 2=HeatSrc)  |  Left/Right: Navigate",
            position="lower_left", font_size=8, color="yellow",
        )

    def _cb_sample(self, value):
        new_idx = int(round(value))
        if new_idx != self.idx:
            self.idx = new_idx
            self._load_and_display()

    def _cb_field(self, value):
        new_idx = int(round(value))
        new_idx = max(0, min(new_idx, len(self.slice_fields) - 1))
        if new_idx != self.slice_field_idx:
            self.slice_field_idx = new_idx
            self._update_slices()

    def _cb_z(self, value):
        self.slice_z = int(value)
        self._update_slices()

    def _cb_y(self, value):
        self.slice_y = int(value)
        self._update_slices()

    def _cb_solid_toggle(self, flag):
        self.show_solids = flag
        if self.solid_actor is not None:
            self.solid_actor.SetVisibility(flag)

    def _cb_volume_toggle(self, flag):
        self.show_vol = flag
        if self.vol_actor is not None:
            self.vol_actor.SetVisibility(flag)

    def _next_sample(self):
        if self.idx < self.total - 1:
            self.idx += 1
            self._load_and_display()

    def _prev_sample(self):
        if self.idx > 0:
            self.idx -= 1
            self._load_and_display()


def main():
    parser = argparse.ArgumentParser(description="Interactive HDF5 Dataset Viewer")
    parser.add_argument("--data", type=str, default="dataset/data.h5", help="Path to HDF5 dataset file")
    parser.add_argument("--sample", type=int, default=0, help="Starting sample index")
    args = parser.parse_args()

    print(f"Loading dataset: {args.data}")
    DatasetViewer(args.data, start_idx=args.sample)


if __name__ == "__main__":
    main()
