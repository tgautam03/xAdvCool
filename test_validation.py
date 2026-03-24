"""
Pytest CI wrapper for xAdvCool analytical validation suite.
Run with: python -m pytest test_validation.py -v
"""
import pytest
import warp as wp

from validate_analytical import (
    test_1d_semi_infinite_conduction,
    test_3d_rectangular_poiseuille,
    test_taylor_green_vortex,
    test_kuwabara_pin_fin,
    test_couette_flow,
    test_advection_diffusion_gaussian,
    test_poiseuille_convergence,
    test_conjugate_slab,
    test_volumetric_heat_source,
    test_mach_sensitivity,
)


@pytest.fixture(scope="session", autouse=True)
def init_warp():
    wp.init()


class TestThermalEngine:
    """Tests for the thermal LBM solver."""

    def test_semi_infinite_conduction(self):
        error = test_1d_semi_infinite_conduction()
        assert error <= 3.0, f"Semi-infinite conduction error {error:.3f}% exceeds 3% threshold"

    def test_advection_diffusion_gaussian(self):
        error = test_advection_diffusion_gaussian()
        assert error <= 3.0, f"Advection-diffusion error {error:.3f}% exceeds 3% threshold"

    def test_conjugate_slab(self):
        error = test_conjugate_slab()
        assert error <= 3.0, f"CHT conjugate slab shape error {error:.3f}% exceeds 3% threshold"

    def test_volumetric_heat_source(self):
        error = test_volumetric_heat_source()
        assert error <= 3.0, f"Heat source Poisson shape error {error:.3f}% exceeds 3% threshold"


class TestFluidEngine:
    """Tests for the fluid LBM solver."""

    def test_rectangular_poiseuille(self):
        error = test_3d_rectangular_poiseuille()
        assert error <= 3.0, f"Poiseuille shape error {error:.3f}% exceeds 3% threshold"

    def test_taylor_green_vortex(self):
        error = test_taylor_green_vortex()
        assert error <= 3.0, f"Taylor-Green vortex error {error:.3f}% exceeds 3% threshold"

    def test_kuwabara_pin_fin(self):
        error = test_kuwabara_pin_fin()
        assert error <= 5.0, f"Kuwabara drag error {error:.3f}% exceeds 5% threshold"

    def test_couette_flow(self):
        error = test_couette_flow()
        assert error <= 3.0, f"Couette flow error {error:.3f}% exceeds 3% threshold"


class TestConvergenceAndSensitivity:
    """Grid convergence and operating envelope characterization."""

    def test_grid_convergence(self):
        order = test_poiseuille_convergence()
        assert order >= 1.8, f"Convergence order {order:.3f} is below 1.8 (expected ~2.0)"

    def test_mach_sensitivity(self):
        error = test_mach_sensitivity()
        assert error <= 5.0, f"Mach sensitivity error {error:.3f}% exceeds 5% at high Ma"

