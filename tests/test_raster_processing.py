import numpy as np
import pytest

from geo_outliers.raster_processing import compute_spectral_indices, scale_band, cloud_mask, zonal_statistics


def test_spectral_indices_are_computed_from_scaled_bands():
    bands={
        "red":np.array([[.2,.4]],dtype=float),
        "green":np.array([[.3,.5]],dtype=float),
        "nir":np.array([[.6,.2]],dtype=float),
        "swir16":np.array([[.3,.1]],dtype=float),
        "swir22":np.array([[.2,.1]],dtype=float),
    }
    out=compute_spectral_indices(bands,["NDVI","NDMI","NBR","NDWI"])
    assert out["NDVI"][0,0]==pytest.approx(.5)
    assert out["NDMI"][0,0]==pytest.approx(1/3)
    assert out["NBR"][0,0]==pytest.approx(.5)
    assert out["NDWI"][0,0]==pytest.approx(-1/3)


def test_landsat_scaling_and_temperature_kelvin_to_celsius():
    refl=scale_band("landsat-c2-l2","red",np.array([10000.]))
    thermal=scale_band("landsat-c2-l2","thermal",np.array([40000.]))
    assert refl[0]==pytest.approx(.075,abs=1e-5)
    assert thermal[0]==pytest.approx(12.5708,abs=1e-3)


def test_sentinel_scl_cloud_mask():
    qa=np.array([[4,8,9,6,3]])
    valid,meta=cloud_mask("sentinel-2-l2a",qa)
    assert valid.tolist()==[[True,False,False,True,False]]
    assert meta["applied"] is True


def test_zonal_statistics_respect_mask():
    a=np.array([[1.,2.],[3.,4.]])
    mask=np.array([[True,False],[True,False]])
    stats=zonal_statistics(a,mask)
    assert stats["count"]==2
    assert stats["mean"]==pytest.approx(2.0)
    assert stats["p95"]>stats["median"]


def test_zero_denominator_becomes_nan():
    bands={"nir":np.array([[1.]]),"red":np.array([[-1.]])}
    out=compute_spectral_indices(bands,["NDVI"])
    assert np.isnan(out["NDVI"][0,0])
