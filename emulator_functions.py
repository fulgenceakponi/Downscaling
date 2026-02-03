# emulator_functions.py
# ======================

import xarray as xr


def load_aladin_dataset(files, variables):
    """
    Charge les fichiers NetCDF ALADIN63 en ignorant
    les conflits de métadonnées (ex: CRS).
    """
    if len(files) == 0:
        raise ValueError("Aucun fichier NetCDF trouvé.")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        compat="override",
        coords="minimal",
        data_vars="minimal",
        parallel=False,
        engine="netcdf4"
    )

    return ds[variables]
