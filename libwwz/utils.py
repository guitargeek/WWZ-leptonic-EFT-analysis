import pyarrow.parquet as pq
import pandas as pd
import os
import json

xsec = dict(
    WZ=4429.7,
    WWZ_4l=4.12,
    WWZ_incl=165.1,
    TTZLOW=49.3,
    TTZnlo=272.8,
    TTZnlo_ext1=272.8,
    TTZnlo_ext2=272.8,
    TTZnlo_ext=272.8,
    ZZ=1381.6,
    ZZ_ext1=1381.6,
    TWZ=11.23,
    DY_high=6_025_300.0,
    TTDL=87315.0,
    TTSLtop=109_100.0,
    TTSLtopbar=109_100.0,
    TTSL=2 * 109_100.0,
)

lumi = {2016: 35.92, 2017: 41.53, 2018: 59.74}


def load_four_lepton_skim(year, short_name, override_path=None):
    if not override_path is None:
        path = override_path
    else:
        path = f"/scratch/skims/mc/{year}/{short_name}/four_lepton_skim"

    _dataset = pq.ParquetDataset(os.path.join(path, "parquet"))
    df = _dataset.read_pandas().to_pandas()

    if short_name in xsec:

        with open(os.path.join(path, "metainfo.json"), "r") as myfile:
            data = myfile.read()
            full_weight_sum = json.loads(data)["genWeightSum"]
        df["weight"] = df["genWeight"] * lumi[year] * xsec[short_name] / full_weight_sum

    return df
