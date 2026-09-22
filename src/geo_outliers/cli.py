from __future__ import annotations

import argparse, json
from pathlib import Path
from .io import load_and_append, basic_clean
from .detector import detect_outliers
from .exports import export_analysis


def main():
    p=argparse.ArgumentParser(description='Apend Detection geospatial statistical outlier detector')
    p.add_argument('input',help='Directory containing one or more shapefiles')
    p.add_argument('--output',default='results')
    p.add_argument('--probability-feature',default='FRP')
    p.add_argument('--noise-level',type=int,choices=[90,95,99],default=99)
    p.add_argument('--geometry-sample',type=int,default=20000)
    p.add_argument('--quadrature-order',type=int,default=48)
    args=p.parse_args()

    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    gdf=load_and_append(args.input)
    gdf,cleaning=basic_clean(gdf)
    scored,summary,fits=detect_outliers(
        gdf,probability_feature=args.probability_feature,
        max_geometry_rows=args.geometry_sample,quadrature_order=args.quadrature_order
    )
    outputs=export_analysis(scored,fits,cleaning,summary,out,noise_level=args.noise_level)
    print(json.dumps({"cleaning":cleaning,"detector":summary,"outputs":outputs},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
