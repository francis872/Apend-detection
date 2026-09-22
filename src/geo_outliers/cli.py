from __future__ import annotations

import argparse, json
from pathlib import Path
from .io import load_and_append, basic_clean
from .detector import detect_outliers, remove_noise


def main():
    p = argparse.ArgumentParser(description='Geospatial statistical outlier detector')
    p.add_argument('input', help='Directory containing one or more shapefiles')
    p.add_argument('--output', default='data/processed')
    p.add_argument('--probability-feature', default='FRP')
    p.add_argument('--noise-level', type=int, choices=[90,95,99], default=99)
    p.add_argument('--geometry-sample', type=int, default=20000)
    args = p.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    gdf = load_and_append(args.input)
    gdf, clean_report = basic_clean(gdf)
    scored, summary, fits = detect_outliers(
        gdf, probability_feature=args.probability_feature,
        max_geometry_rows=args.geometry_sample
    )
    cleaned = remove_noise(scored, args.noise_level)

    scored.to_file(out/'outliers.gpkg', layer='outliers', driver='GPKG')
    cleaned.to_file(out/'cleaned.gpkg', layer='cleaned', driver='GPKG')
    fits.to_csv(out/'distribution_fits.csv', index=False)
    with open(out/'summary.json','w',encoding='utf-8') as f:
        json.dump({'cleaning': clean_report, 'detector': summary}, f, ensure_ascii=False, indent=2)
    print(json.dumps({'cleaning': clean_report, 'detector': summary}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
