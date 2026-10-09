import argparse
import sys
import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from src.predict.predict import Predictor, available_models

# Rrs variables of the Copernicus Marine products and the bands they correspond to
PRODUCT_BANDS = {'RRS400': '400', 'RRS412_5': '412', 'RRS442_5': '442', 'RRS490': '490', 'RRS510': '510',
                 'RRS560': '560', 'RRS620': '620', 'RRS665': '665', 'RRS673_75': '673', 'RRS681_25': '681',
                 'RRS708_75': '708', 'RRS412': '412', 'RRS443': '442', 'RRS555': '560', 'RRS670': '673'}
# Default models, fine-tuned with satellite matchups: the first one whose bands are in the data is used
DEFAULT_MODELS = ['OLCI_sat_ft/concatenatedCNN', 'multi_sat_ft/concatenatedCNN']


if __name__ == '__main__':
    models = available_models()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Predict phytoplankton pigment concentrations [mg m^-3] from remote sensing reflectance (Rrs).',
        epilog='available models (--model):\n'
               + '\n'.join(f'  {name:32}{len(config["INP_VARS"])} bands ({" ".join(config["INP_VARS"])}), '
                           f'{len(config["OUT_VARS_SHORT"])} pigments, {len(splits)} model(s)'
                           for name, (config, splits) in models.items())
               + '\n\nexamples:\n'
                 '  python bin/predict.py --data data/examples/olci_rrs_example.csv\n'
                 '  python bin/predict.py --data my_olci_image.nc\n'
                 '  python bin/predict.py --data my_in_situ_rrs.csv --model OLCI/concatenatedCNN\n'
                 '  python bin/predict.py --data my_log_rrs.csv --logged')
    parser.add_argument('--data', required=True,
                        help='CSV file with one row per sample, or NetCDF file (e.g. images with lat and lon '
                             'dimensions), with one column or variable per band, named by its wavelength (e.g. 412) '
                             'or as in the Copernicus Marine products (e.g. RRS412_5). Values: Rrs in sr^-1')
    parser.add_argument('--model', choices=models, metavar='MODEL',
                        help='one of the available models listed below. Default: the CNN fine-tuned with satellite '
                             'matchups for the bands of your data, OLCI_sat_ft/concatenatedCNN (11 OLCI bands) or '
                             'multi_sat_ft/concatenatedCNN (5 bands)')
    preprocessing = parser.add_mutually_exclusive_group()
    preprocessing.add_argument('--logged', action='store_true', help='the data contain ln(Rrs) instead of Rrs')
    preprocessing.add_argument('--floor_quantile', type=float,
                               help='also predict the samples with non-positive Rrs, raising the Rrs below this '
                                    'quantile of the positive values of their band to it, as in the paper (e.g. 0.05)')
    parser.add_argument('--output_dir', default='predictions', help='folder of the output file (default: predictions)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()

    path_data = Path(args.data)
    netcdf = path_data.suffix in ('.nc', '.nc4')
    data = xr.open_dataset(path_data) if netcdf else pd.read_csv(path_data)
    renames = {name: band for name, band in PRODUCT_BANDS.items() if name in data and band not in data}
    data = data.rename(renames) if netcdf else data.rename(columns=renames)
    if args.model is None:
        args.model = next((model for model in DEFAULT_MODELS
                           if all(band in data for band in models[model][0]['INP_VARS'])), DEFAULT_MODELS[0])
    bands = models[args.model][0]['INP_VARS']

    missing = [band for band in bands if band not in data]
    if missing:
        parser.error(f'{path_data} lacks the bands {missing} needed by {args.model}. See the bands of each model '
                     f'with --help')
    values = (data[bands].to_array() if netcdf else data[bands].astype(float)).to_numpy()
    if args.logged and np.any(values > 0):
        parser.error('with --logged the data must be ln(Rrs), which is negative, but they have positive values. '
                     'If they are Rrs, remove --logged')
    if not args.logged and not np.any(values > 0):
        parser.error('the data have no positive values, so they are not Rrs. If they are ln(Rrs), add --logged')

    print(f'Predicting with {args.model} from {"ln(Rrs)" if args.logged else "Rrs"} at bands {", ".join(bands)}...')
    predictor = Predictor(args.model)
    py = predictor.predict(data, logged=args.logged, floor_quantile=args.floor_quantile)

    dir_predictions = Path(args.output_dir)
    dir_predictions.mkdir(parents=True, exist_ok=True)
    fn = dir_predictions / f'{path_data.stem}_{args.model.replace("/", "_")}{".nc" if netcdf else ".csv"}'
    if netcdf:
        py.to_netcdf(fn)
    else:
        data.join(py, rsuffix='_pred').to_csv(fn, index=False)

    predicted, n = int(py[predictor.pigments[0]].notnull().sum()), py[predictor.pigments[0]].size
    samples = 'pixels' if netcdf else 'samples'
    print(f'Predicted {predicted} of {n} {samples}, as the mean of {len(predictor.models)} model(s)')
    if predicted < n:
        print(f'{n - predicted} {samples} have missing or non-positive Rrs and were not predicted'
              + ('' if args.logged or args.floor_quantile else ' (see --floor_quantile in --help)'))
    print(f'Pigment concentrations [mg m^-3] saved in {fn}')
