# pigment-retrieval-from-olci

Code and trained models of the paper:

> Sánchez-López, B., Talone, M., Cerquides, J., Di Cicco, A., Martínez-Fornos, G. and Slavinec, P. (2026).
> Simultaneous retrieval of 13 phytoplankton pigments from Sentinel-3 OLCI using convolutional neural networks and
> transfer learning. *Frontiers in Remote Sensing*, 7. https://doi.org/10.3389/frsen.2026.1880618

The models estimate the concentration of 13 phytoplankton pigments from the remote sensing reflectance (Rrs) at the
11 visible bands of Sentinel-3 OLCI, or at the 5 bands of multi-sensor products such as GlobColour.

## Get pigment predictions for your data

### 1. Install

You need [git](https://git-scm.com/) and [Python 3.9](https://www.python.org/downloads/release/python-3913/), the
version the models were trained with. Download the code and create a Python environment in its folder:

```bash
git clone https://github.com/Kissyfur/pigment-retrieval-from-olci.git
cd pigment-retrieval-from-olci
py -3.9 -m venv .venv         # on Linux or macOS: python3.9 -m venv .venv
```

Activate the environment (do it again every time you open a new terminal) and install the packages:

```bash
.venv\Scripts\activate        # on Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
```

If Windows PowerShell says that running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once, or use the Command Prompt. If you use conda, you can instead create the environment with
`conda create -n pigments python=3.9` and activate it with `conda activate pigments`.

Run all the next commands from the `pigment-retrieval-from-olci` folder, with the environment activated.

### 2. Check the installation

Predict the pigments of the example file, which has four OLCI pixels:

```bash
python bin/predict.py --data data/examples/olci_rrs_example.csv
```

The script reports that 3 of the 4 samples were predicted (the fourth has negative Rrs, see
[step 5](#5-read-the-output)) and saves them in `predictions/olci_rrs_example_OLCI_sat_ft_concatenatedCNN.csv`, where
the chlorophyll a (`chl_a`) of `pixel_1` is about 0.39 mg m⁻³.

### 3. Prepare your data

Use one of these formats:

- **CSV** file, with one row per sample and one column per band. The other columns (ids, dates, coordinates...) are
  copied to the output. See [data/examples/olci_rrs_example.csv](data/examples/olci_rrs_example.csv).
- **NetCDF** file, with one variable per band. The variables can have any dimensions, e.g. `lat` and `lon` for an
  image, or a sample dimension plus `lat` and `lon` for a square of pixels around each sample: the output has the same
  dimensions.

Name each band by its wavelength in nm (first column of the table) or as in the Copernicus Marine products:

| Band | Copernicus Marine OLCI | Copernicus Marine multi-sensor (GlobColour) |
|------|------------------------|---------------------------------------------|
| 400  | RRS400                 |                                             |
| 412  | RRS412_5               | RRS412                                      |
| 442  | RRS442_5               | RRS443                                      |
| 490  | RRS490                 | RRS490                                      |
| 510  | RRS510                 |                                             |
| 560  | RRS560                 | RRS555                                      |
| 620  | RRS620                 |                                             |
| 665  | RRS665                 |                                             |
| 673  | RRS673_75              | RRS670                                      |
| 681  | RRS681_25              |                                             |
| 708  | RRS708_75              |                                             |

The values must be Rrs in sr⁻¹, as in the Copernicus Marine products. The `Oa01_reflectance`, `Oa02_reflectance`...
bands of the OLCI Level-2 products of EUMETSAT are Rrs × π: divide them by π. If your values are already
log-transformed, ln(Rrs), add `--logged` to the command of the next step.

### 4. Run the prediction

```bash
python bin/predict.py --data path/to/my_data.csv
```

By default, the script uses the CNN of the paper fine-tuned with satellite matchups (CNN_FT in the paper) for the bands
of your data: `OLCI_sat_ft/concatenatedCNN` if it has the 11 OLCI bands, `multi_sat_ft/concatenatedCNN` if it has the
5 multi-sensor bands. For in situ radiometry, or to also predict β-carotene from the OLCI bands, choose the models
trained with in situ data with `--model`:

| `--model`                      | Bands | Pigments           | Trained with                                          |
|--------------------------------|-------|--------------------|-------------------------------------------------------|
| `OLCI_sat_ft/concatenatedCNN`  | 11    | 12 (no β-carotene) | In situ data, then fine-tuned with satellite matchups |
| `multi_sat_ft/concatenatedCNN` | 5     | 13                 | In situ data, then fine-tuned with satellite matchups |
| `OLCI/concatenatedCNN`         | 11    | 13                 | In situ data (mean of 6 models, one per data split)   |
| `multi/concatenatedCNN`        | 5     | 13                 | In situ data (mean of 6 models, one per data split)   |

For example, for in situ Rrs at the OLCI bands:

```bash
python bin/predict.py --data my_in_situ_rrs.csv --model OLCI/concatenatedCNN
```

`concatenatedCNN` is the model of the paper: one CNN module per pigment, joined by a dense output layer. Run
`python bin/predict.py --help` to see all the available models, with their bands (e.g. `OLCI/cnn`, a single CNN for
all the pigments), and all the options:

| Option             | Description                                                                         |
|--------------------|-------------------------------------------------------------------------------------|
| `--data`           | CSV or NetCDF file with your data (required)                                        |
| `--model`          | Model to use (default: the satellite model for the bands of your data)              |
| `--logged`         | Your data contain ln(Rrs) instead of Rrs                                            |
| `--floor_quantile` | Also predict the samples with non-positive Rrs (see step 5)                         |
| `--output_dir`     | Folder of the output file (default: `predictions`)                                  |

### 5. Read the output

The predictions are saved in the `predictions` folder, named after your file and the model, e.g.
`predictions/my_data_OLCI_sat_ft_concatenatedCNN.csv`. For a CSV file, the output is your file with one more column per
pigment. For a NetCDF file, it is a NetCDF file with one variable per pigment and the dimensions of your data. The
concentrations are in mg m⁻³:

| Column  | Pigment                    | Abbreviation in the paper |
|---------|----------------------------|---------------------------|
| `chlid` | Chlorophyllide a           | chlide                    |
| `chl_a` | Chlorophyll a              | chla                      |
| `chl_b` | Chlorophyll b              | chlb                      |
| `chc12` | Chlorophyll c1+c2          | chlc12                    |
| `fucox` | Fucoxanthin                | fuco                      |
| `hxfcx` | 19'-hexanoyloxyfucoxanthin | hex                       |
| `btfcx` | 19'-butanoyloxyfucoxanthin | but                       |
| `diadi` | Diadinoxanthin             | diad                      |
| `allox` | Alloxanthin                | allo                      |
| `diato` | Diatoxanthin               | diato                     |
| `zeaxa` | Zeaxanthin                 | zea                       |
| `betac` | β-carotene                 | caro                      |
| `perid` | Peridinin                  | peri                      |

The models work with the logarithm of Rrs, so samples with a missing or non-positive Rrs in any band are not
predicted, and the script tells how many. In satellite data this is frequent in the red bands of clear waters. To
predict them too, add `--floor_quantile 0.05`: the Rrs of each band below the 5th percentile of its positive values in
your file are raised to that percentile, as the paper did with its training data (5th percentile for the in situ data,
1st for the satellite matchups). Use it with files with many samples, such as images, and take those predictions with
caution: their red-band Rrs become very low values, at or beyond the lower end of the training data.

The models were trained with in situ radiometry from European waters (Mediterranean Sea, Black Sea, Atlantic and
English Channel), and the `_sat_ft` models were then fine-tuned with satellite matchups. See the paper for the accuracy
of each model, and expect it to be lower in waters unlike those.

### Use from Python

From the repository folder, the predictions can also be computed for a pandas DataFrame or an xarray Dataset:

```python
import xarray as xr
from src.predict.predict import Predictor

predictor = Predictor('OLCI_sat_ft/concatenatedCNN')
pigments = predictor.predict(xr.open_dataset('my_image.nc'))  # an xarray Dataset with the dimensions of the image
```

## Repository structure

| Folder                   | Content                                                                                              |
|--------------------------|------------------------------------------------------------------------------------------------------|
| `bin/`                   | Command line scripts: `predict.py` (this guide), `split_data.py`, `run_pipeline.py` and `run_all.py` |
| `src/`                   | Data processing, models, metrics, SHAP analysis and prediction code                                  |
| `experiments_config/`    | Definition of each experiment: datasets, bands, pigments, splits and models                          |
| `hyperparameter_spaces/` | Hyperparameters of the models of each experiment                                                     |
| `experiments/`           | For each experiment and split: train/test samples, scaler of the pigments, trained models, metrics   |
| `notebooks/`             | Creation of the datasets and figures                                                                 |
| `data/`                  | Processed satellite matchups, regions and the example input file                                     |

## Reproduce the experiments

The in situ dataset is available from the PIs of the BiOMaP programme upon request, and the satellite matchups are
built from open data (see the paper and `notebooks/data`). Each experiment config expects its datasets at its
`INP_PATH` and `OUT_PATH`. Then, for example for the 11-band experiment:

```bash
python bin/split_data.py --exp_config experiments_config/OLCI.json
python bin/run_pipeline.py --exp_config experiments_config/OLCI.json --steps train_modules train_models compute_metrics
```

`split_data.py` creates the train/test splits and the scalers of the pigments. `run_pipeline.py` trains the
single-pigment modules (`train_modules`) and the models (`train_models`), and saves their metrics and SHAP values in
`experiments/<experiment>/split_*/metrics` (`compute_metrics`). Both scripts overwrite the content of
`experiments/<experiment>`. `bin/run_all.py` runs sequences of experiments, such as the training with in situ data
followed by the fine-tuning with satellite matchups.

## Citation

If you use this code or the trained models, please cite the paper:

```bibtex
@article{sanchezlopez2026pigments,
  title   = {Simultaneous retrieval of 13 phytoplankton pigments from Sentinel-3 OLCI using convolutional neural
             networks and transfer learning},
  author  = {S{\'a}nchez-L{\'o}pez, Borja and Talone, Marco and Cerquides, Jesus and Di Cicco, Annalisa and
             Mart{\'i}nez-Fornos, Gonzalo and Slavinec, Petra},
  journal = {Frontiers in Remote Sensing},
  volume  = {7},
  year    = {2026},
  doi     = {10.3389/frsen.2026.1880618}
}
```
