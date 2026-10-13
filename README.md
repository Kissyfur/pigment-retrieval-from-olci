# pigment-retrieval-from-olci

Code and trained models of the paper:

> Sánchez-López, B., Talone, M., Cerquides, J., Di Cicco, A., Martínez-Fornos, G. and Slavinec, P. (2026).
> Simultaneous retrieval of 13 phytoplankton pigments from Sentinel-3 OLCI using convolutional neural networks and
> transfer learning. *Frontiers in Remote Sensing*, 7. https://doi.org/10.3389/frsen.2026.1880618

The models estimate the concentration of 13 phytoplankton pigments from the remote sensing reflectance (Rrs), either
11-band from Sentinel-3 OLCI, or 5-band from legacy sensors as well as multi-sensor products such as GlobColour.

## Get pigment predictions for your data

### 1. Install

You will need [git](https://git-scm.com/) and [Python 3.9](https://www.python.org/downloads/release/python-3913/). It
may not work with other versions. Download the code and create a Python environment in its folder:

```bash
git clone https://github.com/Kissyfur/pigment-retrieval-from-olci.git
cd pigment-retrieval-from-olci
py -3.9 -m venv .venv         # on Linux or macOS: python3.9 -m venv .venv
```

Activate the environment (do it every time you open a new terminal) and install the packages:

```bash
.venv\Scripts\activate        # on Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
```

If Windows PowerShell says that running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
once, or use the Command Prompt. If you use conda, you can instead create the environment with
`conda create -n pigments python=3.9` and activate it with `conda activate pigments`.

Run all the next commands from the `pigment-retrieval-from-olci` folder, with the environment activated.

### 2. Check the installation

Predict the pigments from the example file containing four OLCI pixels:

```bash
python bin/predict.py --data data/examples/olci_rrs_example.csv
```

The script will report that 3 of the 4 samples were predicted (the fourth has negative Rrs, see
[step 5](#5-read-the-output)) and save them in `predictions/olci_rrs_example_OLCI_sat_ft_concatenatedCNN.csv`, where
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

By default, the script uses the CNN model detailed in the paper, i.e., trained with in-situ data and fine-tuned with
satellite matchups (CNN_FT in the paper): either `OLCI_sat_ft/concatenatedCNN` or `multi_sat_ft/concatenatedCNN` for
11-band and 5-band Rrs, respectively. Models relying only on in-situ data could provide better performance if applied
to in-situ observations. They can be selected with the `--model` option:

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

All four models are described in the reference paper. Run `python bin/predict.py --help` to see all the available
models as well as the corresponding bands (e.g. `OLCI/cnn`, a single CNN for all the pigments) and options:

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
predicted, and the script will report the number of failed retrievals. Negative Rrs is frequent in satellite
measurements, especially at the longest wavelengths. To overcome this issue, add `--floor_quantile 0.05`: the Rrs of
each band below the 5th percentile of its positive values are set to that percentile.

The models were trained with in situ radiometry from European waters (Mediterranean Sea, Black Sea, Atlantic and
English Channel), and the `_sat_ft` models were then fine-tuned with satellite matchups. See the paper for the accuracy
of each model.

### Use it from another project

The code and the trained models are the Python package `pigment_retrieval`, which you can install from GitHub in the
environment of any project (Python 3.9 to 3.11), without downloading this repository:

```bash
pip install git+https://github.com/Kissyfur/pigment-retrieval-from-olci.git
```

Then predict with the `predict-pigments` command, which has the same options as `bin/predict.py`:

```bash
predict-pigments --data my_data.csv
```

or from your Python code, for a pandas DataFrame or an xarray Dataset with one column or variable per band:

```python
import xarray as xr
from pigment_retrieval.predict.predict import Predictor

predictor = Predictor('OLCI_sat_ft/concatenatedCNN')
pigments = predictor.predict(xr.open_dataset('my_image.nc'))  # an xarray Dataset with the dimensions of the image
```

To update the package to the latest version on GitHub, add `--force-reinstall --no-deps` to the `pip install` command.

## Model card

### Model details

- **Developed by:** Borja Sánchez-López, Marco Talone, Jesus Cerquides, Annalisa Di Cicco, Gonzalo Martínez-Fornos and
  Petra Slavinec.
- **Model type:** regression with 1D convolutional neural networks (Keras, TensorFlow 2.10): one CNN module per
  pigment, joined by a dense output layer.
- **Input:** Rrs [sr⁻¹] at the 11 OLCI bands (400–708 nm) or at 5 bands (412, 442, 490, 560 and 673 nm).
- **Output:** concentrations [mg m⁻³] of 13 phytoplankton pigments as measured by HPLC (12, without β-carotene, for
  `OLCI_sat_ft`).
- **Paper:** https://doi.org/10.3389/frsen.2026.1880618
- **License:** code under Apache-2.0 and trained models under CC BY 4.0 (see [License](#license)).

### Intended use

Research on phytoplankton communities from ocean colour: estimating the concentration of diagnostic pigments from in
situ radiometry, Sentinel-3 OLCI images or multi-sensor products such as Copernicus-GlobColour, in waters similar to
those of the training data.

### Out-of-scope use

- Replacing HPLC measurements where measured concentrations are required.
- Rrs from sensors or bands that do not match the ones above, or from waters optically unlike those of the training
  data, without validating the predictions there.
- Satellite pixels affected by clouds, sun glint, land adjacency or a failed atmospheric correction.

### Bias, risks and limitations

- **Geographical bias:** the in situ training data (185 stations of the BiOMaP programme, 2014–2021) come from
  European waters, which the paper recognizes as a limiting factor for global transferability of both the 11- and
  5-band models.
- **Lower accuracy on satellite data:** in situ, the paper reports for the CNN coefficients of determination (R²) of
  0.70–0.93 and mean absolute percentage errors (MAPE) of 25–55% across the 13 pigments. On satellite data, the R² of
  the fine-tuned models decreases by about 0.2 and their MAPE is generally lower than 70%. Atmospheric correction (the
  atmosphere often accounts for more than 90% of the signal measured by a satellite), adjacency to land or clouds and
  sub-pixel heterogeneity add uncertainty.
- **Pigments:** zeaxanthin and 19'-butanoyloxyfucoxanthin remain challenging regardless of model complexity, due to
  their low concentrations, poor representation in the training data and spectral overlap with dominant pigments.
  `OLCI_sat_ft` does not predict β-carotene, which is missing from its satellite matchups.
- **Satellite training data:** the matchups (668 for 11 bands, 7,287 for 5 bands) combine pigment data from several
  sources (MAREDAT, NOMAD and TARA) with non-uniform pigment suites.
- **ESA OC-CCI:** the applicability to this dataset was not explicitly assessed, although no compatibility issues are
  expected.
- **Non-positive Rrs:** the Rrs set with `--floor_quantile` are at or beyond the lower end of the training data, so the
  predictions of those samples are less reliable.
- **No uncertainty per sample:** the predictions are point estimates.

### Recommendations

- Validate the predictions against local HPLC measurements before using them in new regions, waters or sensors.
- Use the fine-tuned (`_sat_ft`) models for satellite data and the in situ models for in situ radiometry, and report
  which model you used.
- Mask the cloudy, glint-affected and coastal-adjacency pixels with the quality flags of the satellite product before
  predicting.

### Training and evaluation

- **Data:** in situ Rrs and HPLC pigments from 185 stations of the BiOMaP programme (2014–2021). For the fine-tuning,
  Copernicus-GlobColour Rrs matched with HPLC pigments from MAREDAT, NOMAD and TARA.
- **Preprocessing:** Rrs and pigments are log-transformed, after setting the values below a low percentile to that
  percentile. The training data are augmented by mixing pairs of samples and adding Gaussian noise.
- **Training:** one CNN module per pigment, selected by 3-fold cross-validation; the modules are then joined by a dense
  output layer, trained for 300 epochs with the modules frozen and 100 more with all the layers. The `_sat_ft` models
  are then fine-tuned with the satellite matchups.
- **Evaluation:** R² and mean squared error of the log-transformed pigments, and MAE, MAPE, ME and MPE of the
  concentrations, on a 15% test set stratified by clusters of the pigments; for the in situ models the whole process is
  repeated with 6 different splits (see `experiments/*/split_*/metrics`).

### Contact

Open an issue in this repository.

## Repository structure

| Folder                   | Content                                                                                              |
|--------------------------|------------------------------------------------------------------------------------------------------|
| `bin/`                   | Command line scripts: `predict.py` (this guide), `split_data.py`, `run_pipeline.py` and `run_all.py` |
| `pigment_retrieval/`     | Python package: data processing, models, metrics, SHAP analysis and prediction code                  |
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
  title   = {{Simultaneous retrieval of 13 phytoplankton pigments from Sentinel-3 OLCI using convolutional neural
             networks and transfer learning}},
  author  = {S{\'a}nchez-L{\'o}pez, Borja and Talone, Marco and Cerquides, Jesus and Di Cicco, Annalisa and
             Mart{\'i}nez-Fornos, Gonzalo and Slavinec, Petra},
  journal = {Frontiers in Remote Sensing},
  volume  = {7},
  year    = {2026},
  doi     = {10.3389/frsen.2026.1880618}
}
```

## License

This is open research: anyone can use, modify and share the code and the trained models, as long as the authors are
credited. Please cite the paper above in any work that uses them or builds on them.

- The code is licensed under the [Apache License 2.0](LICENSE). Copies and derived works must keep the [NOTICE](NOTICE)
  file, which credits the authors and the paper.
- The trained models (the model and scaler files in `experiments/`) are licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), like the paper.
- Data obtained from other providers (e.g. Copernicus Marine Service, SeaBASS) keep the terms of their providers.

Copyright 2026 Borja Sánchez-López, Marco Talone, Jesus Cerquides, Annalisa Di Cicco, Gonzalo Martínez-Fornos and
Petra Slavinec.

## Funding

This work was supported partly by Biodiversa+, the European Biodiversity Partnership, in the context of the PETRI-MED
project under the 2021–2022 BiodivProtect joint call for research proposals, co-funded by the European Commission
(GA N°101052342), the Spanish Ministry for the Ecological Transition and the Demographic Challenge (MITECO), the Spanish
Fundación Biodiversidad, the Italian Ministry of Universities and Research (MUR), and the Slovenian Ministry of Higher
Education, Science and Innovation; partly by the “Severo Ochoa Centre of Excellence” Accreditation under Grant
CEX 2024-001494-S funded by AEI 10.13039/501100011033; partly by EuroMarine through OYSTER grants for Early Career
Researchers in the context of the project DROPTHEMIC-MED. GM.F. acknowledges support from the FPI Grant
CEX 2019-421 000928-S-21-1. AD contributed to this work also in the context of the Copernicus Marine Service (contract
Refs. 21001L2-COP-TAC OC-2200 and 24251L03-COP-TAC OC-2200).

## Acknowledgements

BiOMaP data were collected under the leadership of the European Commission’s Joint Research Centre. The authors would
like to acknowledge Giuseppe Zibordi and Jean-François Berthon for their continuous support to the BiOMaP programme as
well as all the scientific and technical personnel involved in the preparation and execution of its numerous
oceanographic cruises. The Editor and two reviewers are duly acknowledged for their careful reading of the manuscript
and their insightful comments. Copernicus-GlobColour data used in this study were developed and validated by ACRI-ST,
France, and has been accessed through the Copernicus Marine Service. NOMAD data used in this study were downloaded from
the SeaWiFS Bio-optical Archive and Storage System (SeaBASS) and contributed by participants in the NASA SIMBIOS Program
(NRA-96-MTPE-04 and NRA-99-OES-09) as well as by voluntary contributors. All scientific and technical personnel involved
in the preparation of the following datasets are gratefully acknowledged: 10.5067_SeaBASS_ACIDD_DATA001,
10.5067_SeaBASS_AMLR_DATA001, 10.5067_SeaBASS_BENTHICECOLOGY_FROMSPAC, 10.5067_SeaBASS_BIOCOMPLEXITY_DATA001,
10.5067_SeaBASS_BOWDOINBUOY_DATA001, 10.5067_SeaBASS_CASES_DATA001, 10.5067_SeaBASS_CCE-LTER_DATA001,
10.5067_SeaBASS_CHESAPEAKE_LIGHT_TOWER, 10.5067_SeaBASS_ECOHAB_DATA001, 10.5067_SeaBASS_ECOMON_DATA001,
10.5067_SeaBASS_EGEE3_DATA001, 10.5067_SeaBASS_EGEE5_DATA001, 10.5067_SeaBASS_GOMECC_DATA001,
10.5067_SeaBASS_GULFCARBON_DATA001, 10.5067_SeaBASS_INTRO_DATA001, 10.5067_SeaBASS_LAMONT_SCS_DATA001,
10.5067_SeaBASS_LATTE_DATA001, 10.5067_SeaBASS_LMER-TIES_DATA001, 10.5067_SeaBASS_MOCE_PIGMENT_DATABASE_D,
10.5067_SeaBASS_MURI_CAMOUFLAGE_DATA001, 10.5067_SeaBASS_MVCO_DATA001, 10.5067_SeaBASS_NES-LTER_DATA001,
10.5067_SeaBASS_NOAA_DATA001, 10.5067_SeaBASS_NSF-BWZ_DATA001, 10.5067_SeaBASS_NSF_GULF_RAPID_DATA001,
10.5067_SeaBASS_PACE_ABSCLOSURE_DATA001, 10.5067_SeaBASS_SABOR_DATA001, 10.5067_SeaBASS_SBI_DATA001,
10.5067_SeaBASS_SFMBON_DATA001, 10.5067_SeaBASS_TARA_OCEANS_POLAR_CIRCL, 10.5067_SeaBASS_THREERIVERS_DATA001,
10.5067_SeaBASS_VOLPE_DATA001.
