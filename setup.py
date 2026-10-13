from setuptools import find_packages, setup

setup(
    name='pigment-retrieval-from-olci',
    version='0.1.0',
    description='Phytoplankton pigment concentrations from remote sensing reflectance (Sentinel-3 OLCI)',
    url='https://github.com/Kissyfur/pigment-retrieval-from-olci',
    packages=find_packages(include=['pigment_retrieval', 'pigment_retrieval.*']) + ['pigment_retrieval.trained'],
    # The trained models stay where the experiments save them, and are installed in pigment_retrieval/trained
    package_dir={'pigment_retrieval.trained': '.'},
    package_data={'pigment_retrieval.trained': ['experiments_config/*.json', 'experiments/*/split_*/scaler_y.pkl',
                                                'experiments/*/split_*/models/*.h5',
                                                'experiments/*/split_*/models/rf.pkl',
                                                'experiments/*/split_*/models/xgb.json']},
    # Command installed with the package: predict-pigments --data my_rrs.csv
    entry_points={'console_scripts': ['predict-pigments = pigment_retrieval.predict.cli:main']},
    python_requires='>=3.9',
    # What predicting needs; requirements.txt has the whole environment of the paper
    install_requires=['numpy<2', 'pandas', 'scikit-learn', 'xarray', 'netCDF4', 'tqdm', 'xgboost',
                      'tensorflow>=2.10,<2.16'],
)
