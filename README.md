# Cell Extension Orientation

## How to install
Tested with Python 3.11.
```
git clone https://gitlab.com/ida-mdc/directionality-quantification.git
cd directionality-quantification
pip install -e .
```

## How to use

List arguments:

```
cell-extension-orientation --help
```

Example use case:

```
directionality-quantification --input_raw sample/input_raw.tif --input_labeling sample/input_labels.tif --input_target 
sample/input_target.tif --output sample/result --pixel_in_micron 0.65 --output_res 7:10
```

Generate exemplary output on sample data via unit test:

```
python -m unittest tests/test_sample.py
```