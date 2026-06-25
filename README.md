# Mnemo OCR

A local OCR pipeline for extracting mnemonic diagram titles and sensor labels from industrial process images.

The project:

- reads `PNG`, `JPG`, `JPEG`, and `BMP` images;
- extracts images embedded in `DOCX` files;
- detects sensor regions using HSV color ranges;
- recognizes diagram titles with Tesseract OCR;
- recognizes detected ROIs with PaddleOCR;
- saves masks, cropped ROIs, and the final Excel report.

## Requirements

- Python 3.10;
- Conda or another virtual environment;
- the Tesseract OCR system application.

On macOS, install Tesseract with Homebrew:

```bash
brew install tesseract
```

Verify the installation:

```bash
tesseract --version
```

## Installation

Clone the repository and open its directory:

```bash
git clone https://github.com/bianconeri91/mnemo_ocr.git
cd mnemo_ocr
```

Create and activate an environment:

```bash
conda create -n ocr_mnemo python=3.10
conda activate ocr_mnemo
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The OCR stack uses the following versions:

```text
paddleocr==3.2.0
paddlepaddle==3.1.1
paddlex==3.2.1
pytesseract==0.3.10
```

PaddleOCR may download recognition models during the first run, which requires an internet connection.

## Input data

Create an `input` directory in the project root and place images or DOCX files inside it:

```text
input/
├── scheme_01.png
├── scheme_02.jpg
└── schemes.docx
```

The `input/` directory is listed in `.gitignore`, so its contents will not be published to GitHub.

## Configuration

The main settings are stored in `configs/config.yaml`:

```yaml
paths:
  input: "input"
  output: "outputs"

export:
  excel_filename: "ocr_results.xlsx"
```

The `colors` section contains HSV ranges used to build the combined mask:

```yaml
colors:
  purple: [[120, 80, 50], [150, 255, 255]]
  cyan: [[85, 150, 150], [100, 255, 255]]
```

Each range uses the following format:

```text
[[H_min, S_min, V_min], [H_max, S_max, V_max]]
```

If required labels are missing from the mask, adjust these values to match the colors in your mnemonic diagrams.

## Running the pipeline

Standard launch:

```bash
python -m src.pipeline
```

Alternatively, use the CLI entry point to specify a configuration file:

```bash
python run.py --config configs/config.yaml
```

## Output

After processing, the pipeline creates the following structure:

```text
outputs/
├── ocr_results.xlsx
├── masks/
│   └── scheme_01_mask.png
└── rois/
    └── scheme_01_roi_001_x100_y80_w120_h17.png
```

- `ocr_results.xlsx` contains recognized titles and sensor labels;
- `masks/` contains the final binary masks;
- `rois/` contains the regions passed to PaddleOCR.

The Excel report contains the following columns:

| Field | Description |
|---|---|
| `filename` | source image name |
| `title` | recognized mnemonic diagram title |
| `sensor_name` | recognized sensor label |
| `score` | PaddleOCR confidence score |

The `outputs/` directory is also excluded from Git and is not published to GitHub.

## Project structure

```text
configs/config.yaml       paths, HSV ranges, and export settings
src/config_loader.py      configuration loader
src/ocr_utils.py          Tesseract OCR and PaddleOCR integration
src/visualization.py      mask generation, ROI extraction, and image export
src/pipeline.py           main processing pipeline
run.py                    CLI entry point with configurable settings
tests/                    automated tests
```

## Notes

- An ROI is accepted when its width is at least `90 px` and its height is at least `17 px`.
- The saved ROI height is limited to `17 px`.
- The title is extracted from the upper-left region of the source image.
- The PaddlePaddle warning about a missing `ccache` installation is not an error and does not prevent OCR processing.

## License

This project is distributed under the terms described in [LICENSE](LICENSE).
