# subtitle

This project aligns English subtitles with a Chinese translation to produce a bilingual `.srt` file and a QC report. The process relies on semantic embeddings from the LaBSE model and monotonic dynamic programming to pair lines correctly.

## Prerequisites

- Python 3.10
- Install dependencies with `pip install -r requirements.txt`
- About 1.2 GB of free space for the `sentence-transformers/LaBSE` model, downloaded on first use

GPU hardware is used automatically when available (including Apple M2 via MPS). Otherwise the script falls back to CPU.

## Setup

1. Clone this repository.
2. Add input files under `assets/`:
   - `ENG.srt` – English subtitles with timecodes.
   - `ZH.txt` – Chinese translation, one line per cue.
3. Create an `output/` directory for generated files.

## Usage

From the project root, run:

```bash
python align.py assets/ENG.srt assets/ZH.txt -o output/
```

Use `python align.py --help` to see all options, including chunk size and similarity thresholds.

The script generates two files in `output/`:

- `bilingual.srt` – fused subtitles with English and Chinese lines separated by `\N`.
- `QC_report.txt` – summary table of cps, cosine similarity and any flags.
