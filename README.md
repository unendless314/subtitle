# Subtitle Alignment Demo

This project demonstrates bilingual subtitle alignment using `align.py`.

## Quick Start

Install the dependencies listed in `requirements.txt` and run:

```bash
python align.py --chunk-size 400 --stop-threshold 0.25 --max-gap 3 \
  --cps-threshold 17 --similarity-flag 0.60 --device auto
```

Outputs are written to the `output/` folder. On success the script exits with code `0`.
