import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Heavy imports are done lazily inside functions so that `--help` works even
# when dependencies are missing.


def timestamp(start: float) -> str:
    """Return elapsed time since start as [HH:MM:SS]."""
    elapsed = int(time.perf_counter() - start)
    h, m = divmod(elapsed, 3600)
    m, s = divmod(m, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def log(msg: str, start: float) -> None:
    print(f"{timestamp(start)} {msg}")


def load_inputs(eng_path: Path, zh_path: Path):
    import pysubs2
    subs = pysubs2.load(str(eng_path))
    with zh_path.open("r", encoding="utf-8") as f:
        zh_lines = [line.strip() for line in f if line.strip()]
    if abs(len(subs) - len(zh_lines)) / max(len(subs), 1) > 0.10:
        raise ValueError("Line count difference between ENG and ZH exceeds 10%")
    return subs, zh_lines


def segment_texts(en_lines: List[str], zh_lines: List[str], device: str) -> Tuple[List[str], List[str]]:
    import spacy
    import jieba
    nlp = spacy.load("en_core_web_sm")
    seg_en = [" ".join(tok.text for tok in nlp(line)) for line in en_lines]
    seg_zh = [" ".join(jieba.cut(line)) for line in zh_lines]
    return seg_en, seg_zh


def embed_texts(texts: List[str], model, chunk: int, start_time: float):
    import torch
    embeddings = []
    for i in range(0, len(texts), chunk):
        log(f"Embedding chunk {i//chunk+1}/{(len(texts)-1)//chunk+1} …", start_time)
        part = model.encode(texts[i:i+chunk], convert_to_tensor=True, show_progress_bar=False)
        embeddings.append(part)
    return torch.cat(embeddings, dim=0)


def align(sim_mat, max_gap: int, stop_th: float) -> List[int]:
    import torch
    n_en, n_zh = sim_mat.shape
    dp = torch.full((n_en + 1, n_zh + 1), -1e9)
    ptr = torch.zeros((n_en + 1, n_zh + 1, 2), dtype=torch.long)
    dp[0, 0] = 0.0
    for i in range(1, n_en + 1):
        for j in range(max(1, i - max_gap), min(n_zh, i + max_gap) + 1):
            match = dp[i - 1, j - 1] + sim_mat[i - 1, j - 1]
            skip_en = dp[i - 1, j] - stop_th
            skip_zh = dp[i, j - 1] - stop_th
            scores = torch.stack([match, skip_en, skip_zh])
            best = torch.argmax(scores)
            dp[i, j] = scores[best]
            if best == 0:
                ptr[i, j] = torch.tensor([i - 1, j - 1])
            elif best == 1:
                ptr[i, j] = torch.tensor([i - 1, j])
            else:
                ptr[i, j] = torch.tensor([i, j - 1])
    # Backtrack
    alignment = [-1] * n_en
    i, j = n_en, n_zh
    while i > 0 and j >= 0:
        pi, pj = ptr[i, j]
        if pi == i - 1 and pj == j - 1:
            alignment[i - 1] = j - 1
        i, j = pi.item(), pj.item()
    return alignment


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Align bilingual subtitles and generate QC report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunk-size", type=int, default=400, help="embedding batch size")
    parser.add_argument("--stop-threshold", type=float, default=0.25, help="gap penalty")
    parser.add_argument("--max-gap", type=int, default=3, help="max alignment gap")
    parser.add_argument("--cps-threshold", type=float, default=17, help="CPS flag threshold")
    parser.add_argument("--similarity-flag", type=float, default=0.60, help="similarity flag threshold")
    parser.add_argument("--device", default="auto", help="cuda, mps or cpu")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        eng_path = Path("assets/ENG.srt")
        zh_path = Path("assets/ZH.txt")
        out_dir = Path("output")
        out_dir.mkdir(exist_ok=True)

        log("Loading inputs …", start)
        subs, zh_lines = load_inputs(eng_path, zh_path)
        en_lines = [cue.text.replace("\n", " ") for cue in subs]
        log("Segmenting text …", start)
        seg_en, seg_zh = segment_texts(en_lines, zh_lines, args.device)

        if args.device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            device = args.device
        log(f"Using device: {device}", start)
        model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

        log("Embedding English …", start)
        emb_en = embed_texts(seg_en, model, args.chunk_size, start)
        log("Embedding Chinese …", start)
        emb_zh = embed_texts(seg_zh, model, args.chunk_size, start)

        log("Computing similarity matrix …", start)
        sim_mat = cosine_similarity(emb_en.cpu().numpy(), emb_zh.cpu().numpy())
        sim_tensor = torch.tensor(sim_mat)

        log("Aligning …", start)
        mapping = align(sim_tensor, args.max_gap, args.stop_threshold)

        log("Generating outputs …", start)
        report_lines = []
        for idx, cue in enumerate(subs):
            zh_idx = mapping[idx]
            zh_line = zh_lines[zh_idx] if zh_idx >= 0 else ""
            duration = (cue.end - cue.start).total_seconds()
            cps = len(zh_line) / duration if duration > 0 else 0
            sim = sim_mat[idx, zh_idx] if zh_idx >= 0 else 0.0
            flags = []
            if cps > args.cps_threshold:
                flags.append("CPS_HIGH")
            if sim < args.similarity_flag:
                flags.append("LOW_SIM")
            flag_str = ",".join(flags) if flags else "-"
            report_lines.append(f"{idx+1} | {cps:.2f} | {sim:.2f} | {flag_str}")
            cue.text = f"{cue.text}\n{zh_line}"
        out_srt = out_dir / "bilingual.srt"
        subs.save(str(out_srt))
        out_qc = out_dir / "QC_report.txt"
        out_qc.write_text("\n".join(report_lines), encoding="utf-8")
        log("Done", start)
        return 0
    except MemoryError:
        print("MemoryError: Try reducing --chunk-size", file=sys.stderr)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
