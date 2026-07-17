"""Load all thesis evaluation data from comparison report files.

Each parser reads a specific report.txt or CSV and returns clean Python
dicts/lists. load_all() aggregates everything under one dict keyed by
dataset name; figure functions consume only the keys they need.
"""
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_all(project_root: Path) -> dict:
    comp = project_root / 'comparisons'
    data: dict = {}

    _try(data, 'metrics',      _load_metrics_csv,    comp / '2026-07-17_confidence_intervals' / 'metrics.csv')
    _try(data, ('ci', 'wilcoxon'), _load_ci_report,  comp / '2026-07-17_confidence_intervals' / 'report.txt')
    _try(data, ('compression', 'per_dim_h'), _load_compression, comp / '2026-06-30_compression_analysis' / 'report.txt')
    _try(data, 'rd',           _load_rd_sweep,        comp / '2026-07-01_rd_sweep' / 'report.txt')
    _try(data, 'multi_coder',  _load_multi_coder,     comp / '2026-07-10_multi_coder' / 'report.txt')
    _try(data, 'ood',          _load_ood,             comp / '2026-07-01_ood_eval' / 'report.txt')
    _try(data, 'speaker_probe',_load_speaker_probe,   comp / '2026-07-01_speaker_probe' / 'report.txt')
    _try(data, 'corruption',   _load_corruption,      comp / '2026-07-01_corruption_test' / 'report.txt')
    _try(data, 'complexity',   _load_complexity,      comp / '2026-07-18_complexity_latency' / 'report.txt')
    _try(data, 'vctk',         _load_second_dataset,  comp / '2026-07-10_second_dataset' / 'report.txt')
    # eval_music.py saves to comparisons/<date>_music_eval/metrics.csv — find latest
    music_dirs = sorted(comp.glob('*_music_eval'), reverse=True)
    if music_dirs:
        _try(data, 'music', _load_metrics_csv, music_dirs[0] / 'metrics.csv')

    return data


def _try(data: dict, keys, fn, path: Path) -> None:
    if not path.exists():
        return
    result = fn(path)
    if isinstance(keys, tuple):
        for k, v in zip(keys, result):
            data[k] = v
    else:
        data[keys] = result


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _load_metrics_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---- CI report ------------------------------------------------------------

_RE_CI_ROW = re.compile(
    r'^\s+(\S+)\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+([\d.]+)k'
)
_RE_WX_ROW = re.compile(
    r'^\s+(.+?)\s{2,}(PESQ-WB|STOI)\s+([\d.]+)\s+(<0\.0001|[\d.]+)\s+(\*{1,3}|ns)'
)


def _load_ci_report(path: Path):
    ci: dict = {}
    wilcoxon: list = []
    mode = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if 'PHASE MEANS WITH 95%' in line:
            mode = 'means'
        elif 'WILCOXON SIGNED-RANK' in line:
            mode = 'wilcox'
        elif 'PER-SPEAKER RAW DATA' in line:
            mode = None
        if mode == 'means':
            m = _RE_CI_ROW.match(line)
            if m:
                phase, pm, plo, phi, sm, slo, shi, kbps = m.groups()
                ci[phase] = dict(
                    pesq=float(pm), pesq_lo=float(plo), pesq_hi=float(phi),
                    stoi=float(sm), stoi_lo=float(slo), stoi_hi=float(shi),
                    kbps=float(kbps),
                )
        elif mode == 'wilcox':
            m = _RE_WX_ROW.match(line)
            if m:
                contrast, metric, wstat, pval, sig = m.groups()
                wilcoxon.append(dict(
                    contrast=contrast.strip(), metric=metric,
                    w_stat=float(wstat), p_value=pval, sig=sig,
                ))
    return ci, wilcoxon


# ---- Compression analysis -------------------------------------------------

_RE_COMP_ROW = re.compile(
    r'^\s+phase(\w+)\s+([\d.]+)x\s+([\d.]+)k\s+([\d.]+)k\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
)


def _load_compression(path: Path):
    comp: dict = {}
    per_dim: dict = {}
    phase_cols: list = []
    in_per_dim = False
    for line in path.read_text(encoding='utf-8').splitlines():
        m = _RE_COMP_ROW.match(line)
        if m:
            raw = m.group(1)
            phase = 'D-VAE' if raw.lower() == 'dvae' else raw.upper()
            if len(phase) == 1:
                pass  # C, D, E, F, G
            comp[phase] = dict(
                ratio=float(m.group(2)), eff_kbps=float(m.group(3)),
                theo_kbps=float(m.group(4)), mean_h=float(m.group(5)),
                min_h=float(m.group(6)), max_h=float(m.group(7)),
            )
            continue
        if line.startswith('Dim'):
            tokens = line.split()
            phase_cols = []
            for t in tokens[1:]:
                name = t.replace('phase', '')
                name = 'D-VAE' if name.lower() == 'dvae' else name.upper()
                phase_cols.append(name)
            per_dim = {p: [] for p in phase_cols}
            in_per_dim = True
            continue
        if in_per_dim:
            parts = line.split()
            if len(parts) == len(phase_cols) + 1:
                try:
                    int(parts[0])
                    for p, v in zip(phase_cols, parts[1:]):
                        per_dim[p].append(float(v))
                except ValueError:
                    pass
    return comp, per_dim


# ---- R-D sweep ------------------------------------------------------------

_RE_RD_OURS = re.compile(
    r'^\s+(\d+)\s+(\d+)\s+([\d.]+)k\s+([\d.]+)k\s+([\d.]+)\s+([\d.]+)'
)
_RE_RD_ENC = re.compile(
    r'^\s+EnCodec\s+([\d.]+)\s+kbps\s+([\d.]+)k\s+([\d.]+)\s+([\d.]+)'
)


def _load_rd_sweep(path: Path):
    ours: list = []
    encodec: list = []
    mode = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if 'OURS (scalar' in line:
            mode = 'ours'
        elif 'ENCODEC REFERENCE' in line:
            mode = 'enc'
        if mode == 'ours':
            m = _RE_RD_OURS.match(line)
            if m:
                ours.append(dict(
                    bits=int(m.group(1)), levels=int(m.group(2)),
                    theo_kbps=float(m.group(3)), eff_kbps=float(m.group(4)),
                    pesq=float(m.group(5)), stoi=float(m.group(6)),
                ))
        elif mode == 'enc':
            m = _RE_RD_ENC.match(line)
            if m:
                encodec.append(dict(
                    kbps=float(m.group(1)), eff_kbps=float(m.group(2)),
                    pesq=float(m.group(3)), stoi=float(m.group(4)),
                ))
    return dict(ours=ours, encodec=encodec)


# ---- Multi-coder ----------------------------------------------------------

_RE_MC_ROW = re.compile(
    r'^\s+(\S+)\s+([\d.]+)[×x]\s+([\d.]+)[×x]\s+([\d.]+)[×x]\s+([\d.]+)'
)


def _load_multi_coder(path: Path):
    rows: list = []
    for line in path.read_text(encoding='utf-8').splitlines():
        m = _RE_MC_ROW.match(line)
        if m:
            rows.append(dict(
                phase=m.group(1),
                zlib=float(m.group(2)),
                lzma=float(m.group(3)),
                bz2=float(m.group(4)),
                mean_h=float(m.group(5)),
            ))
    return rows


# ---- OOD evaluation -------------------------------------------------------

_RE_OOD_ROW = re.compile(
    r'^\s+(.+?)\s{2,}([\d.]+)k\s+(\S+)\s+([-\d.]+)'
)


def _load_ood(path: Path):
    rows: list = []
    in_table = False
    for line in path.read_text(encoding='utf-8').splitlines():
        if 'Category' in line and 'kbps' in line:
            in_table = True
            continue
        if in_table:
            m = _RE_OOD_ROW.match(line)
            if m:
                label, kbps, pesq_raw, stoi = m.groups()
                rows.append(dict(
                    label=label.strip(),
                    kbps=float(kbps),
                    pesq=None if pesq_raw == 'n/a' else float(pesq_raw),
                    stoi=float(stoi),
                    is_speech='speech' in label.lower() or 'clean' in label.lower(),
                ))
    return rows


# ---- Speaker probe --------------------------------------------------------

_RE_PROBE_SPK = re.compile(r'^\s+(\d+)\s+([\d.]+)%\s+(\d+)')
_RE_PROBE_ACC = re.compile(r'Logistic regression accuracy\s*:\s*([\d.]+)%')


def _load_speaker_probe(path: Path):
    accuracy = None
    per_speaker: list = []
    for line in path.read_text(encoding='utf-8').splitlines():
        m = _RE_PROBE_ACC.search(line)
        if m:
            accuracy = float(m.group(1))
        m = _RE_PROBE_SPK.match(line)
        if m:
            per_speaker.append(dict(
                speaker=m.group(1),
                recall=float(m.group(2)),
                n_test=int(m.group(3)),
            ))
    return dict(accuracy=accuracy, per_speaker=per_speaker)


# ---- Corruption test ------------------------------------------------------

_RE_CORRUPT_ROW = re.compile(r'^\s+([\d.]+)\s+([\d.]+)%')


def _load_corruption(path: Path):
    rows: list = []
    for line in path.read_text(encoding='utf-8').splitlines():
        m = _RE_CORRUPT_ROW.match(line)
        if m:
            rows.append(dict(rate=float(m.group(1)), success=float(m.group(2))))
    return rows


# ---- Complexity / latency --------------------------------------------------

def _load_complexity(path: Path):
    text = path.read_text(encoding='utf-8')
    num = lambda s: float(s.replace(',', ''))

    ours_block = re.search(r'=== Ours.*?===\n(.*?)\n\n=== EnCodec', text, re.S).group(1)
    enc_block = re.search(r'=== EnCodec.*?===\n(.*?)\n\nTotal params', text, re.S).group(1)

    def parse_block(block):
        d = {}
        d['params'] = num(re.search(r'total\s+([\d,]+)', block).group(1))
        d['macs'] = num(re.search(r'MACs \(1s clip, full fwd\)\s+([\d,]+)', block).group(1))
        d['encode_ms'] = float(re.search(r'encode latency \(CPU\)\s+([\d.]+)\s*ms', block).group(1))
        d['decode_ms'] = float(re.search(r'decode latency \(CPU\)\s+([\d.]+)\s*ms', block).group(1))
        return d

    ours, encodec = parse_block(ours_block), parse_block(enc_block)
    ours['delay_ms'] = float(re.search(r'algorithmic delay.*?=\s*([\d.]+)\s*ms', text).group(1))
    ours['chunk_ms'] = float(re.search(r'default chunking:\s*([\d.]+)\s*ms', text).group(1))
    encodec['delay_ms'] = float(re.search(r'hop = ([\d.]+)\s*ms\)', text).group(1))

    return dict(ours=ours, encodec=encodec)


# ---- Second dataset (VCTK OOD generalization) ------------------------------

_RE_VCTK_COMP_ROW = re.compile(
    r'^\s+(\S+)\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+([\d.]+)k'
)
_RE_VCTK_QUAL_ROW = re.compile(
    r'^\s+(\S+)\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s*$'
)
_RE_VCTK_WX_HEAD = re.compile(
    r'^\s+(\S+\s+vs\s+\S+)\s+(PESQ-WB|STOI|zlib ratio)\s+([\d.]+)\s+(<0\.0001|[\d.]+)\s+(\*{1,3}|ns)'
)
_RE_VCTK_WX_CONT = re.compile(
    r'^\s{10,}(PESQ-WB|STOI|zlib ratio)\s+([\d.]+)\s+(<0\.0001|[\d.]+)\s+(\*{1,3}|ns)'
)


def _load_second_dataset(path: Path):
    compression: dict = {}
    quality: dict = {}
    wilcoxon: list = []
    mode = None
    last_contrast = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if 'COMPRESSION RATIO' in line:
            mode = 'comp'
        elif 'QUALITY METRICS' in line:
            mode = 'qual'
        elif 'WILCOXON SIGNED-RANK' in line:
            mode = 'wilcox'

        if mode == 'comp':
            m = _RE_VCTK_COMP_ROW.match(line)
            if m:
                phase, zm, zlo, zhi, hm, hlo, hhi, kbps = m.groups()
                compression[phase] = dict(
                    ratio=float(zm), ratio_lo=float(zlo), ratio_hi=float(zhi),
                    mean_h=float(hm), h_lo=float(hlo), h_hi=float(hhi),
                    kbps=float(kbps),
                )
        elif mode == 'qual':
            m = _RE_VCTK_QUAL_ROW.match(line)
            if m:
                phase, pm, plo, phi, sm, slo, shi = m.groups()
                quality[phase] = dict(
                    pesq=float(pm), pesq_lo=float(plo), pesq_hi=float(phi),
                    stoi=float(sm), stoi_lo=float(slo), stoi_hi=float(shi),
                )
        elif mode == 'wilcox':
            m = _RE_VCTK_WX_HEAD.match(line)
            if m:
                contrast, metric, wstat, pval, sig = m.groups()
                last_contrast = contrast.strip()
                wilcoxon.append(dict(contrast=last_contrast, metric=metric,
                                      w_stat=float(wstat), p_value=pval, sig=sig))
                continue
            m = _RE_VCTK_WX_CONT.match(line)
            if m and last_contrast:
                metric, wstat, pval, sig = m.groups()
                wilcoxon.append(dict(contrast=last_contrast, metric=metric,
                                      w_stat=float(wstat), p_value=pval, sig=sig))

    return dict(compression=compression, quality=quality, wilcoxon=wilcoxon)
