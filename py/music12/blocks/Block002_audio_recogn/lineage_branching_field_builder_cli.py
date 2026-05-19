# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return float(s.replace(',', '.'))
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if not s:
            return default
        return int(float(s.replace(',', '.')))
    except Exception:
        return default


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _split_tokens(raw: Any) -> Set[str]:
    return {x.strip() for x in str(raw or '').replace(',', ' ').replace('|', ' ').split() if x.strip()}


def _join(tokens: Iterable[str], limit: int = 96) -> str:
    return ' '.join(sorted(set(tokens))[:limit])


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _lineage_tokens(row: Dict[str, Any]) -> Set[str]:
    toks = set()
    toks.update(_split_tokens(row.get('parent_tokens', '')))
    toks.update(_split_tokens(row.get('offspring_tokens', '')))
    return toks


def _box_tokens(row: Dict[str, Any]) -> Set[str]:
    toks = set()
    toks.update(_split_tokens(row.get('residual_box_tokens', '')))
    toks.update(_split_tokens(row.get('box_residual_signature', '')))
    return toks


def _time_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    a0 = _safe_int(a.get('birth_frame'), 0)
    a1 = _safe_int(a.get('end_frame'), a0)
    b0 = _safe_int(b.get('birth_frame'), 0)
    b1 = _safe_int(b.get('end_frame'), b0)
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _time_gap(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    a0 = _safe_int(a.get('birth_frame'), 0)
    a1 = _safe_int(a.get('end_frame'), a0)
    b0 = _safe_int(b.get('birth_frame'), 0)
    b1 = _safe_int(b.get('end_frame'), b0)
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _branch_score(a: Dict[str, Any], b: Dict[str, Any], max_gap: int) -> Dict[str, Any]:
    overlap = _time_overlap(a, b)
    gap = _time_gap(a, b)
    if gap > max_gap and overlap <= 0:
        return {'score': 0.0, 'kind': 'NO_BRANCH_RELATION'}

    ta = _lineage_tokens(a)
    tb = _lineage_tokens(b)
    ba = _box_tokens(a)
    bb = _box_tokens(b)

    token_shared = _jaccard(ta, tb)
    box_shared = _jaccard(ba, bb)

    ha = set(str(a.get('present_harmonics', '')).split())
    hb = set(str(b.get('present_harmonics', '')).split())
    harmonic_shared = _jaccard(ha, hb)

    h57_a = _safe_float(a.get('harmonic_5_7_parenthood_score'), 0.0)
    h57_b = _safe_float(b.get('harmonic_5_7_parenthood_score'), 0.0)
    h57_bridge = min(h57_a, h57_b)

    strength_a = _safe_float(a.get('lineage_strength'), 0.0)
    strength_b = _safe_float(b.get('lineage_strength'), 0.0)
    strength_bridge = min(strength_a, strength_b)

    duration_scale = max(_safe_int(a.get('duration_frames'), 1), _safe_int(b.get('duration_frames'), 1), 1)
    overlap_score = min(overlap / duration_scale, 1.0)
    gap_score = 1.0 - min(gap / max(max_gap, 1), 1.0)

    same_root = str(a.get('root_candidate', '')).strip() == str(b.get('root_candidate', '')).strip()
    same_root_bonus = 0.08 if same_root else 0.0

    score = _clamp(
        overlap_score * 0.20
        + gap_score * 0.12
        + token_shared * 0.20
        + box_shared * 0.14
        + harmonic_shared * 0.12
        + h57_bridge * 0.14
        + strength_bridge * 0.08
        + same_root_bonus
    )

    if score >= 0.62:
        kind = 'STRONG_BRANCH_COUPLING'
    elif score >= 0.42:
        kind = 'SUPPORTED_BRANCH_COUPLING'
    elif score >= 0.24:
        kind = 'WEAK_BRANCH_COUPLING'
    else:
        kind = 'MINIMAL_BRANCH_RELATION'

    if same_root and score >= 0.34:
        relation = 'SAME_ROOT_BRANCH'
    elif harmonic_shared >= 0.42 and h57_bridge >= 0.35:
        relation = 'SHARED_HARMONIC_OFFSPRING'
    elif box_shared >= 0.30:
        relation = 'SHARED_BOX_RESIDUAL_FIELD'
    elif overlap > 0:
        relation = 'TEMPORAL_OVERLAP_BRANCH'
    else:
        relation = 'DELAYED_BRANCH_NEIGHBOR'

    return {
        'score': score,
        'kind': kind,
        'relation': relation,
        'overlap_frames': overlap,
        'gap_frames': gap,
        'token_shared': token_shared,
        'box_shared': box_shared,
        'harmonic_shared': harmonic_shared,
        'h57_bridge': h57_bridge,
        'strength_bridge': strength_bridge,
        'same_root': same_root,
    }


def _node_status(energy: float, branch_count: int) -> str:
    if energy >= 5.0 and branch_count >= 5:
        return 'DENSE_POLYPHONIC_LINEAGE_NODE'
    if energy >= 2.0 and branch_count >= 3:
        return 'CONNECTED_LINEAGE_NODE'
    if branch_count >= 1:
        return 'SPARSE_LINEAGE_NODE'
    return 'ISOLATED_LINEAGE_NODE'


def main() -> None:
    ap = argparse.ArgumentParser(description='Build lineage branching field from virtual-string lineages.')
    ap.add_argument('--lineages_csv', required=True)
    ap.add_argument('--out_branch_links_csv', required=True)
    ap.add_argument('--out_branch_nodes_csv', required=True)
    ap.add_argument('--out_branch_frame_csv', required=True)
    ap.add_argument('--out_readable_csv', required=True)
    ap.add_argument('--out_meta_json', required=True)
    ap.add_argument('--out_summary_txt', required=True)
    ap.add_argument('--max_branch_gap_frames', type=int, default=120)
    ap.add_argument('--min_branch_score', type=float, default=0.24)
    ap.add_argument('--fps', type=float, default=60.0)
    args = ap.parse_args()

    rows = _load_csv(Path(args.lineages_csv))
    rows.sort(key=lambda r: (_safe_int(r.get('birth_frame'), 0), -_safe_float(r.get('lineage_strength'), 0.0)))

    links = []
    readable = []
    energy = defaultdict(float)
    branch_count = defaultdict(int)
    kind_counts = defaultdict(int)
    relation_counts = defaultdict(int)

    for i, a in enumerate(rows):
        a_id = str(a.get('lineage_id', '')).strip()
        a_end = _safe_int(a.get('end_frame'), 0)
        for j in range(i + 1, len(rows)):
            b = rows[j]
            b_birth = _safe_int(b.get('birth_frame'), 0)
            if b_birth - a_end > args.max_branch_gap_frames:
                break
            b_id = str(b.get('lineage_id', '')).strip()
            inter = _branch_score(a, b, args.max_branch_gap_frames)
            score = _safe_float(inter.get('score'), 0.0)
            if score < args.min_branch_score:
                continue
            kind = str(inter['kind'])
            relation = str(inter['relation'])
            links.append({
                'source_lineage_id': a_id,
                'target_lineage_id': b_id,
                'source_root': a.get('root_candidate', ''),
                'target_root': b.get('root_candidate', ''),
                'branch_kind': kind,
                'branch_relation': relation,
                'branch_score': f'{score:.9f}',
                'overlap_frames': inter['overlap_frames'],
                'gap_frames': inter['gap_frames'],
                'token_shared': f"{inter['token_shared']:.9f}",
                'box_shared': f"{inter['box_shared']:.9f}",
                'harmonic_shared': f"{inter['harmonic_shared']:.9f}",
                'h57_bridge': f"{inter['h57_bridge']:.9f}",
                'strength_bridge': f"{inter['strength_bridge']:.9f}",
                'same_root': int(bool(inter['same_root'])),
            })
            readable.append({
                'source': a_id,
                'target': b_id,
                'roots': f"{a.get('root_candidate', '')} ↔ {b.get('root_candidate', '')}",
                'kind': kind,
                'relation': relation,
                'score': f'{score:.3f}',
                'h57': f"{inter['h57_bridge']:.3f}",
                'box': f"{inter['box_shared']:.3f}",
            })
            energy[a_id] += score
            energy[b_id] += score
            branch_count[a_id] += 1
            branch_count[b_id] += 1
            kind_counts[kind] += 1
            relation_counts[relation] += 1

    node_rows = []
    frame_rows = []
    for r in rows:
        lid = str(r.get('lineage_id', '')).strip()
        e = energy[lid]
        bc = branch_count[lid]
        status = _node_status(e, bc)
        birth = _safe_int(r.get('birth_frame'), 0)
        end = _safe_int(r.get('end_frame'), birth)
        node_rows.append({
            'lineage_id': lid,
            'root_candidate': r.get('root_candidate', ''),
            'root_candidate_micro': r.get('root_candidate_micro', ''),
            'lineage_label': r.get('lineage_label', ''),
            'lineage_strength': r.get('lineage_strength', ''),
            'harmonic_5_7_parenthood_score': r.get('harmonic_5_7_parenthood_score', ''),
            'branch_energy': f'{e:.9f}',
            'branch_count': bc,
            'branch_node_status': status,
            'birth_frame': birth,
            'end_frame': end,
            'duration_frames': r.get('duration_frames', ''),
            'residual_box_count': r.get('residual_box_count', ''),
            'register_class': r.get('register_class', ''),
        })
        for frame in range(birth, end + 1):
            frame_rows.append({
                'frame_index': frame,
                'time_sec': f'{frame / max(args.fps, 1e-9):.9f}',
                'lineage_id': lid,
                'root_candidate': r.get('root_candidate', ''),
                'branch_energy': f'{e:.9f}',
                'branch_count': bc,
                'branch_node_status': status,
            })

    _write_csv(Path(args.out_branch_links_csv), links, [
        'source_lineage_id', 'target_lineage_id', 'source_root', 'target_root', 'branch_kind',
        'branch_relation', 'branch_score', 'overlap_frames', 'gap_frames', 'token_shared',
        'box_shared', 'harmonic_shared', 'h57_bridge', 'strength_bridge', 'same_root'
    ])
    _write_csv(Path(args.out_branch_nodes_csv), node_rows, [
        'lineage_id', 'root_candidate', 'root_candidate_micro', 'lineage_label', 'lineage_strength',
        'harmonic_5_7_parenthood_score', 'branch_energy', 'branch_count', 'branch_node_status',
        'birth_frame', 'end_frame', 'duration_frames', 'residual_box_count', 'register_class'
    ])
    _write_csv(Path(args.out_branch_frame_csv), frame_rows, [
        'frame_index', 'time_sec', 'lineage_id', 'root_candidate', 'branch_energy', 'branch_count', 'branch_node_status'
    ])
    _write_csv(Path(args.out_readable_csv), readable, ['source', 'target', 'roots', 'kind', 'relation', 'score', 'h57', 'box'])

    meta = {
        'stage': 'lineage_branching_field_builder',
        'semantic_version': 'lineage_branching_field_v1',
        'inputs': {'lineages_csv': args.lineages_csv},
        'outputs': {
            'branch_links_csv': args.out_branch_links_csv,
            'branch_nodes_csv': args.out_branch_nodes_csv,
            'branch_frame_csv': args.out_branch_frame_csv,
            'readable_csv': args.out_readable_csv,
            'meta_json': args.out_meta_json,
            'summary_txt': args.out_summary_txt,
        },
        'parameters': {
            'max_branch_gap_frames': args.max_branch_gap_frames,
            'min_branch_score': args.min_branch_score,
            'fps': args.fps,
        },
        'result': {
            'lineages_in': len(rows),
            'branch_links': len(links),
            'branch_nodes': len(node_rows),
            'frame_rows': len(frame_rows),
            'branch_kind_counts': dict(kind_counts),
            'branch_relation_counts': dict(relation_counts),
        },
        'ontology_note': (
            'This layer is the first polyphonic lineage ecology layer. It allows overlapping '
            'virtual-string lineages to share offspring, box residuals and harmonic evidence instead '
            'of assigning each child to exactly one parent.'
        ),
    }
    Path(args.out_meta_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta_json).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    txt = [
        'LINEAGE BRANCHING FIELD BUILDER',
        '=' * 72,
        f'lineages_csv      : {args.lineages_csv}',
        '',
        f'lineages_in       : {len(rows)}',
        f'branch_links      : {len(links)}',
        f'branch_nodes      : {len(node_rows)}',
        f'frame_rows        : {len(frame_rows)}',
        '',
        'Branch kind counts:',
    ]
    for k in sorted(kind_counts):
        txt.append(f'  {k}: {kind_counts[k]}')
    txt.append('')
    txt.append('Branch relation counts:')
    for k in sorted(relation_counts):
        txt.append(f'  {k}: {relation_counts[k]}')
    txt.extend([
        '',
        'Principle:',
        '  Polyphony is not merely simultaneous roots.',
        '  It is overlapping lineage ownership fields.',
        '  Harmonic offspring and residual body tokens may support multiple',
        '  virtual-string parents at once.',
        '',
    ])
    Path(args.out_summary_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary_txt).write_text('\n'.join(txt), encoding='utf-8')

    print('lineage branching field builder complete')
    print(json.dumps(meta['result'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
