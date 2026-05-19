# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


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


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _role_weight(role: str) -> float:
    return {
        'DOMINANT_LINEAGE_TERRITORY': 1.00,
        'SECONDARY_LINEAGE_TERRITORY': 0.72,
        'SYMPATHETIC_RESONANCE_TERRITORY': 0.48,
        'TRANSIENT_SHARED_TERRITORY': 0.34,
    }.get(str(role or '').strip(), 0.25)


def _boundary_weight(boundary: str) -> float:
    return {
        'CLEAR_TERRITORY_BOUNDARY': 1.00,
        'SOFT_TERRITORY_BOUNDARY': 0.72,
        'WEAK_TERRITORY_BOUNDARY': 0.46,
        'OPEN_SHARED_FIELD': 0.22,
    }.get(str(boundary or '').strip(), 0.30)


def _governance_energy(row: Dict[str, Any]) -> float:
    territory_score = _safe_float(row.get('territory_score'), 0.0)
    boundary_strength = _safe_float(row.get('territory_boundary_strength'), 0.0)
    role = str(row.get('territory_role', '')).strip()
    boundary = str(row.get('territory_boundary', '')).strip()

    return _clamp(
        territory_score * 0.44
        + boundary_strength * 0.24
        + _role_weight(role) * 0.22
        + _boundary_weight(boundary) * 0.10
    )


def _phase_from_energy(*, energy: float, previous_energy: float, is_top: bool, has_competition: bool) -> str:
    delta = energy - previous_energy

    if energy >= 0.70 and is_top:
        return 'DOMINANCE'
    if delta >= 0.08:
        return 'GROWTH'
    if delta <= -0.10 and energy >= 0.30:
        return 'DECAY'
    if energy < 0.24 and previous_energy >= 0.30:
        return 'RESIDUAL_CONTINUATION'
    if has_competition and energy >= 0.34:
        return 'COMPETITION'
    if energy >= 0.28:
        return 'SUSTAIN'
    return 'LOW_FIELD_PRESENCE'


def _event_kind(prev_top: str, new_top: str, gap: float) -> str:
    if not prev_top and new_top:
        return 'GOVERNANCE_BIRTH'
    if prev_top and new_top and prev_top != new_top:
        if gap >= 0.12:
            return 'DOMINANCE_TAKEOVER'
        return 'SOFT_HANDOVER'
    if prev_top and not new_top:
        return 'GOVERNANCE_DISSOLVE'
    return ''


def _lineage_root(row: Dict[str, Any]) -> str:
    return str(row.get('root_candidate', '')).strip()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            'Track temporal governance of lineage territories. '
            'This layer determines which lineage governs each frame, which lineages compete, '
            'and where dominance shifts, decay and residual continuation happen.'
        )
    )

    ap.add_argument('--territory_nodes_csv', required=True)
    ap.add_argument('--territory_frame_csv', required=True)
    ap.add_argument('--territory_links_csv', required=True)

    ap.add_argument('--out_governance_frame_csv', required=True)
    ap.add_argument('--out_governance_events_csv', required=True)
    ap.add_argument('--out_lineage_lifecycle_csv', required=True)
    ap.add_argument('--out_readable_csv', required=True)
    ap.add_argument('--out_meta_json', required=True)
    ap.add_argument('--out_summary_txt', required=True)

    ap.add_argument('--dominance_threshold', type=float, default=0.42)
    ap.add_argument('--competition_gap', type=float, default=0.10)
    ap.add_argument('--fps', type=float, default=60.0)

    args = ap.parse_args()

    nodes = _load_csv(Path(args.territory_nodes_csv))
    frames = _load_csv(Path(args.territory_frame_csv))
    links = _load_csv(Path(args.territory_links_csv))

    node_by_id = {str(r.get('lineage_id', '')).strip(): r for r in nodes}

    frames_by_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in frames:
        frames_by_index[_safe_int(r.get('frame_index'), 0)].append(r)

    link_neighbors: Dict[str, Set[str]] = defaultdict(set)
    for e in links:
        s = str(e.get('source_lineage_id', '')).strip()
        t = str(e.get('target_lineage_id', '')).strip()
        if s and t:
            link_neighbors[s].add(t)
            link_neighbors[t].add(s)

    governance_rows = []
    event_rows = []
    readable_rows = []

    prev_energy_by_lineage: Dict[str, float] = defaultdict(float)
    phase_by_lineage: Dict[str, List[str]] = defaultdict(list)
    energy_by_lineage: Dict[str, List[float]] = defaultdict(list)
    active_frames_by_lineage: Dict[str, List[int]] = defaultdict(list)

    previous_top = ''

    phase_counts = defaultdict(int)
    event_counts = defaultdict(int)
    top_counts = defaultdict(int)

    for frame in sorted(frames_by_index):
        rows = frames_by_index[frame]

        scored = []
        for r in rows:
            lid = str(r.get('lineage_id', '')).strip()
            node = node_by_id.get(lid, {})
            merged = dict(node)
            merged.update(r)

            energy = _governance_energy(merged)
            if energy <= 0.0:
                continue
            scored.append((lid, energy, merged))

        scored.sort(key=lambda x: (-x[1], x[0]))

        if scored:
            top_id, top_energy, _top_row = scored[0]
            second_energy = scored[1][1] if len(scored) > 1 else 0.0
            dominance_gap = top_energy - second_energy
        else:
            top_id, top_energy, second_energy, dominance_gap = '', 0.0, 0.0, 0.0

        competing = [lid for lid, energy, _r in scored[1:] if top_energy - energy <= args.competition_gap]
        active_lineages = [lid for lid, energy, _r in scored if energy >= args.dominance_threshold * 0.50]

        event = _event_kind(previous_top, top_id, dominance_gap)
        if event:
            event_counts[event] += 1
            event_rows.append({
                'frame_index': frame,
                'time_sec': f'{frame / max(args.fps, 1e-9):.9f}',
                'event_kind': event,
                'previous_governor': previous_top,
                'new_governor': top_id,
                'previous_root': _lineage_root(node_by_id.get(previous_top, {})),
                'new_root': _lineage_root(node_by_id.get(top_id, {})),
                'dominance_gap': f'{dominance_gap:.9f}',
                'top_energy': f'{top_energy:.9f}',
                'second_energy': f'{second_energy:.9f}',
            })

        for lid, energy, _merged in scored:
            prev_e = prev_energy_by_lineage[lid]
            is_top = lid == top_id
            has_comp = lid in competing or (is_top and bool(competing))

            phase = _phase_from_energy(
                energy=energy,
                previous_energy=prev_e,
                is_top=is_top,
                has_competition=has_comp,
            )

            phase_counts[phase] += 1
            phase_by_lineage[lid].append(phase)
            energy_by_lineage[lid].append(energy)
            active_frames_by_lineage[lid].append(frame)
            prev_energy_by_lineage[lid] = energy

        if top_id:
            top_counts[top_id] += 1

        if not top_id:
            governance_state = 'NO_ACTIVE_GOVERNANCE'
        elif dominance_gap >= 0.18:
            governance_state = 'CLEAR_DOMINANT_GOVERNANCE'
        elif competing:
            governance_state = 'CONTESTED_GOVERNANCE'
        else:
            governance_state = 'SOFT_DOMINANT_GOVERNANCE'

        governance_rows.append({
            'frame_index': frame,
            'time_sec': f'{frame / max(args.fps, 1e-9):.9f}',
            'governance_state': governance_state,
            'dominant_lineage_id': top_id,
            'dominant_root': _lineage_root(node_by_id.get(top_id, {})),
            'dominant_energy': f'{top_energy:.9f}',
            'second_energy': f'{second_energy:.9f}',
            'dominance_gap': f'{dominance_gap:.9f}',
            'active_lineage_count': len(active_lineages),
            'competing_lineage_count': len(competing),
            'competing_lineages': ' '.join(competing[:24]),
            'active_lineages': ' '.join(active_lineages[:32]),
            'event_kind': event,
        })

        readable_rows.append({
            'frame': frame,
            'time_sec': f'{frame / max(args.fps, 1e-9):.3f}',
            'state': governance_state,
            'root': _lineage_root(node_by_id.get(top_id, {})),
            'governor': top_id,
            'energy': f'{top_energy:.3f}',
            'gap': f'{dominance_gap:.3f}',
            'competing': len(competing),
            'event': event,
        })

        previous_top = top_id

    lifecycle_rows = []

    for lid, node in node_by_id.items():
        frames_list = active_frames_by_lineage.get(lid, [])
        energies = energy_by_lineage.get(lid, [])
        phases = phase_by_lineage.get(lid, [])

        if not frames_list:
            continue

        phase_hist = defaultdict(int)
        for p in phases:
            phase_hist[p] += 1

        dominant_frames = top_counts.get(lid, 0)

        if dominant_frames >= max(len(frames_list) * 0.45, 1):
            lifecycle = 'DOMINANT_LIFECYCLE'
        elif phase_hist.get('RESIDUAL_CONTINUATION', 0) >= max(len(frames_list) * 0.30, 1):
            lifecycle = 'RESIDUAL_TAIL_LIFECYCLE'
        elif phase_hist.get('COMPETITION', 0) >= max(len(frames_list) * 0.30, 1):
            lifecycle = 'COMPETING_LIFECYCLE'
        elif phase_hist.get('DECAY', 0) > phase_hist.get('GROWTH', 0):
            lifecycle = 'DECAYING_LIFECYCLE'
        else:
            lifecycle = 'SUPPORTING_LIFECYCLE'

        lifecycle_rows.append({
            'lineage_id': lid,
            'root_candidate': node.get('root_candidate', ''),
            'territory_role': node.get('territory_role', ''),
            'territory_boundary': node.get('territory_boundary', ''),
            'lifecycle_class': lifecycle,
            'first_active_frame': min(frames_list),
            'last_active_frame': max(frames_list),
            'active_frame_count': len(frames_list),
            'dominant_frame_count': dominant_frames,
            'mean_governance_energy': f'{_mean(energies):.9f}',
            'max_governance_energy': f'{max(energies) if energies else 0.0:.9f}',
            'phase_counts_json': json.dumps(dict(phase_hist), ensure_ascii=False, sort_keys=True),
            'neighbor_count': len(link_neighbors.get(lid, set())),
            'neighbors': ' '.join(sorted(link_neighbors.get(lid, set()))[:48]),
        })

    out_frame = Path(args.out_governance_frame_csv)
    out_events = Path(args.out_governance_events_csv)
    out_lifecycle = Path(args.out_lineage_lifecycle_csv)
    out_readable = Path(args.out_readable_csv)
    out_meta = Path(args.out_meta_json)
    out_txt = Path(args.out_summary_txt)

    frame_fields = [
        'frame_index', 'time_sec', 'governance_state',
        'dominant_lineage_id', 'dominant_root', 'dominant_energy',
        'second_energy', 'dominance_gap', 'active_lineage_count',
        'competing_lineage_count', 'competing_lineages', 'active_lineages',
        'event_kind',
    ]
    _write_csv(out_frame, governance_rows, frame_fields)

    event_fields = [
        'frame_index', 'time_sec', 'event_kind', 'previous_governor',
        'new_governor', 'previous_root', 'new_root', 'dominance_gap',
        'top_energy', 'second_energy',
    ]
    _write_csv(out_events, event_rows, event_fields)

    lifecycle_fields = [
        'lineage_id', 'root_candidate', 'territory_role', 'territory_boundary',
        'lifecycle_class', 'first_active_frame', 'last_active_frame',
        'active_frame_count', 'dominant_frame_count', 'mean_governance_energy',
        'max_governance_energy', 'phase_counts_json', 'neighbor_count', 'neighbors',
    ]
    _write_csv(out_lifecycle, lifecycle_rows, lifecycle_fields)

    readable_fields = ['frame', 'time_sec', 'state', 'root', 'governor', 'energy', 'gap', 'competing', 'event']
    _write_csv(out_readable, readable_rows, readable_fields)

    governance_state_counts = defaultdict(int)
    for r in governance_rows:
        governance_state_counts[r['governance_state']] += 1

    lifecycle_counts = defaultdict(int)
    for r in lifecycle_rows:
        lifecycle_counts[r['lifecycle_class']] += 1

    meta = {
        'stage': 'temporal_lineage_governor',
        'semantic_version': 'temporal_lineage_governor_v1',
        'inputs': {
            'territory_nodes_csv': args.territory_nodes_csv,
            'territory_frame_csv': args.territory_frame_csv,
            'territory_links_csv': args.territory_links_csv,
        },
        'outputs': {
            'governance_frame_csv': args.out_governance_frame_csv,
            'governance_events_csv': args.out_governance_events_csv,
            'lineage_lifecycle_csv': args.out_lineage_lifecycle_csv,
            'readable_csv': args.out_readable_csv,
            'meta_json': args.out_meta_json,
            'summary_txt': args.out_summary_txt,
        },
        'parameters': {
            'dominance_threshold': args.dominance_threshold,
            'competition_gap': args.competition_gap,
            'fps': args.fps,
        },
        'result': {
            'territory_nodes': len(nodes),
            'territory_frame_rows': len(frames),
            'governance_frames': len(governance_rows),
            'governance_events': len(event_rows),
            'lineage_lifecycles': len(lifecycle_rows),
            'governance_state_counts': dict(governance_state_counts),
            'event_counts': dict(event_counts),
            'phase_counts': dict(phase_counts),
            'lifecycle_counts': dict(lifecycle_counts),
        },
        'ontology_note': (
            'This layer tracks the life of lineage governance over time. '
            'A note does not disappear when dominance ends; it can decay, compete, '
            'or continue as residual resonance. This is temporal governance, not peak picking.'
        ),
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    txt = [
        'TEMPORAL LINEAGE GOVERNOR',
        '=' * 72,
        f'territory_nodes_csv : {args.territory_nodes_csv}',
        f'territory_frame_csv : {args.territory_frame_csv}',
        f'territory_links_csv : {args.territory_links_csv}',
        '',
        f'territory_nodes     : {len(nodes)}',
        f'territory_frames    : {len(frames)}',
        f'governance_frames   : {len(governance_rows)}',
        f'governance_events   : {len(event_rows)}',
        f'lineage_lifecycles  : {len(lifecycle_rows)}',
        '',
        'Governance state counts:',
    ]

    for k in sorted(governance_state_counts):
        txt.append(f'  {k}: {governance_state_counts[k]}')

    txt.append('')
    txt.append('Event counts:')
    for k in sorted(event_counts):
        txt.append(f'  {k}: {event_counts[k]}')

    txt.append('')
    txt.append('Phase counts:')
    for k in sorted(phase_counts):
        txt.append(f'  {k}: {phase_counts[k]}')

    txt.append('')
    txt.append('Lifecycle counts:')
    for k in sorted(lifecycle_counts):
        txt.append(f'  {k}: {lifecycle_counts[k]}')

    txt.extend([
        '',
        'Principle:',
        '  Territory is not enough; music is governance through time.',
        '  This layer tracks birth, growth, dominance, handover, decay,',
        '  competition and residual continuation.',
        '  Note off is not resonance death.',
        '',
    ])

    out_txt.write_text('\n'.join(txt), encoding='utf-8')

    print('temporal lineage governor complete')
    print(json.dumps(meta['result'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
