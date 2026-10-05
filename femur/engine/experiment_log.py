"""Write completed-run research records without loading model checkpoints."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPORT = Path(__file__).resolve().parents[2] / 'research' / 'experiments.md'


def record_completed_run(output_dir, report_path=DEFAULT_REPORT):
    out = Path(output_dir)
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    split = json.loads((out / 'split.json').read_text(encoding='utf-8'))
    with (out / 'history.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[-1]['epoch']) != config['epochs']:
        raise ValueError('Only completed training runs can be recorded.')
    scored = [row for row in rows if row['validation_dice'].strip()]
    if not scored or not rows[-1]['validation_dice'].strip():
        raise ValueError('Completed run must include final validation Dice.')
    best = max(scored, key=lambda row: float(row['validation_dice']))
    final = rows[-1]
    # Replace this run's generated section when resumed or recorded again.
    key = hashlib.sha256(str(out.resolve()).encode('utf-8')).hexdigest()[:20]
    begin, end = f'<!-- experiment:{key}:start -->', f'<!-- experiment:{key}:end -->'
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    section = '\n'.join([
        begin, f'## Completed run: {out.name}', '',
        f'Recorded at {timestamp} (UTC). Run directory: `{out.as_posix()}`.', '',
        '| Metric | Result |', '|---|---|',
        f"| Best validation Dice | {best['validation_dice']} |",
        f"| Best epoch | {best['epoch']} |",
        f"| Final validation Dice | {final['validation_dice']} |",
        f"| Completed epochs | {final['epoch']} |",
        f"| Final training loss | {final['train_loss']} |", '',
        'Dice is mean foreground overlap across validation volumes on the resampled grid.',
        'Independent test Dice is not recorded. Best checkpoint: `best.pt`; final checkpoint: `last.pt`.', '',
        '### Configuration', '', '```json', json.dumps(config, indent=2), '```', '',
        '### Patient split and case paths', '', '```json', json.dumps(split, indent=2), '```', '',
        '### Validation history', '', '| Epoch | Training loss | Validation Dice |',
        '|---:|---:|---:|',
        *[f"| {r['epoch']} | {r['train_loss']} | {r['validation_dice']} |" for r in scored],
        '', end,
    ])
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    content = report.read_text(encoding='utf-8') if report.exists() else '# Experiment results\n'
    if begin in content:
        start = content.index(begin)
        stop = content.index(end, start) + len(end)
        content = content[:start] + section + content[stop:]
    else:
        content = content.rstrip() + '\n\n' + section + '\n'
    report.write_text(content, encoding='utf-8')
    return report
