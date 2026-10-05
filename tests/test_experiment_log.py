"""Research logging checks using synthetic run artifacts; no GPU required."""
import json
import tempfile
import unittest
from pathlib import Path

from femur.engine.experiment_log import record_completed_run


class ExperimentLogTests(unittest.TestCase):
    def test_completed_resumed_and_separate_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / 'research' / 'experiments.md'
            report.parent.mkdir()
            report.write_text('# Existing research\nKeep this note.\n', encoding='utf-8')
            run = root / 'run1'
            run.mkdir()
            (run / 'config.json').write_text(json.dumps({'epochs': 3, 'feature_size': 24}))
            (run / 'split.json').write_text(json.dumps({'train': ['A'], 'val': ['B']}))
            history = run / 'history.csv'
            history.write_text('epoch,train_loss,validation_dice\n1,1,\n2,0.5,0.9\n3,0.4,0.8\n')
            record_completed_run(run, report)
            record_completed_run(run, report)
            text = report.read_text(encoding='utf-8')
            self.assertIn('Keep this note.', text)
            self.assertEqual(text.count('## Completed run: run1'), 1)
            self.assertIn('| Best validation Dice | 0.9 |', text)
            self.assertIn('| Best epoch | 2 |', text)
            self.assertIn('| Final validation Dice | 0.8 |', text)
            self.assertIn('"feature_size": 24', text)
            (run / 'config.json').write_text(json.dumps({'epochs': 4, 'feature_size': 24}))
            with history.open('a') as handle:
                handle.write('4,0.3,0.95\n')
            record_completed_run(run, report)
            text = report.read_text(encoding='utf-8')
            self.assertEqual(text.count('## Completed run: run1'), 1)
            self.assertIn('| Best epoch | 4 |', text)
            other = root / 'run2'
            other.mkdir()
            for name in ('config.json', 'split.json', 'history.csv'):
                (other / name).write_bytes((run / name).read_bytes())
            record_completed_run(other, report)
            self.assertIn('## Completed run: run2', report.read_text(encoding='utf-8'))

    def test_incomplete_run_does_not_write_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{"epochs": 2}')
            (root / 'split.json').write_text('{}')
            (root / 'history.csv').write_text('epoch,train_loss,validation_dice\n1,1,0.5\n')
            report = root / 'experiments.md'
            with self.assertRaises(ValueError):
                record_completed_run(root, report)
            self.assertFalse(report.exists())


if __name__ == '__main__':
    unittest.main()
