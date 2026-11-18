"""Display saved development comparisons without fitting models at startup."""
import json
import math
from pathlib import Path

from backend.paths import PROJECT_ROOT


def score(row, key):
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'Invalid score: {key}')
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Score outside [0, 1]: {key}')
    return float(value)


def comparison_table(data):
    baseline = score(data['LogReg'], 'student_top1')
    lines = [
        'MODEL COMPARISON - saved student-development results',
        'Model              CV Top-1  Student Top-1  vs LogReg (pp)  Student macro F1',
        '-' * 78,
    ]
    for name, row in data.items():
        cv = score(row, 'cv_top1_mean')
        top1 = score(row, 'student_top1')
        f1 = score(row, 'student_macro_f1')
        lines.append(f'{name:<18} {cv:8.2%}  {top1:13.2%}  {(top1-baseline)*100:+14.2f}  {f1:16.2%}')
    lines.extend([
        'pp = percentage points. Negative values mean worse than the LogReg baseline.',
        'Selection uses training trip-grouped CV; student scores are reporting only.',
    ])
    return '\n'.join(lines)


def change(label, before, after):
    return f'{label}: {before:.2%} -> {after:.2%} ({(after-before)*100:+.2f} pp)'


def tuning_summary(data):
    before, after = data['untuned'], data['tuned']
    return '\n'.join([
        f"TUNING COMPARISON - {data['selected_model']} (saved results)",
        change('CV Top-1', score(before, 'cv_top1_mean'), score(after, 'cv_top1_mean')),
        change('Student Top-1', score(before, 'student_top1'), score(after, 'student_top1')),
        change('Student macro F1', score(before, 'student_macro_f1'), score(after, 'student_macro_f1')),
        f"Recorded release used tuned model: {data['deployed_tuned']}",
    ])


def sequence_summary(data):
    return '\n'.join([
        f"DECODER COMPARISON - classifier -> classifier + HMM ({data['trips']} student trips)",
        change('Proxy-label Top-1', score(data, 'classifier_top1'), score(data, 'hmm_top1')),
        change('Strict continuity', score(data, 'classifier_continuity'), score(data, 'hmm_continuity')),
        'HMM is a decoding stage, not another trained classifier. Accuracy uses route-derived proxy labels.',
    ])


def log_model_comparison(logger, model_version, directory=None):
    directory = Path(directory) if directory is not None else PROJECT_ROOT/'artifacts'/'experiments'
    logger.info('Serving model: %s. Comparison below is a saved experiment snapshot, not live evaluation or retraining.', model_version)
    for filename, formatter in (
        ('model_comparison.json', comparison_table),
        ('tuning.json', tuning_summary),
        ('sequence_summary.json', sequence_summary),
    ):
        try:
            data = json.loads((directory/filename).read_text(encoding='utf-8-sig'))
            message = formatter(data)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            # Optional presentation evidence must never prevent valid inference.
            logger.warning('Saved comparison unavailable (%s): %s', filename, error)
        else:
            logger.info('\n%s\n', message)


if __name__ == '__main__':
    import logging
    import sys
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    log_model_comparison(logging.getLogger('pathsnap'), 'offline preview; backend not started')
