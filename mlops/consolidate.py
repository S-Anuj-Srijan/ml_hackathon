"""Combine measured technical results and the maintained operations appendix."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
MARKER = '<!-- operations-start -->'


def consolidate(technical):
    target = ROOT / 'REPORT.md'
    if target.exists() and MARKER in target.read_text(encoding='utf-8'):
        operations = target.read_text(encoding='utf-8').split(MARKER, 1)[1].strip()
    else:
        operations = (ROOT / 'ml ops.md').read_text(encoding='utf-8')
        if len(operations) < 1000:
            raise ValueError('The complete operations appendix is missing')
        operations = re.sub(r'^# PathSnap.*\n', '', operations, count=1).strip()
    technical = technical.replace('../artifacts/', 'artifacts/').replace('../reports/', 'reports/')
    technical = re.sub(r'^# PathSnap Technical Report\n', '', technical, count=1).strip()
    # Keep a single top-level title, with technical sections below it.
    technical = re.sub(r'^(#{1,5}) ', r'#\1 ', technical, flags=re.MULTILINE)
    header = '''# PathSnap — Consolidated Project Report

This is the single submission report for deliverables A–D, measured results and
MLOps operations. README files provide quick-start instructions. Supporting
JSON/CSV files and plots contain the underlying evidence.

The frozen model is `HistGB-8623248d7844`. No retraining is required for the demo.
The results below use student-development data, not the reserved evaluator set.

## Contributor credits

These roles are supported by commits in the repository history, including the
remote main branch. Names are listed by deliverable, without ranking contributors
by commit count. Commit/push counts are not measurements of equal effort.

| Contributor recorded in Git | Documented contribution |
|---|---|
| S-Anuj-Srijan / anujs | Part A data audit, route labels, candidate features, EDA and project integration |
| ryonahh | Part B model zoo, trip-grouped CV, feature experiments, tuning and model export |
| Rufus15-06-2006 / RT | Part C dataset inspection, route alignment and HMM/Viterbi decoder on main |
| Haritej / RHaritej | Part D FastAPI, UI, two-laptop setup and later UI/model telemetry integration on main |

The current release also includes Codex-assisted implementation and documentation.
These credits preserve the recorded history; no equal contribution totals,
fabricated commits, backdated commits or reassigned authorship are claimed.
Newer main-branch enhancements remain on main; this release branch uses the
verified split frontend/backend implementation described below.

## Technical results: data, models and decoding

'''
    target.write_text(header + technical + '\n\n## Complete MLOps implementation and runbook\n\n'
                      + MARKER + '\n\n' + operations + '\n', encoding='utf-8')
    return target


if __name__ == '__main__':
    print(consolidate((ROOT/'reports/technical_report.md').read_text(encoding='utf-8')))
