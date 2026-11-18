# Part A: short data-quality notes

Full methods, reproduction instructions, feature definitions and handoff:
[A_PHASE1_README.md](A_PHASE1_README.md).

- Scope: 80 training and 10 student-test trips; 227,577 observations and 360
  verified source files. No evaluator data was used.
- Formats: `.track` has longitude/latitude/timestamp; `.nodes` has
  longitude/latitude; `.arcs` has two node IDs; `.route` is a sequence of arc IDs.
  No nonfinite values, out-of-range coordinates or non-increasing timestamps
  were found. Source hashes and parsed dtypes are in `file_inventory.csv`.
- Sampling: 96.68% of steps are 1 second. There are 617 gaps over 10 seconds
  across 42 trips; the maximum is 153,064 seconds. Gaps are flagged, not filled.
- Graphs: all 90 networks differ; no isolated nodes, 106 zero-length arcs,
  and 14 disconnected reference-arc transitions across 11 routes.
- Labels are inferred using monotone route-occurrence alignment plus a weak
  elapsed-time progress prior. No per-observation truth timestamps are supplied.
  GPS/reference disagreement, junctions, loops and gaps limit label certainty.
- Distances use each network's UTM zone. Candidates are the nearest 32 directed
  segments within 300m, independently of labels; reference arcs are never forced
  into the candidate set. Training misses improved from 10.71% at K=16/100m
  to 4.0117%; student misses are 0%. Neither split has zero-candidate points.
- The final table has 6,172,468 rows and nine finite numeric features. Undefined
  speed/heading uses -1. No learned preprocessing is fitted globally.
- Keep reference arc IDs, residuals, route occurrences and hit flags out of
  model inputs. Group CV by whole trip. Use the observation table for complete
  evaluation denominators, including empty candidate sets if settings change.
- Candidate miss rates measure coverage against inferred labels, not trained
  model accuracy. Actual model training and automated feature selection are not
  Part A outputs. Four Part A tests and a sampled harness integration check passed.
