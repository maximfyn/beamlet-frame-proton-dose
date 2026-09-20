"""Fixed stratified patient split registry — DoseRAD 2026 proton CT.

Deterministic and immutable over all 75 training patients (abdominal ``1ABB`` /
thoracic ``1THB``), with no overlap between the three sets:

- 61 train (81.3%)      — 29 abdominal / 32 thoracic
- 6 validation (8.0%)   — 3 abdominal / 3 thoracic
- 8 unseen test (10.7%) — 4 abdominal / 4 thoracic

**The released weights trained on all 75**, so for them these 8 (and the 6
validation patients) are in-sample: scoring the released checkpoints on
``TEST_PATIENTS`` measures memorisation, not generalisation.

**This file is the registry**, and the rules that belong with it:

- **Any run passing explicit ``--train-patients``/``--val-patients`` bypasses
  this registry and produces a val number not comparable to any other run.**
  That is the single easiest way to make two experiments look comparable when
  they are not.
- **The split is not re-cut.** The hidden test pool is 65% abdominal against
  this split's 4/4, which is a real mismatch — but the two places it matters
  correct for it by *weighting*: checkpoint selection
  (`scripts/train/train_doserad.py`, ``THORACIC_SHARE``, recorded in the
  checkpoint) and cohort aggregation, where a weighted mean is reported beside
  the unweighted one rather than replacing it. Re-cutting would also break the
  paired comparison every development result rests on: each arm trained on the
  same 61 patients, so only these 8 are clean for all of them.
- **A patient directory holds exactly 1,080 dose files.** Anything shorter is a
  truncated download, not a coherent subset: listing the remote directory caps
  at 1,000 entries and silently drops the alphabetical tail (beams 7–9, gantry
  70–90°). Diff the plan JSON, which cannot be paginated, against the disk —
  and never enumerate the remote directory.
"""
TRAIN_PATIENTS = [
    '1ABB006', '1ABB021', '1ABB030', '1ABB031', '1ABB035', '1ABB036', '1ABB039', '1ABB041',
    '1ABB042', '1ABB045', '1ABB061', '1ABB067', '1ABB078', '1ABB083', '1ABB098', '1ABB102',
    '1ABB109', '1ABB118', '1ABB124', '1ABB128', '1ABB135', '1ABB138', '1ABB143', '1ABB147',
    '1ABB149', '1ABB155', '1ABB161', '1ABB164', '1ABB169', '1THB002', '1THB008', '1THB011',
    '1THB016', '1THB017', '1THB021', '1THB023', '1THB027', '1THB029', '1THB031', '1THB037',
    '1THB043', '1THB045', '1THB048', '1THB052', '1THB063', '1THB067', '1THB074', '1THB076',
    '1THB078', '1THB095', '1THB121', '1THB122', '1THB143', '1THB191', '1THB202', '1THB205',
    '1THB211', '1THB214', '1THB220', '1THB221', '1THB226'
]

VAL_PATIENTS = [
    '1ABB011', '1ABB123', '1ABB145', '1THB058', '1THB195', '1THB217'
]

TEST_PATIENTS = [
    '1ABB020', '1ABB070', '1ABB110', '1ABB115', '1THB054', '1THB119', '1THB120', '1THB218'
]

def get_splits():
    return {
        "train": TRAIN_PATIENTS,
        "val": VAL_PATIENTS,
        "test": TEST_PATIENTS
    }

