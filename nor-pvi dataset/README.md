# `nor-pvi` Dataset

> **_NOTE:_** Dataset is created for research purposes only.

Dataset contains files for three tasks in train/val/test in TSV format.

## Data statistics
| Task | Dataset | Train | Val | Test |Total|
|------|------|-----|------|------|-----|
| PVI           | `nor-pvi` | `1,007`        |  `112`     | `113` | `1,232`|
|Summarization  | `nor-pvi` |`3,254`     |`404`  |`409`|`4,027`|
| Stance        | `nor-pvi` | `984`        | `119`      | `117` |`1,220`|

## Data structure
The files are in TSV format.
### PVI tasks
TSV has column `text` corresponding to the full political speech and column `label` is for the viewpoints.
One speech might contain one or more viewpoints

### Summarization
Similar format.
Column `text` has full speeches and `label` is for the summaries.
One speech has one summary.

### Stance
Column `text` is the viewpoint and column `label` is the stance label.