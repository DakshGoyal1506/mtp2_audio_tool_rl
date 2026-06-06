# Results Table Template

Use this template for external experiment reports. Do not commit generated result dumps, raw outputs, checkpoints, logs, or audio.

## Main Comparison

| Run | Model/checkpoint | Prompting mode | Training mode | Manifest | Notes |
| --- | --- | --- | --- | --- | --- |
| Baseline |  | no tools | none |  |  |
| Tool prompting |  | tools prompted | none |  |  |
| GRPO |  | tools prompted | GRPO |  |  |

## Metrics

| Run | Answer accuracy | Tool-use rate | Valid tool-call rate | Invalid tool-call rate | Tool-result-used rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline |  |  |  |  |  |
| Tool prompting |  |  |  |  |  |
| GRPO |  |  |  |  |  |

## Failure Cases

| sample_id | Failure type | Baseline output | GRPO output | Notes |
| --- | --- | --- | --- | --- |
|  | wrong answer |  |  |  |
|  | invalid tool call |  |  |  |
|  | unnecessary tool use |  |  |  |

## Notes

- Keep full prediction JSONL files outside Git.
- Commit only tiny examples, schemas, and concise summaries.
- Record the external manifest path and model/checkpoint identity outside this table when needed.
