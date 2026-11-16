# Fresh hybrid curriculum v1

Round 1 used 200 verified examples and improved raw development macro from
32.69% to 37.78%. The reported 286.3 minutes work out to 1.07 percentage points
of macro improvement per hour. GPU fine-tuning occupied 3 minutes 43 seconds.
The saved API journal contained 1,761 attempts.

| Operation | API attempts | Cumulative request minutes |
| --- | ---: | ---: |
| Scenario generation | 87 | 65.2 |
| Benchmark semantic judging | 597 | 51.9 |
| Teacher answers | 422 | 39.0 |
| Teacher answer audits | 408 | 38.9 |
| Scenario audits | 86 | 28.1 |
| Student assessment | 158 | 16.6 |
| Model checks | 3 | 0.04 |

Request times are sums from the saved CSV. They are distinct from elapsed
stage times and do not include local work or time between requests.

The v1 preset freezes a fresh 60/60/80 mix. It uses grounded, rule-based
construction for six suitable tasks and retains teacher verification for
complex cases. The first adaptive window requests 80 candidates plus a
buffer of eight. Refill requests depend on remaining slots and coverage
deficits. The selection favors student mistakes and harder cases within
task coverage. It requires at least two screened and two selected examples
per task. Generation proceeds without demonstrations drawn from previously
accepted training scenarios in the Colab preset.

New rounds snapshot earlier candidate IDs and use separate programmatic
seeds. Prior training plans are bound to run identity. Exclusion checks cover
IDs, labeled source families and canonical scenario content, with near-duplicate
checks for teacher examples and frozen benchmark exclusions. Free-form teacher
examples lack authoritative family labels, so their additional family protection
depends on content and near-duplicate detection. The benchmark and holdout
remain outside training, and release scoring is disabled in the preset.
Experimental macro ranking is written separately from safety-gated selection.

Up to four requests run concurrently for independent benchmark judgments and
student assessments. Compatible teacher answers and audits are grouped into
same-schema batches of up to three, with separately keyed case payloads. Each
result still goes through local validation and source audit. A malformed batch
response may fall back to individual requests. Provider rejections and budget
stops propagate. Successful responses are cached by exact request content and
settings within the new run. The budget is reserved per physical request and
Supabase receives API usage as it is logged.

The new run records API attempt and timing totals in
`api_stage_profile.json`, alongside baseline and round timing files.
The no-cost mock produced 200 examples across all 18 tasks, screened 88
candidates, batched 12 homogeneous answers into four requests, and used zero
additional calls when the batch cache was reopened. Two local 60-example
deterministic rounds passed the frozen 360/108 exclusion checks in 0.66 seconds,
with six tasks represented in each. A 32-job simulated latency test took
0.838 seconds serially and 0.202 seconds with four workers. This measures local
dispatch with simulated delay, without real provider rate limits, token costs,
or GPU work. Real speed and benchmark gains require the Colab experiment.
