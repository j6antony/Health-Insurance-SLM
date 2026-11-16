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

## Prelaunch review update

Generation now plans up to four quota-limited batches together. The caller
reserves the requested task counts across the group. Workers save provider
receipts, while validation, overlap checks, prompt preparation and admission
run in planned order. Exact duplicates and excluded cases are blocked before
audit. Admission rechecks overlap after earlier batches are committed. Resume
reuses completed generation and audit receipts, including when one sibling
request fails. A provider response lost before its receipt is on disk may
still need a new budgeted request.

Refills reserve only the measured task deficits. A deficit-only refill has
zero buffer. With one open adaptive slot, the refill may request one extra
candidate, compared with eight extras in the first 80-slot window. A coverage
shortfall still stops the round rather than admitting unsuitable examples.

The new notebook writes to `colab_l4_hybrid_fresh_round1_v2`. Enable
`RESEARCH_MACRO_CONTINUATION` in its configuration before the initial launch
to record the highest-macro checkpoint as the input for any explicitly
authorized later research round. The one-round notebook still stops at one
round. Safety and significance gates continue to control the accepted best.
The report records safety failures for ranked candidates and hybrid runs
reject release evaluation.

The no-cost controller test ran one mock round, reopened the saved run and
explicitly extended it to a second mock round. Both rounds produced 200
distinct pairs with the 60/60/80 composition and all 18 tasks. The 360-case
development and 108-case holdout exclusions were verified against all 400
training records. The higher-macro candidate was flagged for an unsupported
claim and seeded only the opt-in research continuation. A separate interruption
test reused all four generation replies and three completed audit replies.
The controller mock recorded 305 simulated API requests across two rounds:
40 generation, 40 scenario audit, 110 teacher answer (101 batched),
100 answer audit (98 batched), and 15 student assessment requests. Mock
evaluations exercised checkpoint policy without live semantic judge requests.
The stage export distinguishes summed request durations, elapsed spans and
the union of active request intervals, so concurrent time is counted once.
Colab wall time, provider behavior and real score changes remain unmeasured.

## Four-round Colab preset

Use `distillation_colab_hybrid_4round.ipynb` with the matching bundle and
`colab_l4_hybrid_fresh_4round_v1` Drive directory. This preset trains four
rounds of 200 fresh examples each, evaluates the baseline and each candidate
on the frozen development set, and shares a single cumulative API cap.
It checks the required historical and remediation plans plus the prior
one-round hybrid plan if that plan exists when the new run is configured.
A resumed run keeps its original exclusion plan list.

`RESEARCH_MACRO_CONTINUATION` defaults to false. Set it before the first
launch if you want the highest-scoring experimental checkpoint to start
later rounds. Accepted checkpoints still use the existing safety gates.
The release holdout is excluded from training and is not evaluated.
