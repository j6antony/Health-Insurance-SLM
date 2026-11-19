# Test B — actual GPU canary results, November 19, 2026

**Passed, imported, and GPU released.** The exact recovered adapter was continued for two optimizer steps on one NVIDIA L4. The complete result archive and adapter hashes were verified on the external SSD before the Colab runtime was disconnected and deleted. Colab then showed **0 active sessions, 0 compute units/hour, and “Not connected to runtime.”** No checkpoint promotion occurred.

| Measurement | Actual result |
| --- | --- |
| Provider / GPU | Google Colab Pro / NVIDIA L4, compute capability 8.9 |
| Python / PyTorch / CUDA runtime | 3.12.3 / 2.8.0+cu126 / 12.6 |
| NVIDIA driver | 580.82.07 (nvidia-smi reports driver support up to CUDA 13.0) |
| Model load | Pinned Qwen3-8B loaded in 4-bit NF4/double quantization, FP16 compute |
| Continued adapter | Rank 32 RSLoRA, alpha 32, dropout 0.0 |
| Dataset | 85 existing examples: 64 new synthetic + 21 revalidated replay |
| Training | Learning rate 0.00002; microbatch 1; accumulation 4; context 4,096 |
| Optimizer steps | **2**; effective batch 4; about 0.094 dataset epochs, not a full pass |
| Step losses | **0.1075, 0.5408**, both finite |
| Mean training loss | 0.324165508 |
| Training / worker / wrapper time | 19.47 / 164.63 / 167.85 seconds |
| Peak PyTorch allocated / reserved VRAM | **13.34 / 15.17 GiB** |
| CUDA OOM | None observed |
| Saved adapter | Nonempty, checksum verified, changed weights, original preserved |
| Saved adapter reload | Passed in the compatible runtime |
| Real inference | 445 generated tokens; nonempty; truncated=False |
| Worker compute credits | Approximately 0.088716 units |
| Worker credit valuation | **US$0.008663**, below the approved US$1 cap |
| GPU shutdown | Confirmed: 0 active sessions and 0 units/hour |

The cost is a retail valuation of existing credits, not an invoice-confirmed new charge. The approved basis was CA$13.99 per 100 compute units × approximately 1.94 units/hour × 0.6980 USD/CAD = US$0.189441388/hour. The currency conversion uses the latest published [Bank of Canada daily rate for November 18](https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates-lookup/?dF=2026-11-12&dT=2026-11-19&lP=lookup_daily_exchange_rates_2017.php&rangeType=dates&sR=2017-01-01&se=FXUSDCAD). It is an indicative conversion, not a card settlement rate. No extra credits or subscription were purchased.

Worker timing excludes setup, browser/download overhead and earlier idle allocation. The displayed balance was 94.88 units before this run window and 94.56 after shutdown; that approximate 0.32-unit change includes waiting and transfer time and display rounding. The worker's US$1 guard and additional 1,200-second watchdog did not control external idle billing. The allocation has now been released.

## Identities and import

- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`. Loaded model identity matches.
- Starting legacy adapter identity: `e8e5edc7a75f82507db3c3fe0afca721728cd5f8abe03a80fa997139970fbc0e`. Original files rehashed after import and remain unchanged.
- Job: `c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5`.
- Dataset: `51cb0864daa7680acd3afd5d68d31276bac2289cc24bf7921cbc850e81252810`.
- Original package SHA-256: `19ec9e08f6bd85eb7ee585dbb62939db7c3fbcaa588f70f853b89b82a5751e2d`.
- Returned ZIP SHA-256: `3b1d18aa220072c59105bebf84ed7909c0e8198053ca958fd5c64dfbf4280f02`.
- Saved adapter config SHA-256: `287a1ba1c904ab551c34a0daabc9d7e292b8172e03cfcae6e5e815bd9b59dec2`.
- Saved adapter weights SHA-256: `7924db3680b12eeda110c52bbed2eaca016827e9459c2a56b6f880976dfe21db`.
- Imported checkpoint: `/Volumes/Extreme SSD/Codex Distillation Loop/gpu_jobs/imported_results/c03c14696b8ab4b394266f83d850641ae3e71f9eaa3cc749a34de12489834fc5/adapter`.
- Complete returned directory: `/Volumes/Extreme SSD/Codex Distillation Loop/experiments/gpu-canary-20261119/test-b/returned/unpacked`.

The existing `python3 -m autonomous_learning import-results` mechanism completed successfully against the original job receipt. The imported adapter was rehashed and checked for rank/alpha/dropout/RSLoRA compatibility. The original recovered adapter, original package and 85-example dataset were preserved. The new checkpoint remains an engineering candidate, not the best checkpoint.

## Library versions

```json
{
  "accelerate": "1.11.0",
  "bitsandbytes": "0.48.2",
  "datasets": "4.4.1",
  "lighteval": "0.11.0",
  "peft": "0.18.0",
  "safetensors": "0.6.2",
  "torch": "2.8.0+cu126",
  "transformers": "4.57.3",
  "trl": "0.25.1"
}
```

## Actual inference example

This is the saved response from the synthetic cross-document-consistency smoke input. It is shown as an actual generation, **not a verified insurance answer or quality score**. Rule identifiers and evidence content remain unverified; do not use this response as an approved training target.

```json
{"result": [{"rule_id": "Presence", "status": "passed", "reason": "Claim form present"}, {"rule_id": "Readability", "status": "passed", "reason": "Claim form legible"}, {"rule_id": "patient_id, service_date and signature", "status": "passed", "reason": "Patient ID matches; service date matches; signature present"}, {"rule_id": "Service dates agree across readable documents", "status": "passed", "reason": "Service date matches across readable documents"}, {"rule_id": "Conditional referral", "status": "issue_detected", "reason": "Referral required; signed referral present"}], "evidence": [{"document_id": "DOC-1", "page_number": 1, "start": 0, "end": 100, "quote": "CLAIM FORM\nSynthetic fictional training document\nPatient identifier = SYN-P-a95dcecc59\nClaim identifier = SYN-C-a95dcecc59af3eea5de5\nService date = 04 Feb 2025\nSignature: present"}, {"document_id": "DOC-2", "page_number": 1, "start": 0, "end": 100, "quote": "INVESTIGATION REPORT\nSynthetic fictional training document\nPatient identifier = SYN-P-a95dcecc59\nClaim identifier = SYN-C-a95dcecc59af3eea5de5\nSignature: present"}, {"document_id": "DOC-3", "page_number": 1, "start": 0, "end": 100, "quote": "SIGNED REFERRAL\nSynthetic fictional training document\nPatient identifier = SYN-P-a95dcecc59\nClaim identifier = SYN-C-a95dcecc59af3eea5de5\nService date = 04 Feb 2025\nSignature: present"}], "source_references": ["CLAIM FORM", "INVESTIGATION REPORT", "SIGNED REFERRAL"], "requires_review": false}
```

## Warnings and limits

- Recovered and newly saved PEFT adapter configs omit the historical base revision; job and loaded model identity pin b968826d9c46dd6066d109eabc6255188de91218.
- Runtime tokenizer revision metadata is null; tokenizer loading explicitly used the pinned job revision.
- Trainer aligned model/generation PAD/BOS token settings to tokenizer (pad=151643, bos=None).
- Greedy generation ignored temperature/top_p/top_k inherited settings.
- Earlier host-Python/ensurepip/interpreter-selection errors were resolved without changing the immutable package or training dataset.
- Example inference contains unverified rule/evidence content and must not be used as an approved label or quality result.

**This proves the load → continue → two-step train → save → reload → infer → transfer → import infrastructure works for this exact pinned single-L4 canary. It does not demonstrate model improvement, independent evaluation readiness, another GPU environment, or unattended operation.**
