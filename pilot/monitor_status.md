# P9 run monitor

Read-only checks by pilot/monitor.py. Infra failures are excluded from results and cost no API tokens (the sandbox is refused before the agent connects).

## 2026-11-16 09:50 CST
- Reported BenchFlow spend: $102.21 of the $210 cap (study 2 and the pilot are billed separately).
- Rollouts so far: 1065; infra failures (not results): 367 (34%).
- Top causes: acp_error 235, CloudMetadataPowerError:delegate_power 171, sandbox_or_image 131, CloudMetadataPowerError:other_refusal 65, install_failure 59.
- Tasks skipped up front as known sandbox refusals: 7.
- Note: Cause of CloudMetadataPowerError found: BenchFlow sandbox lockdown refuses images that ship setuid/sudo or file capabilities, or that lack setpriv or a readable /proc. It depends on the task image, so retries fail again. Each refusal costs 0 tokens and about 100 s. Changed: study_run.py (backup study_run_v3_before_monitor.py) now skips known refusers and logs them as infra (ledger study_infra_skip), pauses when infra > 25%, holds any new batch until Chari approves, and appends here after each batch. Secondary batch 06 (started before this brief) is at about 41% infra; letting its last trial finish because refusals cost nothing and stopping would discard runs already paid for. No new run will start without approval. Cap unchanged at 210; nothing deleted.

## 2026-11-16 10:01 CST
- Reported BenchFlow spend: $103.75 of the $210 cap (study 2 and the pilot are billed separately).
- Rollouts so far: 1078; infra failures (not results): 376 (35%).
- Top causes: acp_error 235, CloudMetadataPowerError:delegate_power 173, sandbox_or_image 136, CloudMetadataPowerError:other_refusal 67, install_failure 59.
- Tasks skipped up front as known sandbox refusals: 7.
- Note: Chari raised the infra-failure pause threshold from 25% to 40% (monitor.py; backup monitor_v1_thresh25.py). No BenchFlow batch started. Next: live study 3 via direct API (second task family + resubmission attack), separate budget, hard stop 15 USD.

## 2026-11-16 10:06 CST
- Reported BenchFlow spend: $104.00 of the $210 cap (study 2 and the pilot are billed separately).
- Rollouts so far: 1079; infra failures (not results): 377 (35%).
- Top causes: acp_error 235, CloudMetadataPowerError:delegate_power 173, sandbox_or_image 137, CloudMetadataPowerError:other_refusal 67, install_failure 59.
- Tasks skipped up front as known sandbox refusals: 7.
- Note: Secondary batch 06 finished (sec06). Backed up with study_backup.py. No further run started; next step needs Chari.

## 2026-11-16 11:18 CST
- Reported BenchFlow spend: $104.00 of the $210 cap (study 2 and the pilot are billed separately).
- Rollouts so far: 1079; infra failures (not results): 377 (35%).
- Top causes: acp_error 235, CloudMetadataPowerError:delegate_power 173, sandbox_or_image 137, CloudMetadataPowerError:other_refusal 67, install_failure 59.
- Tasks skipped up front as known sandbox refusals: 7.
- Note: Chari raised the infra-failure pause threshold from 40% to 75% (monitor.py; backups monitor_v1_thresh25.py, monitor_v2_thresh40.py). No BenchFlow run started; any new batch still needs Chari approval.

## 2026-11-16 11:23 CST
- Reported BenchFlow spend: $104.00 of the $210 cap (study 2 and the pilot are billed separately).
- Rollouts so far: 1079; infra failures (not results): 377 (35%).
- Top causes: acp_error 235, CloudMetadataPowerError:delegate_power 173, sandbox_or_image 137, CloudMetadataPowerError:other_refusal 67, install_failure 59.
- Tasks skipped up front as known sandbox refusals: 7.
- Note: Chari removed the infra-failure pause entirely (not for the last batch, not for upcoming batches). Backups: monitor_v1_thresh25.py, monitor_v2_thresh40.py, monitor_v3_thresh75.py. Kept: the 210 USD cap, the fail-fast skip of known sandbox refusals, infra counting in this file, and Chari approval before any new batch.

## 2026-11-16 11:27 CST
- Reported BenchFlow spend: $104.00 of the $210 cap (study 2 and the pilot are billed separately).
- Rollouts so far: 1079; infra failures (not results): 377 (35%).
- Top causes: acp_error 235, CloudMetadataPowerError:delegate_power 173, sandbox_or_image 137, CloudMetadataPowerError:other_refusal 67, install_failure 59.
- Tasks skipped up front as known sandbox refusals: 7.
- Note: Chari changed the cost rule: no fixed stop and no ask before every batch; before each batch the runner estimates total spend (all API spend on this paper) plus the batch, and holds to ask Chari only if that exceeds 300 USD. Now: spent $114.59 + batch estimate $10.67 = $125.26 (ask above $300). Backups: study_run_v4_before_300rule.py, monitor_v4_before_300rule.py.
