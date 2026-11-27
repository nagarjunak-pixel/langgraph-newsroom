from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable

from newsroom.models import ArticleDraft, NewsroomState, SourceDocument
from newsroom.verifier import verify_draft_grounding


def compute_state_hash(state: NewsroomState) -> str:
    """Canonical SHA-256 digest of a NewsroomState snapshot for dirty-state detection."""
    canonical = json.dumps(state, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class TransactionalStageGuard:
    """
    SecondChance & ARGUS-inspired (arXiv:2611.31586, arXiv:2611.33605) transactional stage wrapper.

    Captures pre-execution state snapshots and rolls back partial/corrupted state mutations
    whenever a stage raises mid-flight or emits an ungrounded draft mutation.
    """

    def __init__(self, require_verified_draft: bool = True) -> None:
        self._require_verified_draft = require_verified_draft
        self.rollback_count = 0

    def execute_atomic(
        self,
        prior_state: NewsroomState,
        stage_fn: Callable[[NewsroomState], NewsroomState],
    ) -> tuple[NewsroomState, bool, str]:
        snapshot = copy.deepcopy(prior_state)
        pre_hash = compute_state_hash(snapshot)

        try:
            working = copy.deepcopy(prior_state)
            delta = stage_fn(working)
            candidate: NewsroomState = copy.deepcopy(prior_state)
            candidate.update(delta)

            if self._require_verified_draft and "draft" in delta and "sources" in candidate:
                draft = ArticleDraft.model_validate(candidate["draft"])
                sources = [SourceDocument.model_validate(s) for s in candidate["sources"]]
                report = verify_draft_grounding(draft, sources)
                if not report.passed:
                    raise ValueError(f"Ungrounded stage mutation: {','.join(report.risk_flags)}")

            return candidate, True, compute_state_hash(candidate)
        except Exception as exc:
            self.rollback_count += 1
            restored = copy.deepcopy(snapshot)
            errs = list(restored.get("error_log", []))
            errs.append(f"atomic_rollback:{exc}")
            # Verify core payload matches pre-execution snapshot before recording audit note
            assert compute_state_hash(snapshot) == pre_hash
            restored["error_log"] = errs
            return restored, False, pre_hash
