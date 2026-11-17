"""
Physician Review Engine for the clinical NLP pipeline.

Allows clinicians to ACCEPT, EDIT, REJECT, or VERIFY extracted clinical items.
Maintains an immutable audit log of all review actions.
Enforces the mandatory clinical decision support disclaimer.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime
import uuid

from src.common.schemas import (
    PatientRecord,
    ReviewAction,
    ReviewActionType,
    SectionItem,
)
from src.utils import get_logger

logger = get_logger("PhysicianReviewEngine")

DISCLAIMER_NOTICE = (
    "Decision support only. The physician remains responsible for clinical decisions."
)


class PhysicianReviewEngine:
    """
    Manages clinical validation and human-in-the-loop review of patient record items.
    """

    def __init__(self, patient_record: Optional[PatientRecord] = None):
        self.patient_record = patient_record
        self._audit_log: List[ReviewAction] = []
        self._status_map: Dict[str, ReviewActionType] = {}
        logger.info("PhysicianReviewEngine initialized. %s", DISCLAIMER_NOTICE)

    @property
    def disclaimer(self) -> str:
        return DISCLAIMER_NOTICE

    def accept(self, item_id: str, reviewer: str, note: Optional[str] = None) -> ReviewAction:
        """Mark an item as accepted by the reviewing physician."""
        action = ReviewAction(
            id=str(uuid.uuid4()),
            item_id=item_id,
            action=ReviewActionType.ACCEPT,
            reviewer=reviewer,
            timestamp=datetime.utcnow(),
            note=note,
        )
        self._audit_log.append(action)
        self._status_map[item_id] = ReviewActionType.ACCEPT
        logger.info("Review action ACCEPT recorded")
        return action

    def edit(
        self,
        item_id: str,
        new_value: str,
        reviewer: str,
        old_value: Optional[str] = None,
        note: Optional[str] = None,
    ) -> ReviewAction:
        """Edit the content or interpretation of an item."""
        action = ReviewAction(
            id=str(uuid.uuid4()),
            item_id=item_id,
            action=ReviewActionType.EDIT,
            old_value=old_value,
            new_value=new_value,
            reviewer=reviewer,
            timestamp=datetime.utcnow(),
            note=note,
        )
        self._audit_log.append(action)
        self._status_map[item_id] = ReviewActionType.EDIT

        # If attached to a PatientRecord, update the section item display if found
        if self.patient_record:
            for item in self.patient_record.section_items:
                if item.id == item_id or item.entity_id == item_id:
                    item.display_text = new_value

        logger.info("Review action EDIT recorded")
        return action

    def reject(self, item_id: str, reviewer: str, reason: Optional[str] = None) -> ReviewAction:
        """Reject an item (e.g. false positive extraction or incorrect attribution)."""
        action = ReviewAction(
            id=str(uuid.uuid4()),
            item_id=item_id,
            action=ReviewActionType.REJECT,
            reviewer=reviewer,
            timestamp=datetime.utcnow(),
            note=reason,
        )
        self._audit_log.append(action)
        self._status_map[item_id] = ReviewActionType.REJECT
        logger.info("Review action REJECT recorded")
        return action

    def verify(self, item_id: str, reviewer: str, note: Optional[str] = None) -> ReviewAction:
        """Formally verify an item against original source provenance."""
        action = ReviewAction(
            id=str(uuid.uuid4()),
            item_id=item_id,
            action=ReviewActionType.VERIFY,
            reviewer=reviewer,
            timestamp=datetime.utcnow(),
            note=note,
        )
        self._audit_log.append(action)
        self._status_map[item_id] = ReviewActionType.VERIFY
        logger.info("Review action VERIFY recorded")
        return action

    def get_audit_log(self) -> List[ReviewAction]:
        """Return the complete audit log of review actions."""
        return list(self._audit_log)

    def get_item_status(self, item_id: str) -> Optional[ReviewActionType]:
        """Get the latest review status for an item."""
        return self._status_map.get(item_id)
