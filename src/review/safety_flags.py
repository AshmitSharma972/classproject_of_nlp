"""Heuristic completeness and provenance flags; never diagnostic or prescriptive."""
from __future__ import annotations
import re

from src.common.schemas import AssertionStatus, EntityType


def build_safety_flags(raw_text, entities, assertions, time_expressions, relations, demographics=None):
    flags = []
    assertion_by_id = {a.entity_id: a for a in assertions}
    entities_by_id = {e.id: e for e in entities}
    linked = {eid for t in time_expressions for eid in t.linked_entity_ids}
    med_ids = {e.id for e in entities if e.type == EntityType.MEDICATION}
    med_rel_ids = set()
    for r in relations:
        if r.head_entity_id in med_ids and r.relation_type.value in ("MEDICATION-DOSAGE", "MEDICATION-FREQUENCY"):
            med_rel_ids.add(r.head_entity_id)
    for time_expr in time_expressions:
        for e in entities:
            a = assertion_by_id.get(e.id)
            if (e.span.source_sentence_id == time_expr.span.source_sentence_id and
                e.span.end <= time_expr.span.start and a and a.status == AssertionStatus.NEGATED):
                flags.append(_flag("duration-on-negated-entity", "Consider verifying the duration attribution.",
                                   time_expr.span.start, time_expr.span.end))
                break
    for e in entities:
        if e.type == EntityType.MEDICATION and e.id not in med_rel_ids:
            flags.append(_flag("drug-without-dose-frequency", "Consider asking whether a dose or frequency was omitted.",
                               e.span.start, e.span.end))
    allergens = [e for e in entities if e.type == EntityType.ALLERGEN]
    meds = [e for e in entities if e.type == EntityType.MEDICATION]
    for a in allergens:
        for m in meds:
            if a.span.text.casefold() == m.span.text.casefold():
                flags.append(_flag("drug-allergen-overlap", "Verify whether this name refers to an allergy or medication.",
                                   a.span.start, a.span.end))
    for eid in linked:
        values = [t for t in time_expressions if eid in t.linked_entity_ids]
        if len({t.normalized_value for t in values}) > 1:
            e = entities_by_id[eid]
            flags.append(_flag("conflicting-durations", "Consider verifying the reported durations.",
                               e.span.start, e.span.end))
    low = raw_text.casefold()
    if demographics:
        form_age = demographics.get("age")
        age_match = re.search(r"\b(?:age\s*|aged\s*)(\d{1,3})\b|\b(\d{1,3})[- ]year[- ]old\b|\b(\d{1,3}) years? old\b", low)
        if form_age is not None and age_match:
            narration_age = next((part for part in age_match.groups() if part is not None), None)
            if narration_age and int(narration_age) != int(form_age):
                flags.append(_flag("demographic-age-conflict",
                                   f"Narration says {narration_age}, form says {form_age} - please verify.",
                                   age_match.start(), age_match.end()))
        form_allergies = (demographics.get("known_allergies") or "").strip()
        form_none = form_allergies.casefold() in {"none", "no", "no known allergies", "nka"}
        narration_allergens = [e.span.text for e in entities if e.type == EntityType.ALLERGEN]
        allergy_phrase = re.search(r"\b(?:allergic|allergy|allergies)\s+(?:to|include)\s+([^.;,]+)", low)
        narration_none = bool(re.search(r"\b(?:no known allergies|not allergic to|no allergies)\b", low))
        narration_positive = bool(allergy_phrase) and not narration_none
        mismatch = (form_none and narration_positive) or (not form_none and form_allergies and narration_none)
        if not form_none and form_allergies and allergy_phrase and narration_positive:
            form_items = [item.strip().casefold() for item in re.split(r"[,;]|\band\b", form_allergies) if item.strip()]
            mentioned = allergy_phrase.group(1).strip().casefold()
            if form_items and not any(item in mentioned or mentioned in item for item in form_items):
                mismatch = True
        if mismatch:
            said = ", ".join(narration_allergens) if narration_allergens else (allergy_phrase.group(1).strip() if allergy_phrase else "no known allergies")
            form = form_allergies or "Not provided"
            first = next((e for e in entities if e.type == EntityType.ALLERGEN), None)
            flags.append(_flag("demographic-allergy-conflict",
                               f"Narration says allergy: {said}; form says {form} - please verify.",
                               first.span.start if first else None, first.span.end if first else None))
    for item, words in {
        "allergies": ("allergy", "allergies", "allergic"),
        "onset": ("since", "started", "began", "onset"),
        "current medications": ("taking", "takes", "medication", "medicine"),
        "relevant negatives": ("no ", "denies", "without", "negative for"),
    }.items():
        if not any(w in low for w in words):
            flags.append(_flag("not-mentioned-" + item.replace(" ", "-"),
                               f"Consider asking about {item}; it was not mentioned.", None, None))
    return flags


def _flag(flag_id, message, start, end):
    return {"id": flag_id, "severity": "review", "message": message,
            "evidence": [] if start is None else [{"start": start, "end": end, "source": "english"}]}
