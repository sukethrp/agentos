from __future__ import annotations
import re
from typing import Literal

SSN_PATTERN = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")
DOB_PATTERN = re.compile(r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b|\b(?:19|20)\d{2}[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])\b")
MRN_PATTERN = re.compile(r"\bMRN[:\s#]*\d{6,12}\b|\b\d{6,12}\s*\(?MRN\)?\b", re.I)
PHI_INDICATORS = re.compile(r"\b(?:patient|ssn|social security|date of birth|dob|medical record|mrn|health insurance|hipaa)\b", re.I)


class DataClassifier:
    def classify(self, data: str) -> Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL", "PHI"]:
        if not data or not data.strip():
            return "PUBLIC"
        text = data.strip()
        if SSN_PATTERN.search(text) or DOB_PATTERN.search(text) or MRN_PATTERN.search(text):
            return "PHI"
        if PHI_INDICATORS.search(text):
            return "PHI"
        if any(k in text.lower() for k in ["confidential", "secret", "proprietary", "restricted"]):
            return "CONFIDENTIAL"
        if any(k in text.lower() for k in ["internal", "draft", "preliminary"]):
            return "INTERNAL"
        return "PUBLIC"
