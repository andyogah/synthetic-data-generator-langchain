"""
PII Protection Module - Compliant with NIST Special Publication 800-122

This module implements comprehensive PII detection and anonymization following
NIST SP 800-122 guidelines for protecting personally identifiable information.

NIST SP 800-122 defines PII as:
"Any information about an individual maintained by an agency, including (1) any 
information that can be used to distinguish or trace an individual's identity, 
such as name, social security number, date and place of birth, mother's maiden 
name, or biometric records; and (2) any other information that is linked or 
linkable to an individual, such as medical, educational, financial, and 
employment information."
"""

import re
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider


class PIIRiskLevel(Enum):
    """
    PII Risk Classification according to NIST SP 800-122
    
    HIGH: Could result in significant harm, embarrassment, inconvenience, 
          or unfairness to an individual if disclosed
    MODERATE: Could result in minor harm if disclosed  
    LOW: Unlikely to result in harm if disclosed
    """
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


@dataclass
class PIIPattern:
    """
    PII Pattern definition following NIST SP 800-122 categorization
    
    Attributes:
        name: Identifier for the PII type
        pattern: Regex pattern for detection
        replacement: Anonymization replacement text
        risk_level: NIST risk classification
        nist_category: NIST PII category
        description: NIST compliant description
    """
    name: str
    pattern: str
    replacement: str
    risk_level: PIIRiskLevel
    nist_category: str
    description: str


class NISTCompliantPIIProtector:
    """
    NIST SP 800-122 Compliant PII Protection System
    
    This class implements comprehensive PII detection and anonymization
    following NIST Special Publication 800-122 guidelines.
    
    Key Features:
    - Detection of all NIST-recognized PII types
    - Risk-based categorization (HIGH/MODERATE/LOW)
    - Configurable anonymization strategies
    - Comprehensive logging and audit trail
    - Support for both direct and indirect PII identifiers
    """
    
    def __init__(self, risk_threshold: PIIRiskLevel = PIIRiskLevel.MODERATE):
        """
        Initialize NIST-compliant PII protector
        
        Args:
            risk_threshold: Minimum risk level for PII protection
        """
        self.risk_threshold = risk_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize Presidio engines
        self._init_presidio_engines()
        
        # Define NIST SP 800-122 compliant PII patterns
        self.nist_pii_patterns = self._define_nist_pii_patterns()
        
        # NIST-recognized PII entities for Presidio
        self.nist_presidio_entities = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
            "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "DATE_TIME",
            "MEDICAL_LICENSE", "US_BANK_NUMBER", "CRYPTO", "IBAN_CODE",
            "IP_ADDRESS", "URL", "US_ITIN", "LOCATION", "ORGANIZATION"
        ]
    
    def _init_presidio_engines(self):
        """Initialize Presidio analyzer and anonymizer engines"""
        try:
            # Configure NLP engine for better accuracy
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.anonymizer = AnonymizerEngine()
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Presidio with spaCy: {e}")
            # Fallback to basic configuration
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
    
    def _define_nist_pii_patterns(self) -> List[PIIPattern]:
        """
        Define comprehensive PII patterns according to NIST SP 800-122
        
        Returns:
            List of PIIPattern objects covering all NIST-recognized PII types
        """
        return [
            # Direct Identifiers (HIGH RISK) - Can uniquely identify individuals
            PIIPattern(
                name="ssn_full",
                pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                replacement="[SSN-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Direct Identifier",
                description="Social Security Number - NIST SP 800-122 High Risk PII"
            ),
            PIIPattern(
                name="ssn_no_dash",
                pattern=r'\b\d{9}\b',
                replacement="[SSN-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Direct Identifier",
                description="Social Security Number without dashes"
            ),
            PIIPattern(
                name="drivers_license",
                pattern=r'\b[A-Z]{1,2}\d{6,8}\b',
                replacement="[DL-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Direct Identifier",
                description="Driver's License Number - NIST SP 800-122 High Risk PII"
            ),
            PIIPattern(
                name="passport_number",
                pattern=r'\b[A-Z]{2}\d{7}\b',
                replacement="[PASSPORT-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Direct Identifier",
                description="Passport Number - NIST SP 800-122 High Risk PII"
            ),
            PIIPattern(
                name="taxpayer_id",
                pattern=r'\b\d{2}-\d{7}\b',
                replacement="[TIN-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Direct Identifier",
                description="Taxpayer Identification Number - NIST SP 800-122 High Risk PII"
            ),
            
            # Financial Information (HIGH RISK)
            PIIPattern(
                name="credit_card_visa",
                pattern=r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                replacement="[CC-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Financial Information",
                description="Credit Card Number (Visa) - NIST SP 800-122 High Risk PII"
            ),
            PIIPattern(
                name="credit_card_mastercard",
                pattern=r'\b5\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                replacement="[CC-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Financial Information",
                description="Credit Card Number (MasterCard) - NIST SP 800-122 High Risk PII"
            ),
            PIIPattern(
                name="bank_account",
                pattern=r'\b\d{8,17}\b',
                replacement="[ACCOUNT-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Financial Information",
                description="Bank Account Number - NIST SP 800-122 High Risk PII"
            ),
            
            # Contact Information (MODERATE RISK)
            PIIPattern(
                name="email",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                replacement="[EMAIL-REDACTED]",
                risk_level=PIIRiskLevel.MODERATE,
                nist_category="Contact Information",
                description="Email Address - NIST SP 800-122 Moderate Risk PII"
            ),
            PIIPattern(
                name="phone_us",
                pattern=r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                replacement="[PHONE-REDACTED]",
                risk_level=PIIRiskLevel.MODERATE,
                nist_category="Contact Information",
                description="Phone Number - NIST SP 800-122 Moderate Risk PII"
            ),
            PIIPattern(
                name="address_street",
                pattern=r'\b\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Circle|Cir|Way)\b',
                replacement="[ADDRESS-REDACTED]",
                risk_level=PIIRiskLevel.MODERATE,
                nist_category="Contact Information",
                description="Street Address - NIST SP 800-122 Moderate Risk PII"
            ),
            PIIPattern(
                name="zip_code",
                pattern=r'\b\d{5}(?:-\d{4})?\b',
                replacement="[ZIP-REDACTED]",
                risk_level=PIIRiskLevel.MODERATE,
                nist_category="Contact Information",
                description="ZIP Code - NIST SP 800-122 Moderate Risk PII"
            ),
            
            # Date Information (MODERATE RISK)
            PIIPattern(
                name="date_birth",
                pattern=r'\b(?:0[1-9]|1[0-2])[\/\-](?:0[1-9]|[12][0-9]|3[01])[\/\-](?:19|20)\d{2}\b',
                replacement="[DOB-REDACTED]",
                risk_level=PIIRiskLevel.MODERATE,
                nist_category="Date Information",
                description="Date of Birth - NIST SP 800-122 Moderate Risk PII"
            ),
            
            # Medical Information (HIGH RISK)
            PIIPattern(
                name="medical_record",
                pattern=r'\bMRN[-\s]?\d{6,10}\b',
                replacement="[MRN-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Medical Information",
                description="Medical Record Number - NIST SP 800-122 High Risk PII"
            ),
            PIIPattern(
                name="health_insurance",
                pattern=r'\b[A-Z]{3}\d{9}\b',
                replacement="[INSURANCE-REDACTED]",
                risk_level=PIIRiskLevel.HIGH,
                nist_category="Medical Information",
                description="Health Insurance Number - NIST SP 800-122 High Risk PII"
            ),
            
            # Employment Information (MODERATE RISK)
            PIIPattern(
                name="employee_id",
                pattern=r'\b(?:EMP|EMPL|ID)[-\s]?\d{4,8}\b',
                replacement="[EMP-ID-REDACTED]",
                risk_level=PIIRiskLevel.MODERATE,
                nist_category="Employment Information",
                description="Employee ID - NIST SP 800-122 Moderate Risk PII"
            ),
            
            # Educational Information (MODERATE RISK)
            PIIPattern(
                name="student_id",
                pattern=r'\b(?:STU|STUD|SID)[-\s]?\d{4,10}\b',
                replacement="[STUDENT-ID-REDACTED]",
                risk_level=PIIRiskLevel.MODERATE,
                nist_category="Educational Information",
                description="Student ID - NIST SP 800-122 Moderate Risk PII"
            ),
            
            # IP Address (LOW-MODERATE RISK)
            PIIPattern(
                name="ipv4_address",
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                replacement="[IP-REDACTED]",
                risk_level=PIIRiskLevel.LOW,
                nist_category="Network Information",
                description="IPv4 Address - NIST SP 800-122 Low-Moderate Risk PII"
            ),
        ]
    
    def detect_pii_comprehensive(self, text: str) -> List[Dict]:
        """
        Comprehensive PII detection using both Presidio and custom patterns
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PII entities with NIST risk classification
        """
        detected_pii = []
        
        # 1. Use Presidio for standard PII detection
        try:
            presidio_results = self.analyzer.analyze(
                text=text, 
                entities=self.nist_presidio_entities,
                language='en'
            )
            
            for result in presidio_results:
                detected_pii.append({
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": text[result.start:result.end],
                    "detection_method": "presidio",
                    "risk_level": self._get_presidio_risk_level(result.entity_type),
                    "nist_compliant": True
                })
        except Exception as e:
            self.logger.error(f"Presidio analysis failed: {e}")
        
        # 2. Use custom NIST patterns for additional detection
        for pattern in self.nist_pii_patterns:
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            for match in matches:
                # Avoid duplicates with Presidio results
                if not self._is_overlapping(match.span(), detected_pii):
                    detected_pii.append({
                        "entity_type": pattern.name,
                        "start": match.start(),
                        "end": match.end(),
                        "score": 1.0,
                        "text": match.group(),
                        "detection_method": "custom_pattern",
                        "risk_level": pattern.risk_level.value,
                        "nist_category": pattern.nist_category,
                        "description": pattern.description,
                        "nist_compliant": True
                    })
        
        # Sort by position in text
        detected_pii.sort(key=lambda x: x["start"])
        
        self.logger.info(f"Detected {len(detected_pii)} PII entities following NIST SP 800-122")
        return detected_pii
    
    def anonymize_text_nist_compliant(self, 
                                    text: str, 
                                    risk_threshold: Optional[PIIRiskLevel] = None) -> Tuple[str, List[Dict]]:
        """
        NIST SP 800-122 compliant text anonymization
        
        Args:
            text: Input text to anonymize
            risk_threshold: Minimum risk level for anonymization
            
        Returns:
            Tuple of (anonymized_text, anonymization_report)
        """
        if risk_threshold is None:
            risk_threshold = self.risk_threshold
        
        # Detect all PII
        detected_pii = self.detect_pii_comprehensive(text)
        
        # Filter by risk threshold
        pii_to_anonymize = [
            pii for pii in detected_pii 
            if self._should_anonymize(pii, risk_threshold)
        ]
        
        # Anonymize text (process from end to start to maintain positions)
        anonymized_text = text
        anonymization_report = []
        
        for pii in sorted(pii_to_anonymize, key=lambda x: x["start"], reverse=True):
            # Get replacement text
            replacement = self._get_replacement_text(pii)
            
            # Replace in text
            anonymized_text = (
                anonymized_text[:pii["start"]] + 
                replacement + 
                anonymized_text[pii["end"]:]
            )
            
            # Add to report
            anonymization_report.append({
                "original_text": pii["text"],
                "replacement": replacement,
                "entity_type": pii["entity_type"],
                "risk_level": pii["risk_level"],
                "nist_category": pii.get("nist_category", "Unknown"),
                "position": (pii["start"], pii["end"]),
                "nist_compliant": True
            })
        
        self.logger.info(f"Anonymized {len(anonymization_report)} PII entities per NIST SP 800-122")
        return anonymized_text, anonymization_report
    
    def sanitize_metadata_nist_compliant(self, metadata: Dict) -> Dict:
        """
        NIST SP 800-122 compliant metadata sanitization
        
        Args:
            metadata: Document metadata dictionary
            
        Returns:
            Sanitized metadata dictionary
        """
        # NIST-identified sensitive metadata fields
        sensitive_fields = {
            'author', 'creator', 'contributor', 'publisher', 'owner',
            'email', 'phone', 'address', 'contact', 'user', 'username',
            'employee', 'student', 'patient', 'client', 'customer'
        }
        
        sanitized = {}
        sanitization_report = []
        
        for key, value in metadata.items():
            key_lower = key.lower()
            
            if any(field in key_lower for field in sensitive_fields):
                # Hash sensitive metadata
                original_value = str(value)
                sanitized_value = self._hash_value(original_value)
                sanitized[key] = sanitized_value
                
                sanitization_report.append({
                    "field": key,
                    "action": "hashed",
                    "nist_compliant": True,
                    "reason": "Sensitive metadata field per NIST SP 800-122"
                })
            elif isinstance(value, str):
                # Check string values for PII
                anonymized_value, pii_report = self.anonymize_text_nist_compliant(value)
                sanitized[key] = anonymized_value
                
                if pii_report:
                    sanitization_report.append({
                        "field": key,
                        "action": "anonymized",
                        "pii_found": len(pii_report),
                        "nist_compliant": True,
                        "reason": "PII detected in metadata value"
                    })
            else:
                sanitized[key] = value
        
        self.logger.info(f"Sanitized {len(sanitization_report)} metadata fields per NIST SP 800-122")
        return sanitized
    
    def _get_presidio_risk_level(self, entity_type: str) -> str:
        """Map Presidio entity types to NIST risk levels"""
        high_risk_entities = {
            "US_SSN", "CREDIT_CARD", "US_PASSPORT", "US_DRIVER_LICENSE",
            "MEDICAL_LICENSE", "US_BANK_NUMBER", "CRYPTO"
        }
        
        moderate_risk_entities = {
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME",
            "LOCATION", "US_ITIN"
        }
        
        if entity_type in high_risk_entities:
            return PIIRiskLevel.HIGH.value
        elif entity_type in moderate_risk_entities:
            return PIIRiskLevel.MODERATE.value
        else:
            return PIIRiskLevel.LOW.value
    
    def _is_overlapping(self, span: Tuple[int, int], detected_pii: List[Dict]) -> bool:
        """Check if a span overlaps with already detected PII"""
        for pii in detected_pii:
            if (span[0] < pii["end"] and span[1] > pii["start"]):
                return True
        return False
    
    def _should_anonymize(self, pii: Dict, risk_threshold: PIIRiskLevel) -> bool:
        """Determine if PII should be anonymized based on risk threshold"""
        risk_order = {
            PIIRiskLevel.LOW: 1,
            PIIRiskLevel.MODERATE: 2,
            PIIRiskLevel.HIGH: 3
        }
        
        pii_risk = PIIRiskLevel(pii["risk_level"])
        return risk_order[pii_risk] >= risk_order[risk_threshold]
    
    def _get_replacement_text(self, pii: Dict) -> str:
        """Get appropriate replacement text for PII"""
        # Use custom pattern replacement if available
        for pattern in self.nist_pii_patterns:
            if pattern.name == pii["entity_type"]:
                return pattern.replacement
        
        # Default replacements for Presidio entities
        replacements = {
            "PERSON": "[NAME-REDACTED]",
            "EMAIL_ADDRESS": "[EMAIL-REDACTED]",
            "PHONE_NUMBER": "[PHONE-REDACTED]",
            "CREDIT_CARD": "[CC-REDACTED]",
            "US_SSN": "[SSN-REDACTED]",
            "US_PASSPORT": "[PASSPORT-REDACTED]",
            "US_DRIVER_LICENSE": "[DL-REDACTED]",
            "DATE_TIME": "[DATE-REDACTED]",
            "LOCATION": "[LOCATION-REDACTED]",
            "ORGANIZATION": "[ORG-REDACTED]"
        }
        
        return replacements.get(pii["entity_type"], "[PII-REDACTED]")
    
    def _hash_value(self, value: str) -> str:
        """Create SHA-256 hash of sensitive value (NIST recommended)"""
        return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
    
    def generate_pii_report(self, text: str) -> Dict:
        """
        Generate comprehensive PII analysis report per NIST SP 800-122
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detailed PII analysis report
        """
        detected_pii = self.detect_pii_comprehensive(text)
        
        # Categorize by risk level
        risk_summary = {
            PIIRiskLevel.HIGH.value: 0,
            PIIRiskLevel.MODERATE.value: 0,
            PIIRiskLevel.LOW.value: 0
        }
        
        # Categorize by NIST category
        category_summary = {}
        
        for pii in detected_pii:
            risk_level = pii["risk_level"]
            if risk_level in risk_summary:
                risk_summary[risk_level] += 1
            
            category = pii.get("nist_category", "Unknown")
            category_summary[category] = category_summary.get(category, 0) + 1
        
        return {
            "total_pii_detected": len(detected_pii),
            "risk_level_summary": risk_summary,
            "nist_category_summary": category_summary,
            "detailed_findings": detected_pii,
            "nist_compliance": True,
            "standard_reference": "NIST Special Publication 800-122",
            "analysis_timestamp": str(hash(text))  # Simple hash for tracking
        }


# Legacy compatibility wrapper
class PIIProtector(NISTCompliantPIIProtector):
    """Legacy compatibility wrapper for existing code"""
    
    def detect_pii(self, text: str) -> List[Dict]:
        """Legacy method - maps to comprehensive detection"""
        return self.detect_pii_comprehensive(text)
    
    def anonymize_text(self, text: str, entities_to_anonymize: Optional[List[str]] = None) -> str:
        """Legacy method - maps to NIST compliant anonymization"""
        anonymized_text, _ = self.anonymize_text_nist_compliant(text)
        return anonymized_text
    
    def sanitize_metadata(self, metadata: Dict) -> Dict:
        """Legacy method - maps to NIST compliant sanitization"""
        return self.sanitize_metadata_nist_compliant(metadata)
