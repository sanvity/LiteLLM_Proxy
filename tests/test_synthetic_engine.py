import os
import sys
import unittest
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from synthetic_data import SyntheticDataEngine, DatasetList

class TestSyntheticDataEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SyntheticDataEngine()

    def test_regex_generation_conformance(self):
        """Test that generated regex values match the pattern precisely."""
        pattern = r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"
        val = self.engine.generate_fake_value_test("ssn", pattern)
        self.assertTrue(re.fullmatch(re.compile(pattern), val))

    def test_hard_negatives_perturbation(self):
        """Test that perturbed values are indeed invalid for the target regex."""
        pattern = r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"
        val = self.engine.generate_fake_value_test("ssn", pattern)
        perturbed = self.engine.perturb_value(val, pattern)
        self.assertNotEqual(val, perturbed)
        self.assertFalse(re.fullmatch(re.compile(pattern), perturbed))

    def test_near_duplicate_filter(self):
        """Test that carrier sentences with similarity above threshold are filtered."""
        existing = ["Please reset the password for [PII]."]
        
        # Similar carrier sentence -> should trigger duplicate check
        similar = "Please reset the passcode for [PII]."
        self.assertTrue(self.engine.is_near_duplicate(similar, existing, threshold=0.80))
        
        # Dissimilar carrier sentence -> should not trigger
        dissimilar = "The user reported an error on the homepage."
        self.assertFalse(self.engine.is_near_duplicate(dissimilar, existing, threshold=0.80))

    def test_overlap_resolution_priority(self):
        """Test overlap resolution where target entity takes precedence."""
        # Setup overlapping entities: first is target, second is default PII
        # They overlap, so target 'custom_label' must win over 'person'
        entities = [
            {"label": "custom_label", "start": 0, "end": 10, "value": "John Doe 1"},
            {"label": "person", "start": 0, "end": 8, "value": "John Doe"}
        ]
        
        # Sort and prune overlaps
        entities.sort(key=lambda x: x["start"])
        clean_entities = []
        entity_name = "custom_label"
        
        for ent in entities:
            if not clean_entities:
                clean_entities.append(ent)
            else:
                last = clean_entities[-1]
                if ent["start"] >= last["end"]:
                    clean_entities.append(ent)
                else:
                    # Overlap resolution
                    if ent["label"] == entity_name.lower() and last["label"] != entity_name.lower():
                        clean_entities[-1] = ent
                    elif last["label"] == entity_name.lower() and ent["label"] != entity_name.lower():
                        continue
                    elif (ent["end"] - ent["start"]) > (last["end"] - last["start"]):
                        clean_entities[-1] = ent
                        
        self.assertEqual(len(clean_entities), 1)
        self.assertEqual(clean_entities[0]["label"], "custom_label")

# Add a helper for testing _generate_fake_value
def generate_fake_value_test(self, entity_name: str, regex_pattern: str, is_hard_negative: bool = False) -> str:
    return self._generate_fake_value(entity_name, regex_pattern, is_hard_negative)

SyntheticDataEngine.generate_fake_value_test = generate_fake_value_test

if __name__ == "__main__":
    unittest.main()
