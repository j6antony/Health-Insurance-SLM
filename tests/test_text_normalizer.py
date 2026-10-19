
from src.preprocessing.text_normalizer import normalize_text


sample_text = """
Hospital    Information

Patient   Name :   John Smith
Total   Amount :   $1,250.00


Medicine: Paracetamol 500mg
"""

cleaned = normalize_text(sample_text)

print("--- Original ---")
print(sample_text)

print("\n--- Normalized ---")
print(cleaned)

assert "Hospital Information" in cleaned
assert "Patient Name : John Smith" in cleaned
assert "$1,250.00" in cleaned
assert "Paracetamol 500mg" in cleaned

print("\nText normalization test passed!")
