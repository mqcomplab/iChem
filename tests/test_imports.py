#!/usr/bin/env python3
"""Test script to verify bblean imports work correctly after cleanup"""

print("Testing iChem.bblean imports...")

# Test main imports
from iChem.bblean import BitBirch
print("✓ BitBirch imported successfully")

from iChem.bblean.utils import min_safe_uint
print("✓ min_safe_uint imported successfully")

from iChem.bblean._memory import _ArrayMemPagesManager, _mmap_file_and_madvise_sequential
print("✓ _ArrayMemPagesManager imported successfully")
print("✓ _mmap_file_and_madvise_sequential imported successfully")

# Test that BitBirch can be instantiated
bb = BitBirch()
print("✓ BitBirch instance created successfully")

print("\n✅ All imports working correctly!")
