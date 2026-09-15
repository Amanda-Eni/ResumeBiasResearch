"""
Quick verification that all required packages are installed and working.
Run with: python verify_setup.py
"""
import sys

def check_package(name, import_name=None):
    """Check if a package is installed and print its version."""
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name}: NOT INSTALLED ({e})")
        return False

print("=" * 60)
print("ResumeBiasResearch - Package Verification")
print("=" * 60)
print(f"\nPython: {sys.version}")
print(f"Executable: {sys.executable}\n")

print("Core data science packages:")
packages = [
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('scikit-learn', 'sklearn'),
    ('xgboost', 'xgboost'),
    ('imbalanced-learn', 'imblearn'),
]

all_ok = True
for name, import_name in packages:
    if not check_package(name, import_name):
        all_ok = False

print("\nFairness and explainability:")
for name, import_name in [('fairlearn', 'fairlearn'), ('shap', 'shap')]:
    if not check_package(name, import_name):
        all_ok = False

print("\nVisualization:")
for name, import_name in [('matplotlib', 'matplotlib'), ('seaborn', 'seaborn')]:
    if not check_package(name, import_name):
        all_ok = False

print("\nNLP:")
for name, import_name in [('spacy', 'spacy'), ('nltk', 'nltk')]:
    if not check_package(name, import_name):
        all_ok = False

print("\nspaCy English model:")
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("John Smith has 5 years of experience.")
    print(f"  ✓ en_core_web_sm loaded. Tokens: {[t.text for t in doc]}")
except Exception as e:
    print(f"  ✗ en_core_web_sm: FAILED ({e})")
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✓ All packages verified. You're ready to proceed!")
else:
    print("✗ Some packages are missing. Install with: pip install <package>")
print("=" * 60)