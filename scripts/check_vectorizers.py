import joblib
from pathlib import Path

models_dir = Path("models")

for f in models_dir.glob("*.pkl"):
    try:
        obj = joblib.load(f)
        if hasattr(obj, "get_feature_names_out"):
            n = len(obj.get_feature_names_out())
        elif hasattr(obj, "vocabulary_"):
            n = len(obj.vocabulary_)
        else:
            continue

        print(f.name, "->", n, "features")

    except Exception:
        pass
