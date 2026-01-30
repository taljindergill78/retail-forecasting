"""
Build manifest JSONs for S3 source objects (fingerprint: path + sha256 + row count).
Reads S3 URIs from params.yaml and calls make_manifest for each.
Used by the DVC 'manifests' stage so the pipeline invalidates when raw S3 data changes.
"""
import os
import sys
import subprocess

# Project root = parent of tools/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS_PATH = os.path.join(ROOT, "params.yaml")
MANIFEST_DIR = os.path.join(ROOT, "data_manifests", "walmart")


def load_params():
    import yaml
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    manifests = params.get("manifests") or {}
    train_uri = manifests.get("train_s3_uri") or ""
    features_uri = manifests.get("features_s3_uri") or ""
    stores_uri = manifests.get("stores_s3_uri") or ""

    os.makedirs(MANIFEST_DIR, exist_ok=True)
    make_manifest = os.path.join(ROOT, "tools", "make_manifest.py")

    specs = [
        (train_uri, os.path.join(MANIFEST_DIR, "train.manifest.json")),
        (features_uri, os.path.join(MANIFEST_DIR, "features.manifest.json")),
        (stores_uri, os.path.join(MANIFEST_DIR, "stores.manifest.json")),
    ]

    missing = [label for (uri, _), label in zip(specs, ["train_s3_uri", "features_s3_uri", "stores_s3_uri"]) if not (uri and uri.strip())]
    if missing:
        print("Skipping manifest stage: the following params are empty in params.yaml:", missing)
        print("Set manifests.train_s3_uri, manifests.features_s3_uri, manifests.stores_s3_uri to S3 URIs to enable.")
        # Write empty placeholder JSONs so DVC still has outputs and the pipeline can run
        for _, out_path in specs:
            with open(out_path, "w") as f:
                f.write("{}\n")
        return

    for s3_uri, out_path in specs:
        if not (s3_uri and s3_uri.strip()):
            with open(out_path, "w") as f:
                f.write("{}\n")
            continue
        rc = subprocess.call([sys.executable, make_manifest, s3_uri.strip(), out_path])
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
