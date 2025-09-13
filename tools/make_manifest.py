import sys, json, hashlib, boto3
from urllib.parse import urlparse

# Usage:
#   python tools/make_manifest.py s3://bucket/path/to/file.csv data_manifests/walmart/train.manifest.json

def s3_stream_and_hash(s3_uri):
    u = urlparse(s3_uri)
    bucket, key = u.netloc, u.path.lstrip("/")
    s3 = boto3.client("s3")

    obj = s3.get_object(Bucket=bucket, Key=key)
    sha = hashlib.sha256()
    rows = 0

    for chunk in obj["Body"].iter_lines():
        if chunk is None:
            continue
        sha.update(chunk)
        rows += 1

    return rows, sha.hexdigest()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/make_manifest.py <s3_uri> <out_json>")
        sys.exit(1)

    s3_uri = sys.argv[1]
    out_json = sys.argv[2]

    rows, digest = s3_stream_and_hash(s3_uri)
    manifest = {
        "dataset": "walmart",
        "s3_path": s3_uri,
        "rows_including_header": rows,
        "sha256": digest,
    }
    with open(out_json, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {out_json}")