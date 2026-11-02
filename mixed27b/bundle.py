"""Model manifest verification; never fall back to an unadapted base."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_bundle(bundle, base):
    bundle, base = Path(bundle).resolve(), Path(base).resolve()
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['active_adapters'] != ['default', 'mixed']:
        raise ValueError('Manifest must activate both default and mixed adapters')
    required = {
        'default/adapter_config.json', 'default/adapter_model.safetensors',
        'mixed/adapter_config.json', 'mixed/adapter_model.safetensors',
        'decision_head.safetensors',
    }
    if not required <= manifest['bundle_sha256'].keys():
        raise ValueError('Incomplete adapter/head manifest')
    for root, hashes in [(bundle, manifest['bundle_sha256']), (base, manifest['base_sha256'])]:
        for name, expected in hashes.items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Invalid manifest path: {name}')
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f'Required model component missing: {path}')
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f'SHA256 mismatch: {path}; expected {expected}, got {actual}')
    return manifest, sha256(manifest_path)
