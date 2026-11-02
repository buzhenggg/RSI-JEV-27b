import argparse
import json
from pathlib import Path
from .bundle import verify_bundle


def main():
    parser = argparse.ArgumentParser(description='Verify or run the fixed mixed27b dual-LoRA engine')
    parser.add_argument('command', choices=['verify', 'predict', 'fetch-base'])
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--request', help='JSON containing state, questions, images and optional prompt')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'fetch-base':
        from huggingface_hub import snapshot_download
        manifest = json.loads((Path(args.bundle)/'manifest.json').read_text())
        snapshot_download(manifest['base_model'], revision=manifest['base_revision'],
            local_dir=args.base, allow_patterns=list(manifest['base_sha256']))
        print(json.dumps({'downloaded': True, 'base': args.base, 'revision': manifest['base_revision']}))
        return
    if args.command == 'verify':
        manifest, digest = verify_bundle(args.bundle, args.base)
        print(json.dumps({'passed': True, 'manifest_sha256': digest, 'active_adapters': manifest['active_adapters']}))
        return
    if not args.request:
        parser.error('predict requires --request')
    from .engine import Mixed27BEngine
    engine = Mixed27BEngine(bundle=args.bundle, base=args.base, device=args.device,
                           image_root=str(Path(args.request).resolve().parent))
    try:
        payload = json.loads(Path(args.request).read_text())
        response, raw = engine(**payload)
        print(json.dumps({'response': response, 'raw_output': raw}, ensure_ascii=False))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
