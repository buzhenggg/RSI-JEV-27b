"""Download the two adapters and head; no model merge or runtime check."""
import argparse
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument('--local-dir', default='.')
p.add_argument('--revision', default='main', help='Use a published immutable commit when reproducing a release')
a = p.parse_args()
snapshot_download('k2021211255/RSI-JEV-27b', revision=a.revision, local_dir=a.local_dir,
                  allow_patterns=['bundle/*', 'bundle/**/*'])
