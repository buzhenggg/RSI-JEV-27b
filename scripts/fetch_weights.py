"""Download the complete evaluated model from one release repository."""
import argparse
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument('--local-dir', default='.')
p.add_argument('--revision', required=True, help='Immutable model commit from release.json')
a = p.parse_args()
snapshot_download('k2021211255/RSI-JEV-27b', revision=a.revision, local_dir=a.local_dir,
                  allow_patterns=['base/*', 'base/**/*', 'bundle/*', 'bundle/**/*'])
