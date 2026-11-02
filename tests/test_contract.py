"""CPU contract/negative checks: do not allocate a model or read benchmark gold."""
import json
import tempfile
import unittest
from pathlib import Path
from mixed27b.bundle import sha256, verify_bundle
from mixed27b.engine import render_question
from decision_index.engines import Unsupported


class ContractTests(unittest.TestCase):
    def test_training_render_preserved(self):
        keys, prompt = render_question({}, {'type':'choice','instructions':'Which direction?', 'criteria':{'B':'right','A':'left'}})
        self.assertEqual(keys, ['A','B'])
        self.assertEqual(prompt, 'Which direction?\nA. left\nB. right\nAnswer with the option letter only.')

    def test_options_are_never_filtered(self):
        question = {'type':'choice','instructions':'Pick','criteria':{str(i):str(i) for i in range(27)}}
        with self.assertRaises(Unsupported): render_question({}, question)
        with self.assertRaises(Unsupported): render_question({}, {'type':'noul'})

    def test_nonempty_context_is_not_discarded(self):
        _, prompt = render_question({'context':'visible'}, {'type':'choice','instructions':'Pick','criteria':{'x':'one','y':'two'}})
        self.assertIn('visible',prompt)

    def test_missing_corrupt_and_inactive_components_fail(self):
        with tempfile.TemporaryDirectory() as d:
            bundle=Path(d)/'bundle';base=Path(d)/'base';bundle.mkdir();base.mkdir()
            names=['default/adapter_config.json','default/adapter_model.safetensors','mixed/adapter_config.json','mixed/adapter_model.safetensors','decision_head.safetensors']
            for n in names:
                p=bundle/n;p.parent.mkdir(exist_ok=True);p.write_bytes(n.encode())
            (base/'config.json').write_text('{}')
            manifest={'active_adapters':['default','mixed'],'bundle_sha256':{n:sha256(bundle/n) for n in names},'base_sha256':{'config.json':sha256(base/'config.json')}}
            path=bundle/'manifest.json';path.write_text(json.dumps(manifest));verify_bundle(bundle,base)
            victim=bundle/'mixed/adapter_model.safetensors';original=victim.read_bytes();victim.unlink()
            with self.assertRaises(FileNotFoundError):verify_bundle(bundle,base)
            victim.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):verify_bundle(bundle,base)
            victim.write_bytes(original);manifest['active_adapters']=['default'];path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_bundle(bundle,base)


if __name__=='__main__':unittest.main()
