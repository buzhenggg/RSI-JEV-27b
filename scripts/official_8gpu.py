"""Shard orchestration only; inference and scoring use the unchanged official SDK."""
import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    for name in ['suite', 'bundle', 'base', 'out']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--reference')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    out, suite = Path(args.out).resolve(), Path(args.suite).resolve()
    if args.worker is not None:
        from decision_index.vision.run import run
        run('mixed27b.engine:Mixed27BEngine', {'bundle': args.bundle, 'base': args.base},
            suite, out/'shards'/str(args.worker), rows_path=out/'inputs'/f'{args.worker}.jsonl',
            workers=1, chunk=16, resume=False)
        return

    out.mkdir(parents=True, exist_ok=False)
    children, logs = [], []

    def status(stage, **fields):
        value = {'stage': stage, 'time': stamp(), **fields}
        write(out/'pipeline_status.json', value)
        print(json.dumps(value), flush=True)

    try:
        status('preflight')
        gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        gpus = {int(line.split(',')[0]): int(line.split(',')[1]) for line in gpu_text.splitlines()}
        if not all(i in gpus and gpus[i] < 1000 for i in range(8)):
            raise RuntimeError(f'Expected eight idle GPUs; observed memory MiB: {gpus}')
        if importlib.metadata.version('mixed27b-decision-engine') != '0.2.0':
            raise RuntimeError('Use the installed minimal v0.2.0 Engine')
        from decision_index.vision.rows import verify
        check = verify(suite)
        write(out/'suite_verification.json', check)
        if not check['ok']:
            raise RuntimeError('Official suite verification failed; see suite_verification.json')
        rows_file = suite/'vision-rows.jsonl'
        rows = records(rows_file)
        expected = {r['id']: r for r in rows}
        if len(rows) != 12210 or len(expected) != len(rows):
            raise RuntimeError('Expected 12210 unique official rows')
        (out/'inputs').mkdir()
        for rank in range(8):
            (out/'inputs'/f'{rank}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[rank::8]))
        import mixed27b.engine as engine
        import decision_index.vision.run as official_run
        import decision_index.vision.score as official_score
        provenance = {
            'started_at': stamp(), 'engine_version': '0.2.0', 'rows': len(rows),
            'physical_gpus': list(range(8)), 'workers_per_process': 1, 'batch_size': 1,
            'sharding': 'Original suite order, round robin index modulo 8; no duplicated rows',
            'rows_sha256': sha(rows_file), 'orchestrator_sha256': sha(__file__),
            'engine_source_sha256': sha(engine.__file__),
            'official_runner_sha256': sha(official_run.__file__),
            'official_scorer_sha256': sha(official_score.__file__),
            'bundle': args.bundle, 'base': args.base,
            'bundle_files_sha256': {str(p.relative_to(args.bundle)): sha(p) for p in Path(args.bundle).rglob('*') if p.is_file()},
            'kernel_policy': 'Library default autotuning; no kernel override or probability-parity gate',
            'latency_scope': 'Eight independent single-GPU processes; not the required RTX PRO 6000 measurement',
            'reference': args.reference,
        }
        if args.reference:
            provenance['reference_sha256'] = sha(args.reference)
        write(out/'provenance.json', provenance)
        for rank in range(8):
            log = (out/f'worker-{rank}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
            env.pop('PYTHONPATH', None)
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', str(rank),
                   '--suite', str(suite), '--bundle', args.bundle, '--base', args.base, '--out', str(out)]
            children.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd='/tmp'))
        write(out/'processes.json', [{'physical_gpu': i, 'pid': p.pid, 'rows': len(rows[i::8])} for i, p in enumerate(children)])
        while True:
            codes = [p.poll() for p in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f'Worker failed: exit codes {codes}; original logs/results retained')
            progress = {}
            for rank in range(8):
                p = out/'shards'/str(rank)/'status.json'
                if p.exists():
                    progress[str(rank)] = json.loads(p.read_text())
            status('inference', completed=sum(v.get('completed', 0) for v in progress.values()), total=len(rows), shards=progress)
            if all(code == 0 for code in codes):
                break
            time.sleep(15)

        status('merging_and_scoring')
        raw_by_id, predictions, environments = {}, {}, []
        for rank in range(8):
            directory = out/'shards'/str(rank)
            environments.append(json.loads((directory/'environment.json').read_text()))
            for line in (directory/'results.jsonl').read_text().splitlines(keepends=True):
                record = json.loads(line)
                rid = record['run_id']
                if rid in raw_by_id or rid not in expected:
                    raise RuntimeError(f'Duplicate/unexpected run ID: {rid}')
                row = expected[rid]
                if record['status'] != 'ok':
                    raise RuntimeError(f'Non-successful result: {rid}: {record["status"]}')
                for key, value in row['_evaluation'].items():
                    if record.get(key) != value:
                        raise RuntimeError(f'Official evaluation metadata mismatch: {rid}/{key}')
                answer = record['response']['answers']['answer']
                probabilities = answer['probabilities']
                if set(probabilities) != set(row['questions']['answer']['criteria']):
                    raise RuntimeError(f'Option coverage mismatch: {rid}')
                vals = list(probabilities.values())
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in vals) or abs(sum(vals)-1) > .01 or answer['choice'] not in probabilities:
                    raise RuntimeError(f'Invalid output probabilities/choice: {rid}')
                raw_by_id[rid] = line
                predictions[rid] = answer
        if set(raw_by_id) != set(expected):
            raise RuntimeError('Missing predictions')
        with (out/'results.jsonl').open('x') as f:
            for row in rows:
                f.write(raw_by_id[row['id']])
        env = dict(environments[0])
        env.update(rows_path=str(rows_file), rows_sha256=sha(rows_file), sharded_run=True,
                   parallel_processes=8, physical_gpus=list(range(8)), shard_environments=environments,
                   latency='Per-request wall time from eight independent single-GPU runner processes; not official latency certification')
        env['effective_image_budget'] = {
            'max_image_pixels': max(e['effective_image_budget']['max_image_pixels'] for e in environments),
            'max_image_tokens': max(e['effective_image_budget']['max_image_tokens'] or 0 for e in environments),
            'rows_with_image_tokens': sum(e['effective_image_budget']['rows_with_image_tokens'] for e in environments),
        }
        write(out/'environment.json', env)
        scores = official_score.score_run(rows_file, out/'results.jsonl', out)
        if not scores['complete'] or any(b['answered'] != b['rows'] for b in scores['benchmarks'].values()):
            raise RuntimeError('Official scores are not complete and fully answered')
        comparison = {'current_public': scores['public_vision_index'], 'rows': len(rows),
                      'correct': sum(predictions[r['id']]['choice'] == r['expected']['answer'] for r in rows)}
        if args.reference:
            refs = {r['id']: r for r in records(args.reference)}
            if set(refs) != set(expected):
                raise RuntimeError('Reference IDs do not cover the same suite')
            ref_preds = {rid: chr(65 + r['prediction']) for rid, r in refs.items()}
            old_score = official_score.score(rows, ref_preds)
            comparison.update(reference_public=old_score['public_vision_index'],
                              public_delta=scores['public_vision_index']-old_score['public_vision_index'])
            differences = []; fixed = regressed = 0; max_probability_difference = 0.0
            for r in rows:
                rid = r['id']; old = ref_preds[rid]; new = predictions[rid]['choice']; gold = r['expected']['answer']
                if chr(65 + refs[rid]['gold']) != gold:
                    raise RuntimeError(f'Reference gold mismatch: {rid}')
                probabilities = predictions[rid]['probabilities']
                max_probability_difference = max(max_probability_difference, max(abs(probabilities[k]-v) for k,v in zip(sorted(probabilities),refs[rid]['probabilities'])))
                fixed += old != gold and new == gold
                regressed += old == gold and new != gold
                if old != new:
                    differences.append({'id':rid,'benchmark':r['benchmark'],'reference':old,'current':new,'gold':gold})
            comparison.update(changed_predictions=len(differences), fixed=fixed, regressed=regressed,
                              max_probability_abs_difference=max_probability_difference,
                              probability_difference_is_informational_only=True,
                              benchmarks={k:{'current':v,'reference':old_score['benchmarks'][k]} for k,v in scores['benchmarks'].items()})
            (out/'prediction_differences.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in differences))
        write(out/'comparison.json', comparison)
        write(out/'completion_check.json', {'passed':True,'rows':len(rows),'unique_ids':len(raw_by_id),
            'all_status_ok':True,'all_options_and_probabilities_valid':True,'official_score_complete':True,
            'merged_lines_unchanged':True,'results_sha256':sha(out/'results.jsonl'),
            'scores_sha256':sha(out/'vision-scores.json'),'finished_at':stamp()})
        write(out/'status.json', {'event':'complete','completed':len(rows),'counts':{'ok':len(rows)}})
        status('complete', **comparison)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
        status('failed', error=repr(exc))
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
