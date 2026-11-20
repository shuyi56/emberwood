"""Garbage collection for the shared project folder and the render queue's results (John, 2026-11-20).

    python3 rq_gc.py                          # dry run: prints what would go, writes nothing
    python3 rq_gc.py --apply                  # deletes, and writes stats/gc-<date>.md and gc-<date>.json
    options: --root DIR (default /mnt/project-files)  --repo DIR (a clone of the repo, for references)
             --refs FILE ...  (more text to treat as references: open PR bodies, a thread's current files)
             --hold PATH ...  (paths that stay whatever the rules say: a thread's current work)

What goes (each rule keeps anything referenced; see "Kept" below):
  results/  draft       a check/draft result older than DRAFT_DAYS (the nightly job's rule since 2026-11-19)
            superseded  a result older than GRACE_H with a newer result for the same thread and view command
            old sheet   a daily sheet older than DRAFT_DAYS that is not one of the newest KEEP_SHEETS
            failed      a -failed folder older than DRAFT_DAYS
  folders   superseded  NAME-vN-REST.EXT images, .glb and .blend under a higher version of the same series:
                        the newest version stays, and the newest two while the series changed in the last day
            temp        Blender backups (.blend1, .blend2), __pycache__, .pyc, .tmp, editor ~ and lock files
            duplicate   a byte-identical copy of another file (the referenced or the older copy stays)
            fold cache  a solved-folds cache entry (NAME-<hex key>.npy|npz, e.g. hose/boot-cache) whose key no render has
                        used since a newer key of the same piece replaced it on the same branch
Kept, whatever its age:
  * anything named in rejections.jsonl, in a pending, rendering or conflict request, in a text file of the project
    folder outside render-queue/, in the repo (main and every pushed branch), or in a --refs file
  * every final result of main's current commit, the newest result of each thread's view command, and the newest
    KEEP_SHEETS sheets
  * uploads/ (John's own files), code and text files, and anything touched in the last GRACE_H hours
  * the newest version of every series and the fold cache entry each branch last used
Borderline items (never deleted, listed in the manifest): fold cache entries no render log names (the solver ran
outside the queue, so the key may be current work), and superseded versions whose series is still being written.
"""
import argparse, collections, datetime as dt, hashlib, json, os, re, shutil, subprocess, sys, time

DRAFT_DAYS = 3
GRACE_H = 24
KEEP_SHEETS = 2
MEDIA = ('.png', '.gif', '.jpg', '.jpeg', '.webp', '.glb', '.blend', '.mp4', '.exr', '.npz', '.npy')
SERIES_EXT = ('.png', '.gif', '.jpg', '.jpeg', '.webp', '.glb', '.blend', '.mp4', '.exr')
TEXT = ('.md', '.json', '.jsonl', '.py', '.js', '.mjs', '.html', '.txt', '.sh', '.ps1', '.csv', '.yaml', '.yml')
TEMP = re.compile(r'(\.blend\d+|\.pyc|\.tmp|~|\.part)$|^\.~lock|^~\$')
CACHE = re.compile(r'^(.*?)-([0-9a-f]{12,})\.(npy|npz)$')     # NAME-<key>.npy|npz: a solved-folds cache entry
SKIP_TOP = ('uploads',)
VER = re.compile(r'^(.*?)[-_]v(\d+)([-_.].*)$')
KEY = re.compile(r'\b((?:views|glb|sheet|frames)-[0-9a-f]{12})\b')


def ts(t):
    return dt.datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=dt.timezone.utc).timestamp() if t else None


def size(p):
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs
                   if not os.path.islink(os.path.join(r, f)))
    return os.path.getsize(p)


def newest_mtime(p):
    if os.path.isdir(p):
        return max([os.path.getmtime(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs] or [os.path.getmtime(p)])
    return os.path.getmtime(p)


class Refs:
    """Everything that may name a file or a result key. A candidate is referenced when its key, its path relative to
    the root, or its file name appears in any of it."""

    def __init__(self, root, rq, repo, extra):
        self.chunks = []
        for p in extra or []:
            self.chunks.append(open(p, errors='ignore').read())
        rj = os.path.join(rq, 'rejections.jsonl')
        if os.path.exists(rj):
            self.chunks.append(open(rj, errors='ignore').read())
        try:
            q = json.load(open(os.path.join(rq, 'queue.json')))
            live = [r for r in q.get('requests', []) if r.get('state') in ('pending', 'rendering', 'conflict')]
            self.chunks.append(json.dumps(live))
        except (OSError, ValueError):
            pass
        rqrel = os.path.relpath(rq, root)
        for r, ds, fs in os.walk(root):
            rel = os.path.relpath(r, root)
            if rel.split(os.sep)[0] in (rqrel,) + SKIP_TOP:
                ds[:] = []
                continue
            ds[:] = [d for d in ds if d != '__pycache__']
            for f in fs:
                if f.lower().endswith(TEXT) and not f.startswith('gc-'):
                    p = os.path.join(r, f)
                    if os.path.getsize(p) < 8 << 20:
                        self.chunks.append(open(p, errors='ignore').read())
        self.blob = '\n'.join(self.chunks)
        self.repo, self.repo_hits = repo, None

    def load_repo(self, needles):
        """One git grep per branch for all the names at once."""
        self.repo_hits = set()
        if not self.repo or not needles:
            return
        refs = subprocess.run(['git', '-C', self.repo, 'for-each-ref', '--format=%(refname)', 'refs/remotes/origin'],
                              capture_output=True, text=True).stdout.split()
        refs = [r for r in refs if not r.endswith('/HEAD')] or ['HEAD']
        pat = '\n'.join(sorted(needles))
        for ref in refs:
            out = subprocess.run(['git', '-C', self.repo, 'grep', '-h', '-o', '-F', '-f', '-', ref, '--'],
                                 input=pat, capture_output=True, text=True).stdout
            self.repo_hits.update(l.strip() for l in out.splitlines())

    def names(self, *needles):
        return [n for n in needles if n and (n in self.blob or (self.repo_hits and n in self.repo_hits))]


def main_commits(repo):
    if not repo:
        return set(), None
    out = subprocess.run(['git', '-C', repo, 'rev-list', 'origin/main'], capture_output=True, text=True).stdout.split()
    return set(out), (out[0] if out else None)


def result_candidates(rq, refs, main_set, main_head, now, hold):
    res = os.path.join(rq, 'results')
    try:
        q = json.load(open(os.path.join(rq, 'queue.json')))
    except (OSError, ValueError):
        q = {'requests': []}
    owner = collections.defaultdict(list)          # job key -> requests it served
    for r in q.get('requests', []):
        for k in r.get('jobs') or []:
            owner[k].append(r)
    rows = []
    for d in sorted(os.listdir(res)) if os.path.isdir(res) else []:
        p = os.path.join(res, d)
        try:
            m = json.load(open(os.path.join(p, 'manifest.json')))
        except (OSError, ValueError):
            m = {}
        key = d[:-7] if d.endswith('-failed') else d
        fin = ts(m.get('finished')) or newest_mtime(p)
        param = m.get('param') or ''
        rs = owner.get(key, [])
        rows.append({'dir': d, 'path': p, 'key': key, 'kind': m.get('kind') or d.split('-')[0], 'param': param,
                     'commit': m.get('commit') or '', 'finished': fin, 'failed': d.endswith('-failed'),
                     'check': '[quality check]' in param or 'draft' in param.lower()
                              or (m.get('settings') or {}).get('quality') == 'check',
                     'threads': sorted({r['thread'] for r in rs}), 'branches': sorted({r['branch'] for r in rs})})
    newest = {}                                     # (thread, param) -> newest finish
    for r in rows:
        for t in r['threads'] or ['?']:
            k = (t, r['kind'], r['param'])
            newest[k] = max(newest.get(k, 0), r['finished'])
    sheets = sorted((r for r in rows if r['kind'] == 'sheet' and not r['failed']), key=lambda r: -r['finished'])
    keep_sheets = {r['dir'] for r in sheets[:KEEP_SHEETS]}
    out = []
    for r in rows:
        age_d = (now - r['finished']) / 86400
        why = None
        if r['failed'] and age_d > DRAFT_DAYS:
            why = 'failed result older than %d days' % DRAFT_DAYS
        elif r['kind'] == 'sheet':
            if r['dir'] not in keep_sheets and age_d > DRAFT_DAYS:
                why = 'daily sheet older than %d days, %d newer sheets kept' % (DRAFT_DAYS, KEEP_SHEETS)
        elif r['check'] and age_d > DRAFT_DAYS and not (r['commit'] and r['commit'] == main_head):
            why = 'draft/check render older than %d days' % DRAFT_DAYS
        elif age_d * 24 > GRACE_H and r['threads'] and all(newest[(t, r['kind'], r['param'])] > r['finished']
                                                            for t in r['threads']):
            if r['commit'] and r['commit'] == main_head and not r['check']:
                pass                                # main's current finals are the "before" cache
            else:
                why = 'superseded: a newer render of the same view for the same thread exists'
        if not why:
            continue
        named = refs.names(r['key'], 'render-queue/results/' + r['dir'])
        if named or any(h in r['path'] for h in hold):
            out.append(dict(r, rule='kept: referenced', why=why, refs=named))
            continue
        out.append(dict(r, rule='delete', why=why))
    return out


def folder_files(root, rq):
    rqrel = os.path.relpath(rq, root)
    for r, ds, fs in os.walk(root):
        rel = os.path.relpath(r, root)
        top = rel.split(os.sep)[0]
        if top in SKIP_TOP or (top == rqrel and rel.startswith(os.path.join(rqrel, 'results'))):
            ds[:] = []
            continue
        if top.startswith('_trash'):
            ds[:] = []
            continue
        for d in list(ds):
            if d == '__pycache__':
                yield os.path.join(r, d), 'dir'
                ds.remove(d)
        for f in fs:
            yield os.path.join(r, f), 'file'


def folder_candidates(root, rq, refs, now, hold):
    out, series, by_hash = [], collections.defaultdict(list), collections.defaultdict(list)
    files = list(folder_files(root, rq))
    for p, typ in files:
        rel = os.path.relpath(p, root)
        name = os.path.basename(p)
        age_h = (now - newest_mtime(p)) / 3600
        if typ == 'dir':
            if age_h > 1:
                out.append({'path': p, 'rel': rel, 'rule': 'delete', 'why': 'temp: Python bytecode cache'})
            continue
        if TEMP.search(name):
            if age_h > 1:
                out.append({'path': p, 'rel': rel, 'rule': 'delete', 'why': 'temp: editor or Blender backup file'})
            continue
        low = name.lower()
        if not low.endswith(MEDIA):
            continue
        m = VER.match(name)
        if m and low.endswith(SERIES_EXT):
            series[(os.path.dirname(p), m[1], m[3])].append((int(m[2]), p, age_h))
        sz = os.path.getsize(p)
        if sz > 64 << 10 and not low.endswith(('.npy', '.npz')):   # (cache entries may share folds under two keys)
            by_hash[sz].append(p)
    # superseded versions
    for (d, stem, rest), vs in series.items():
        vs.sort()
        if len(vs) < 2:
            continue
        active = min(a for _, _, a in vs) < GRACE_H
        keep_n = 2 if active else 1
        for v, p, age_h in vs[:-keep_n]:
            rel = os.path.relpath(p, root)
            why = 'superseded by %s' % os.path.basename(vs[-1][1])
            if age_h < GRACE_H / 2:
                out.append({'path': p, 'rel': rel, 'rule': 'borderline', 'why': why + ', but written in the last 12 hours'})
                continue
            out.append({'path': p, 'rel': rel, 'rule': 'delete', 'why': why})
    # duplicates
    for sz, ps in by_hash.items():
        if len(ps) < 2:
            continue
        groups = collections.defaultdict(list)
        for p in ps:
            h = hashlib.sha256()
            with open(p, 'rb') as fh:
                for b in iter(lambda: fh.read(1 << 20), b''):
                    h.update(b)
            groups[h.hexdigest()].append(p)
        for g in groups.values():
            if len(g) < 2:
                continue
            g.sort(key=lambda p: (not refs.names(os.path.relpath(p, root), os.path.basename(p)),
                                  os.path.getmtime(p), len(p)))
            for p in g[1:]:
                out.append({'path': p, 'rel': os.path.relpath(p, root), 'rule': 'delete',
                            'why': 'duplicate of %s' % os.path.relpath(g[0], root)})
    out += cache_candidates(root, rq, now)
    # references and holds
    seen, final = set(), []
    for c in out:
        if c['path'] in seen:
            continue
        seen.add(c['path'])
        if any(c['rel'].startswith(h) or c['path'].startswith(h) for h in hold):
            c['rule'] = 'kept: held'
        elif c['rule'] == 'delete' and not c['why'].startswith(('temp', 'fold cache')):
            named = refs.names(c['rel'], os.path.basename(c['path']))
            if named:
                c['rule'], c['refs'] = 'kept: referenced', named
        final.append(c)
    return final


def cache_candidates(root, rq, now):
    """A fold cache entry is orphaned when the render logs show a newer key of the same piece replacing it on every
    branch that used it. Entries no log names are borderline (the solver ran outside the queue)."""
    res = os.path.join(rq, 'results')
    try:
        q = json.load(open(os.path.join(rq, 'queue.json')))
    except (OSError, ValueError):
        q = {'requests': []}
    branch_of = collections.defaultdict(set)
    for r in q.get('requests', []):
        for k in r.get('jobs') or []:
            branch_of[k].add(r['branch'])
    entries = {}
    for p, typ in folder_files(root, rq):
        m = CACHE.match(os.path.basename(p)) if typ == 'file' else None
        if m:
            entries[m[2]] = {'path': p, 'kind': (os.path.dirname(p), m[1]), 'uses': []}
    if not entries:
        return []
    for dd in os.listdir(res) if os.path.isdir(res) else []:
        p = os.path.join(res, dd)
        logs = [os.path.join(p, x) for x in os.listdir(p) if x.endswith('.log')] if os.path.isdir(p) else []
        txt = ''.join(open(l, errors='ignore').read() for l in logs)
        if not txt:
            continue
        try:
            fin = ts(json.load(open(os.path.join(p, 'manifest.json'))).get('finished')) or os.path.getmtime(p)
        except (OSError, ValueError):
            fin = os.path.getmtime(p)
        for k, e in entries.items():
            if k in txt:
                for b in branch_of.get(dd.replace('-failed', ''), {'?'}):
                    e['uses'].append((fin, b))
    last = {}                                       # (kind, branch) -> (time, key) of the newest use
    for k, e in entries.items():
        for t, b in e['uses']:
            if t > last.get((e['kind'], b), (0, None))[0]:
                last[(e['kind'], b)] = (t, k)
    newest_kind = {}
    for k, e in entries.items():
        mt = os.path.getmtime(e['path'])
        if mt > newest_kind.get(e['kind'], (0, None))[0]:
            newest_kind[e['kind']] = (mt, k)
    out = []
    for k, e in entries.items():
        rel = os.path.relpath(e['path'], root)
        age_h = (now - os.path.getmtime(e['path'])) / 3600
        if newest_kind[e['kind']][1] == k or age_h < GRACE_H:
            continue
        if not e['uses']:
            out.append({'path': e['path'], 'rel': rel, 'rule': 'borderline',
                        'why': 'fold cache: no queue render used this key (it may come from a run outside the queue)'})
        elif all(last[(e['kind'], b)][1] != k for _, b in e['uses']):
            out.append({'path': e['path'], 'rel': rel, 'rule': 'delete',
                        'why': 'fold cache: orphaned, a newer key replaced it on %s' %
                               ', '.join(sorted({b for _, b in e['uses']}))})
    return out


def mb(n):
    return '%.1f MB' % (n / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/project-files')
    ap.add_argument('--rq', default=None)
    ap.add_argument('--repo', default=None)
    ap.add_argument('--refs', nargs='*', default=[])
    ap.add_argument('--hold', nargs='*', default=[])
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    rq = os.path.abspath(a.rq or os.path.join(root, 'render-queue'))
    now = time.time()
    refs = Refs(root, rq, a.repo, a.refs)
    main_set, main_head = main_commits(a.repo)
    pre = [os.path.basename(p) for p, _ in folder_files(root, rq)]
    keys = os.listdir(os.path.join(rq, 'results')) if os.path.isdir(os.path.join(rq, 'results')) else []
    refs.load_repo({n for n in pre + keys if len(n) > 6})
    cands = result_candidates(rq, refs, main_set, main_head, now, a.hold)
    for c in cands:
        c['rel'] = os.path.relpath(c['path'], root)
    cands += folder_candidates(root, rq, refs, now, a.hold)
    for c in cands:
        c['bytes'] = size(c['path']) if os.path.exists(c['path']) else 0
    dele = [c for c in cands if c['rule'] == 'delete']
    total = sum(c['bytes'] for c in dele)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    lines = ['# Garbage collection %s%s' % (stamp, '' if a.apply else ' (dry run)'), '',
             '%s %d items, %s.' % ('Deleted' if a.apply else 'Would delete', len(dele), mb(total)), '']
    by_why = collections.defaultdict(list)
    for c in dele:
        by_why[c['why'].split(':')[0].split(' by ')[0].split(' of ')[0]].append(c)
    for w, cs in sorted(by_why.items(), key=lambda kv: -sum(c['bytes'] for c in kv[1])):
        lines.append('## %s (%d, %s)' % (w, len(cs), mb(sum(c['bytes'] for c in cs))))
        lines += ['- `%s` %s: %s' % (c['rel'], mb(c['bytes']), c['why']) for c in sorted(cs, key=lambda c: c['rel'])]
        lines.append('')
    keep = [c for c in cands if c['rule'] != 'delete']
    if keep:
        lines.append('## Kept or borderline (not deleted)')
        lines += ['- `%s` %s: %s; %s%s' % (c['rel'], mb(c['bytes']), c['rule'], c['why'],
                                          (' (named in: %s)' % ', '.join(c.get('refs', [])[:2])) if c.get('refs') else '')
                  for c in sorted(keep, key=lambda c: c['rel'])]
    report = '\n'.join(lines) + '\n'
    if a.apply:
        for c in dele:
            if os.path.isdir(c['path']):
                shutil.rmtree(c['path'], ignore_errors=True)
            elif os.path.exists(c['path']):
                os.remove(c['path'])
        os.makedirs(os.path.join(rq, 'stats'), exist_ok=True)
        base = os.path.join(rq, 'stats', 'gc-%s' % stamp)
        n = 1
        while os.path.exists(base + '.md'):
            n += 1
            base = os.path.join(rq, 'stats', 'gc-%s-%d' % (stamp, n))
        open(base + '.md', 'w').write(report)
        json.dump({'at': stamp, 'deleted_bytes': total,
                   'items': [{k: c.get(k) for k in ('rel', 'bytes', 'rule', 'why', 'refs')} for c in cands]},
                  open(base + '.json', 'w'), indent=1)
        print('manifest: %s.md' % base)
    if a.json:
        print(json.dumps({'deleted_bytes': total, 'items': [{k: c.get(k) for k in ('rel', 'bytes', 'rule', 'why')}
                                                            for c in cands]}, indent=1))
    else:
        sys.stdout.write(report)


if __name__ == '__main__':
    main()
