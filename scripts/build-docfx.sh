#!/usr/bin/env bash
# Build the DocFX site the repeatable way: metadata + build + build-host scrub.
# The scrub is a BUILD STEP (not a hand patch): docfx metadata bakes the build
# host's absolute path into manifest.json source_base_path; per the anonymity
# rule no build-host path may ship, so it is normalized to a neutral value on
# every build. Re-run this script instead of raw docfx commands.
set -euo pipefail
cd "$(dirname "$0")/.."
dotnet tool restore
dotnet tool run docfx metadata
dotnet tool run docfx build
python3 - << 'PYEOF'
import json, glob, os
for mf in glob.glob('docs/api-site/manifest.json'):
    m = json.load(open(mf))
    if 'source_base_path' in m:
        m['source_base_path'] = '.'
    json.dump(m, open(mf, 'w'), indent=2)
print('manifest scrubbed')
PYEOF
python3 - << 'PYEOF'
import re, os, glob
base='docs/api-site'; link_re=re.compile(r'href="([^"]+)"')
broken=[(f,h) for f in glob.glob(base+'/**/*.html',recursive=True)
        for h in link_re.findall(open(f).read())
        if not h.startswith(('http','mailto','#'))
        and not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(f),h.split('#')[0])))]
print('BROKEN:', len(broken))
PYEOF
