"""Run organizer validator when supplied; otherwise run strict local validator.
The task PDF contains a literal <ссылка организаторов> placeholder, so the
official script is not bundled or impersonated here.
"""
import subprocess,sys
from pathlib import Path
root=Path(__file__).parent
official=root/'official_validate_dataset.py'
cmd=[sys.executable,str(official if official.exists() else root/'dataset'/'validate_dataset.py'),'--root',str(root/'dataset')]
print('validator:', 'OFFICIAL' if official.exists() else 'LOCAL STRICT COMPATIBILITY CHECK')
raise SystemExit(subprocess.call(cmd))
