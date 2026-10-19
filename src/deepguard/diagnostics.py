"""Check the actual local environment before opening the upload interface."""
from pathlib import Path
import sys
import shutil
from .utils import ROOT,config,sha256


def diagnose(check_encoder=False):
    report={'python':sys.executable,'project_folder':str(ROOT),'checks':[]}
    def check(name,action):
        try:
            detail=action()
            report['checks'].append(dict(name=name,ok=True,detail=detail))
        except Exception as exc:
            report['checks'].append(dict(name=name,ok=False,detail=f'{type(exc).__name__}: {exc}'))
    def ffprobe():
        path=shutil.which('ffprobe')
        if not path:raise RuntimeError('Install FFmpeg with FFprobe and reopen the terminal. See INSTALLATION.md.')
        return path
    check('FFprobe',ffprobe)
    def assets():
        from .media import neural_face_assets
        return list(neural_face_assets())
    check('Face detector assets',assets)
    def model(path,cfg=None):
        from .models import load_head
        from .inference import Analyzer
        path=Path(path)
        head=load_head(path)
        settings=cfg if cfg is not None else head['config']
        analyzer=Analyzer(settings,str(path))
        if analyzer.setup_problem:raise RuntimeError(analyzer.setup_problem)
        return {'file':str(path),'sha256':sha256(path),'trained_videos':head['training_videos']}
    cfg=config()
    check('Standard model and configuration',lambda:model(ROOT/'checkpoints/visual.json',cfg))
    if cfg.get('retry_model'):
        check('Retry model',lambda:model(ROOT/cfg['retry_model']))
    if check_encoder:
        def encoder():
            from .models import FaceEncoder
            FaceEncoder()
            return 'Verified pretrained image encoder loads successfully.'
        check('Image encoder',encoder)
    report['ready']=all(item['ok'] for item in report['checks'])
    report['scope']='Startup readiness only; this does not certify prediction accuracy.'
    return report
