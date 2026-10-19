from deepguard.diagnostics import diagnose


def test_missing_program_is_named_and_blocks_readiness(monkeypatch):
    monkeypatch.setattr('deepguard.diagnostics.shutil.which',lambda name:None)
    report=diagnose()
    assert not report['ready']
    failure=next(c for c in report['checks'] if c['name']=='FFprobe')
    assert not failure['ok'] and 'Install FFmpeg' in failure['detail']
    assert report['python'] and report['project_folder']


def test_unavailable_encoder_is_reported_without_crashing(monkeypatch):
    def fail():raise RuntimeError('encoder download unavailable')
    monkeypatch.setattr('deepguard.models.FaceEncoder',fail)
    report=diagnose(check_encoder=True)
    failure=next(c for c in report['checks'] if c['name']=='Image encoder')
    assert not report['ready'] and not failure['ok']
    assert 'encoder download unavailable' in failure['detail']
