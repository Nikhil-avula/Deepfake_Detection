"""CPU workflow smoke test using synthetic crops and a tiny substitute network.
This does not evaluate deepfake accuracy. Export packaging stubs the prediction;
actual EfficientNet inference was checked separately with the provided sample.
Run from a development environment containing the project dependencies.
"""
import sys,json,tempfile,io,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from torch import nn
root=Path(__file__).resolve().parent
nb=json.loads((root/'DeepGuard_FF_C23_Kaggle.ipynb').read_text())
codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
scope={}
with tempfile.TemporaryDirectory(dir='/tmp',prefix='ff-notebook-smoke-') as td:
    first=codes[0].replace("assert torch.cuda.is_available(), 'Enable GPU in Kaggle notebook settings before training.'",'')
    first=first.replace("torch.cuda.get_device_name(0)","'CPU test'")
    first=first.replace("round(torch.cuda.get_device_properties(0).total_memory/2**30, 1)",'0')
    first=first.replace("'/kaggle/working/deepguard_ff'",repr(td))
    exec(first,scope)
    scope.update(DEVICE='cpu',WORKERS=0,HEAD_EPOCHS=1,FINETUNE_EPOCHS=1)
    exec(codes[1],scope)
    data=Path(td)/'input';data.mkdir()
    for method in ['original','deepfakes','face2face','faceshifter','faceswap','neuraltextures']:
        folder=data/method;folder.mkdir()
        for i in range(6) if method=='original' else range(0,6,2):
            (folder/(f'{i:03}.mp4' if method=='original' else f'{i:03}_{i+1:03}.mp4')).touch()
    scope['DATA_ROOT']=data
    exec(codes[2],scope)
    meta=Path(td)/'official';meta.mkdir()
    for filename,ids in [('train.json',['000','001']),('val.json',['002','003']),('test.json',['004','005'])]:
        (meta/filename).write_text(json.dumps([ids]))
    # Cross-partition fake is excluded rather than leaking source identities.
    (data/'deepfakes'/'000_002.mp4').touch()
    exec(codes[2],scope)
    scope['SPLITS_DIR']=meta
    exec(codes[3],scope)
    assert len(scope['manifest'])==21
    prepared=[]
    rng=np.random.default_rng(1)
    for row in scope['manifest'].to_dict('records'):
        folder=Path(td)/'faces'/row['uid'];folder.mkdir(parents=True)
        crops=[]
        for j in range(3):
            p=folder/f'{j}.jpg'
            Image.fromarray(rng.integers(0,255,(224,224,3),dtype=np.uint8)).save(p,quality=95)
            crops.append(str(p))
        prepared.append(dict(**row,crops=crops,faces=3,usable=True,error=''))
    scope['prepared']=pd.DataFrame(prepared)
    scope['usable']=scope['prepared'].copy()
    scope['prepared'].drop(columns=['crops']).to_csv(Path(td)/'coverage.csv',index=False)
    scope['CFG']=dict(scope['runtime'].PREPROCESS)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.features=nn.Sequential(nn.Conv2d(3,4,3,stride=16),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
            self.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(4,1))
        def forward(self,x):return self.classifier(self.features(x))
    scope['runtime'].build_model=lambda pretrained=False:Tiny()
    exec(codes[7],scope) # loaders
    exec(codes[8],scope) # training
    exec(codes[8],scope) # resume completed run
    assert len(scope['history'])==2
    exec(codes[9],scope) # validation threshold
    exec(codes[10],scope) # final test
    metrics=json.loads((Path(td)/'metrics.json').read_text())
    assert metrics['test_total']==7 and metrics['test_analyzed']==7
    scope['ASSET_DIR']=root.parents[1]/'assets'
    def stub_predict(video,bundle,device='cpu'):
        row=scope['usable'].loc[scope['usable'].path==str(video)].iloc[0]
        score=float(scope['val_prediction'].loc[scope['val_prediction'].uid==row.uid,'fake_score'].iloc[0])
        return dict(fake_score=score)
    scope['runtime'].predict_video=stub_predict
    exec(codes[11].replace('/kaggle/working/deepguard_ff_export',str(Path(td)/'export_bundle')),scope)
    compile((Path(td)/'export'/'app.py').read_text(),'export_app.py','exec')
    assert (Path(td)/'export'/'requirements-local.txt').exists()
    print('PASS: export packaging, generated local app syntax; full runtime tested separately')
    print('PASS: inventory, source-disjoint split, video batching, both training stages, checkpoint resume, validation threshold, final metrics')
