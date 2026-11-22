"""Plumbing check: the demo loads a BROS joint checkpoint (random linker head, so the output is meaningless) and runs the words pipeline on a made-up form."""
import json
import os
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
import tempfile; tmp = Path(tempfile.mkdtemp()) / 'J1'

(tmp / 'seed_42').mkdir(parents=True)
shutil.copy(ROOT / 'results' / 'bros_joint' / 'J1' / 'metrics.json', tmp / 'metrics.json')
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ['DEMO_CKPT_DIR'] = str(tmp / 'seed_42')
import importlib; D = importlib.import_module("src.demo")
from models.joint_extractor import JointFormExtractor

cfg = json.load(open(tmp / 'metrics.json'))['config']
m = JointFormExtractor(cfg['backbone'], len(D.ID2LABEL), D.ID2LABEL, spatial_in_features=int(cfg['spatial_in_features']))
torch.save({'model_state_dict': m.state_dict()}, tmp / 'seed_42' / 'best_model.pt')
loaded = D.get_loaded_model()
assert loaded is not None and loaded.backbone == 'bros', getattr(loaded, 'backbone', None)
print('loaded backbone', loaded.backbone)
ocr = [([[100, 100], [300, 100], [300, 140], [100, 140]], 'Invoice Number:', 1.0), ([[320, 100], [480, 100], [480, 140], [320, 140]], 'INV-84920', 1.0),
       ([[100, 200], [250, 200], [250, 240], [100, 240]], 'Total Due:', 1.0), ([[270, 200], [420, 200], [420, 240], [270, 240]], '$4,250.00', 1.0)]
ents, links, recs = D.words_pipeline(loaded, ocr, 600, 400)
print('entities', len(ents), 'links', len(links), '(untrained head: any number is fine)')
print('[+] demo plumbing with a BROS joint checkpoint ok')
