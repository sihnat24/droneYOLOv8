# drone detection autoresearch

## context
YOLOv8n fine-tuned on VisDrone2019-DET aerial object detection dataset.
Device: Apple M-series Mac (MPS).
Dataset: 10% fraction for fast iteration.

## your role
You are an autonomous research agent. Your job is to improve mAP50.
You may ONLY modify train.py.
You may NOT modify dataset.yaml, the dataset, or the eval logic.

## metric
Higher mAP50 is better.
Baseline: 0.072

## experiment loop
1. Read program.md and train.py in full
2. Form one hypothesis — what single change might improve mAP50?
3. Edit train.py with that change
4. Run: python train.py > run.log 2>&1
5. Extract result: grep "mAP50" run.log | tail -1
6. If mAP50 improved: git add train.py && git commit -m "exp: <what you changed> mAP50=<new score>"
7. If not improved: git reset --hard
8. Record in results.tsv: experiment number, what changed, mAP50, keep/revert
9. Repeat from step 2

## constraints
- One change per experiment
- Never change more than ~5 lines at once
- Each run must complete without crashing
- Do not change fraction= or epochs= (keep budget fixed for fair comparison)
- If a run crashes twice, abandon and try something else

## parameters in scope (in rough priority order)
- flipud (currently 0.0 — wrong for aerial, try 0.5)
- degrees (currently 0.0 — try 10-15)
- imgsz (currently 640 — try 1280, watch memory)
- scale (currently 0.5)
- lr0, lrf
- mosaic (currently 1.0)
- hsv_h, hsv_s, hsv_v
- dropout, weight_decay
- batch size

## simplicity rule
All else being equal, simpler is better. Don't keep a change 
that adds complexity for a marginal gain.

## known context
- NMS time limit warnings are expected on VisDrone — not a crash
- MPS device may OOM at imgsz=1280 with large batch — reduce batch first
- val loss tracks closely with train loss — no overfitting yet
- precision is noisy, recall is stuck low — model is finding few objects