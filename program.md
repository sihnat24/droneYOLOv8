# drone detection autoresearch

## context
YOLOv8n fine-tuned on VisDrone2019-DET aerial object detection dataset.
Device: Apple M-series Mac (MPS).
Dataset: 10% fraction for fast iteration.
Task: ISR (intelligence, surveillance, reconnaissance) — detecting vehicles
and people from drone footage. Missing a real object is far worse than a 
false alarm. A human analyst reviews false positives.

## your role
You are an autonomous research agent. Your job is to improve the ISR score.
You may ONLY modify train.py.
You may NOT modify dataset.yaml, score.py, or the dataset itself.
Do not modify max_det or conf — these are locked to prevent recall gaming

## metric
ISR score = 0.75 * recall + 0.25 * precision
Higher is better. Baseline: 0.0983

Secondary reference (do not optimize directly):
- mAP50 baseline: 0.072
- recall baseline: 0.075
- precision baseline: 0.170

## run commands
python train.py > run.log 2>&1
python score.py >> run.log 2>&1

## extract result
grep "score" run.log | tail -1

## experiment loop
1. Read program.md and train.py in full
2. Form one hypothesis — what single change might improve the ISR score?
3. Write your hypothesis and expected effect as a comment in results.tsv before running
4. Edit train.py with that change
5. Run the run commands above
6. Extract the score
7. If score improved: git add train.py && git commit -m "exp: <what changed> score=<new> recall=<recall>"
8. If not improved: git reset --hard
9. Record in results.tsv: experiment, hypothesis, score, recall, precision, keep/revert
10. Repeat from step 2

## results.tsv format
experiment	hypothesis	score	recall	precision	keep
baseline	none	0.0983	0.075	0.170	yes

## constraints
- One change per experiment
- Never change more than ~5 lines at once
- Each run must complete without crashing
- Do not change fraction= or epochs= (keep budget fixed for fair comparison)
- If a run crashes twice, abandon and try something else
- Do not game the score — improving recall by predicting boxes everywhere is not valid

## parameters in scope (priority order)
- flipud (currently 0.0 — wrong for aerial, try 0.5)
- degrees (currently 0.0 — try 10-15)
- imgsz (currently 640 — try 1280, reduce batch first if OOM)
- scale (currently 0.5)
- lr0, lrf
- mosaic (currently 1.0)
- hsv_h, hsv_s, hsv_v
- dropout, weight_decay
- batch size

## known context
- NMS time limit warnings are expected on VisDrone — not a crash, ignore
- MPS device may OOM at imgsz=1280 — try batch=8 before trying batch=4
- val loss tracks train loss closely — no overfitting, model is undertrained
- recall is stuck at 0.075 — the model is missing most real objects
- precision is noisy across epochs — don't over-index on single epoch precision values


## time budget
Fixed at epochs=10, fraction=0.1. Do not change these.
Note: imgsz=1280 will run slower — this is acceptable, 
the epoch count is what matters for fair comparison, not wall clock time.

## stopping condition
Stop after 25 experiments or when score exceeds 0.20, whichever comes first.
Report a summary of what worked and what didn't.

## simplicity rule
All else being equal, simpler is better. A 0.001 gain that adds 
significant complexity should be reverted.


## crashes 
 MPS tensor shape mismatches can occur with aggressive augmentation. if a run crashes with a tensor error, try the same change with a 
  smaller augmentation value first (e.g. flipud=0.3 before flipud=0.5)

