# Persistent graph validation

Validated on `replica/room_0/00`, stride 20, with 100 sampled RGB frames and ground-truth depth/camera poses. The run used 4,090 existing SAM3 masks and computed actual DINOv3 and SigLIP image descriptors and world-coordinate boxes. It did not rerun SAM3; reuse of cached segmentation is intentional. This is Replica scene data, not a real robot deployment.

## Actual results

| Metric | Raw SAM tracks | Persistent graph |
|---|---:|---:|
| IDF1 | 0.577 | 0.610 |
| Identity precision | 0.505 | 0.582 |
| Identity recall | 0.672 | 0.642 |
| IDs, including background/unmatched detections | 101 | 224 |
| IDs per matched foreground GT object | 3.25 | 1.98 |
| GT objects per matched predicted ID | 1.84 | 1.13 |
| Revisit IDs retained | 43 | 45 |
| Revisit IDs switched | 14 | 12 |
| Revisits missed on one side | 27 | 27 |
| Revisit retention among detected pairs | 75.4% | 78.9% |
| OpenLex3D L0 query mAP | 0.51 | 0.49 |
| OpenLex3D L1 query mAP | 0.61 | 0.58 |

Text evaluation for the graph uses the compact memory's stored SigLIP means, as the query CLI does. Retrieval did not improve over raw tracks, despite better identity scores. The graph has 2,037 geometric edges, including background surfaces; its compressed node-array snapshot is 3,255,136 bytes (about 3.1 MiB), plus small node metadata. Match diagnostics and observation caches are separate and not needed to load compact memory.

Twelve retained revisit events changed their associated SAM-ID set. Examples backed by GT mask matching:

- GT table 7 retained persistent ID 12 from frame 160 to 300, despite SAM ID changing from 27 to 25.
- GT switch 16 retained persistent ID 20 from frame 180 to 300, despite SAM ID changing from 24 to 23.
- GT blinds 30 retained persistent ID 45 from frame 320 to 880, despite SAM ID changing from 43 to 91/92.

A repeated-match audit uses mask IoU >= 0.5 and at least three matching frames per GT/node pair. It found **11 GT objects split across nodes** and **2 nodes matching multiple GT objects**, compared with 21 splits and 17 mixed identities for raw SAM tracks. Nodes 71 and 102 warrant inspection; the diagnostic JSON lists their GT matches. These counts flag likely identity errors; they do not establish perfect segmentation or classify every error mechanism.

## Tuning and reproducibility

Nine configurations were measured on this same scene. The initial threshold 0.65 with IoU-only geometry produced 831 nodes and IDF1 0.473. The selected baseline uses weights DINO 0.45 / SigLIP 0.15 / geometry 0.40, threshold 0.45, equal IoU and exponential centre similarity, 1 m centre gate, four descriptors per modality, no hard label equality, and duplicate-mask IoU 0.60. These defaults are tuned on room_0, so the improvement is not held-out evidence.

Changing only the threshold to 0.60 via `run.py --graph-only` produced 449 nodes. Rebuilding with the baseline returned the exact original observation assignments. Modification times of masks, DINO features, SigLIP features and geometry were unchanged. Save/load restored configuration, descriptor banks, stable IDs, timestamps and edges; continued matching after reload gave the same ID as uninterrupted memory.

Sixteen CPU tests cover revisits, distinct objects, one-to-one assignment, prototype bounds, missing/invalid modalities, score normalization, enable switches, distance gating, optional centre similarity, duplicate consolidation, prefix replay, parameter changes, compact queries and save/load. The notebook was executed headlessly against the actual caches, and its frame/object selectors and rebuild/save controls were exercised programmatically. Browser interaction itself was not automated.

## Artifacts and commands

Actual artifacts are under `outputs/replica/room_0/00/stride20/sam3_gt/`:

- `graph/`: assignments, compact memory, spatial edges, score traces and node history.
- `eval/metrics.json`: identity and retrieval metrics.
- `eval/identity_validation.json`: repeated-match audit and concrete revisit examples.
- `eval/association_sweep.json`, `eval/association_sweep_more.json`, `eval/tuning/`: measured parameter variants.
- `eval/visualize_graph_executed.ipynb`: executed headless notebook.
- `eval/revisit_table_12.png`: real RGB/SAM comparison of the retained table ID.

```bash
uv run --no-sync python run.py --source replica/room_0/00 --stride 20 --pose-source gt
uv run --no-sync python run.py --source replica/room_0/00 --pose-source gt --graph-only
PYTHONPATH=src:. uv run --no-sync python -m graph.validation --source replica/room_0/00
PYTHONPATH=src uv run --no-sync python -m unittest discover -s tests -v
```

## Remaining limitations

- Tested/tuned on one Replica sequence; other scenes and real robot camera data remain unvalidated.
- Cached SAM masks used the scene's GT class-name vocabulary; a fixed-vocabulary run without that information remains untested.
- Static objects and a shared metric, z-up coordinate frame are assumed. Independent session reconstructions require alignment. No moved-object recovery, loop-closure correction or deletion policy is implemented.
- Greedy assignment is deterministic but not globally optimal. Aggressive duplicate-mask consolidation can merge overlapping physical objects; strict consolidation can retain duplicates.
- Missing modalities lower scores rather than redistribute weights. In particular, geometry alone cannot exceed the current 0.45 threshold when other enabled cues are absent and geometry weight is 0.40. This deliberately prevents low-evidence score inflation; disable absent modalities globally if appropriate.
- Conservative box expansion retains extents from partial views but may preserve or accumulate erroneous geometry. Boxes represent observed surfaces, not complete physical volumes.
- Near/above relationships are geometric approximations. Large background boxes create many edges. No contact, support, reachability or navigation claims are made.
- Model inference still has the original GPU/checkpoint requirements. Graph replay and notebook controls load no models, but diagnostic caches grow with observations and should be retained separately from compact robot memory.
- IDs survive incremental updates and reloads; a parameter rebuild starts fresh IDs. Use the generated assignments for that configuration when comparing objects.
- `graph.plot` still describes the legacy batch merger. The new notebook and saved match traces are the inspection tools for the incremental graph.
