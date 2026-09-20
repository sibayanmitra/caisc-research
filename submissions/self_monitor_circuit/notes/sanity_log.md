- [23:45:15] ============================================================
- [23:45:15] Hard timeout: 1800s (30.0 min)
- [23:45:15] ## Phase 0 Setup Log

- [23:45:15] Phase 0 started at 2026-05-27 23:45:15
- [23:45:15] TransformerLens version: 3.2.1
- [23:45:16] Transformers version: 5.9.0
- [23:45:19] PyTorch version: 2.11.0+cu128
- [23:45:19] CUDA available: True
- [23:45:19] GPU count: 4
- [23:45:19]   GPU 0: NVIDIA RTX 6000 Ada Generation
- [23:45:19]   GPU 1: NVIDIA RTX 6000 Ada Generation
- [23:45:19]   GPU 2: NVIDIA RTX 6000 Ada Generation
- [23:45:19]   GPU 3: NVIDIA RTX 6000 Ada Generation
- [23:45:19] Set CUDA_VISIBLE_DEVICES=0 (single GPU mode per user instruction)
- [23:45:19] 
============================================================
- [23:45:19] STEP 1: Load Qwen2.5-7B-Instruct via TL (user-selected primary)
- [23:45:19] ============================================================
- [23:45:19] Note: DeepSeek-R1-Distill-Qwen-7B skipped per user choice to avoid extra download.
- [23:45:19] Qwen2.5-7B-Instruct is officially supported in TL 3.2.1.
- [23:45:22] 
Attempting TL load of: Qwen/Qwen2.5-7B-Instruct
- [23:50:45]   FAILED after 321.6s: RuntimeError: [enforce fail at alloc_cpu.cpp:127] err == 0. DefaultCPUAllocator: can't allocate memory: you tried to allocate 271581184 bytes. Error code 12 (Cannot allocate memory)
- [23:50:47] 
✗ Qwen2.5-7B-Instruct failed to load — escalating to user
- [23:53:15] ============================================================
- [23:53:15] Hard timeout: 1800s (30.0 min)
- [23:53:15] ## Phase 0 Setup Log

- [23:53:15] Phase 0 started at 2026-05-27 23:53:15
- [23:53:15] TransformerLens version: 3.2.1
- [23:53:16] Transformers version: 5.9.0
- [23:53:17] PyTorch version: 2.11.0+cu128
- [23:53:17] CUDA available: True
- [23:53:18] GPU count: 4
- [23:53:18]   GPU 0: NVIDIA RTX 6000 Ada Generation
- [23:53:18]   GPU 1: NVIDIA RTX 6000 Ada Generation
- [23:53:18]   GPU 2: NVIDIA RTX 6000 Ada Generation
- [23:53:18]   GPU 3: NVIDIA RTX 6000 Ada Generation
- [23:53:18] Set CUDA_VISIBLE_DEVICES=0 (single GPU mode per user instruction)
- [23:53:18] 
============================================================
- [23:53:18] STEP 1: Load Qwen2.5-7B-Instruct via TL (user-selected primary)
- [23:53:18] ============================================================
- [23:53:18] Note: DeepSeek-R1-Distill-Qwen-7B skipped per user choice to avoid extra download.
- [23:53:18] Qwen2.5-7B-Instruct is officially supported in TL 3.2.1.
- [23:53:20] 
Attempting TL load of: Qwen/Qwen2.5-7B-Instruct
- [23:53:25]   Using from_pretrained_no_processing (recommended for fp16 to skip LN-folding step)
- [23:54:19]   SUCCESS: loaded in 54.3s
- [23:54:19]   Layers: 28, d_model: 3584, n_heads: 28, d_head: 128
- [23:54:19] 
Verifying hooks on Qwen/Qwen2.5-7B-Instruct
- [23:54:19]   Tokens shape: torch.Size([1, 5])
- [23:54:19]   Logits shape: torch.Size([1, 5, 152064])
- [23:54:19]   Cache keys (sample): ['hook_embed', 'blocks.0.hook_resid_pre', 'blocks.0.ln1.hook_scale', 'blocks.0.ln1.hook_normalized', 'blocks.0.attn.hook_q']
- [23:54:19]   blocks.0.hook_resid_post shape: torch.Size([1, 5, 3584])
- [23:54:19]   blocks.0.hook_resid_post norm: 54.12
- [23:54:19]   blocks.0.attn.hook_z shape: torch.Size([1, 5, 28, 128])
- [23:54:19]   Top prediction after 'The capital of France is': ' Paris'
- [23:54:19]   ✓ All hooks fire, no NaN detected
- [23:54:19] 
✓ MODEL READY: Qwen2.5-7B-Instruct via TL
- [23:54:19] Saved model choice to /home/anonymous/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit/notes/model_choice.json: Qwen/Qwen2.5-7B-Instruct via tl
- [00:37:55] 
## Phase 1 Step 1: Generate CoT outputs

- [00:37:55] ============================================================
- [00:37:55] Phase 1 started at 2026-05-28 00:37:55
- [00:37:55] Generation budget: 3600s (60.0 min)
- [00:37:55] Loading model...
- [00:38:42]   Model loaded in 47.2s
- [00:38:42]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [00:38:42]   Layers: 28, d_model: 3584, n_heads: 28
- [00:38:42] Loading MATH dataset...
- [00:38:43]   lighteval/MATH failed, trying hendrycks/competition_math
- [00:47:32] 
## Phase 1 Step 1: Generate CoT outputs

- [00:47:32] ============================================================
- [00:47:32] Phase 1 started at 2026-05-28 00:47:32
- [00:47:32] Generation budget: 10800s (180.0 min)
- [00:47:32] Loading model...
- [00:48:18]   Model loaded in 46.2s
- [00:48:18]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [00:48:18]   Layers: 28, d_model: 3584, n_heads: 28
- [00:48:18] Loading MATH dataset...
- [00:48:18]   lighteval/MATH failed, trying hendrycks/competition_math
- [00:50:37] 
## Phase 1 Step 1: Generate CoT outputs

- [00:50:37] ============================================================
- [00:50:37] Phase 1 started at 2026-05-28 00:50:37
- [00:50:37] Generation budget: 10800s (180.0 min)
- [00:50:37] Loading model...
- [00:51:25]   Model loaded in 47.9s
- [00:51:25]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [00:51:25]   Layers: 28, d_model: 3584, n_heads: 28
- [00:51:25] Loading MATH dataset...
- [00:51:29]   Loaded 500 problems from HuggingFaceH4/MATH-500
- [00:51:29]   Hard problems (Level 3-5, hard subjects): 0
- [00:51:29]   Other problems: 500
- [00:51:29] 
Generation params: samples=2, max_new=800, T=0.7, top_p=0.95
- [01:06:05]   Generated 10 so far (0.01/sec, 15.5 min in, 164.5 min left)
- [01:20:46]   Generated 20 so far (0.01/sec, 30.1 min in, 149.9 min left)
- [01:34:53]   Generated 30 so far (0.01/sec, 44.3 min in, 135.7 min left)
- [01:51:16]   Generated 40 so far (0.01/sec, 60.6 min in, 119.4 min left)
- [02:05:51]   Generated 50 so far (0.01/sec, 75.2 min in, 104.8 min left)
- [02:21:14]   Generated 60 so far (0.01/sec, 90.6 min in, 89.4 min left)
- [02:35:02]   Generated 70 so far (0.01/sec, 104.4 min in, 75.6 min left)
- [02:49:11]   Generated 80 so far (0.01/sec, 118.6 min in, 61.4 min left)
- [03:03:14]   Generated 90 so far (0.01/sec, 132.6 min in, 47.4 min left)
- [03:18:19]   Generated 100 so far (0.01/sec, 147.7 min in, 32.3 min left)
- [03:18:19] Checkpoint: 50 problems, 100 generations, 32.3 min budget left
- [03:32:18]   Generated 110 so far (0.01/sec, 161.7 min in, 18.3 min left)
- [03:46:47]   Generated 120 so far (0.01/sec, 176.2 min in, 3.8 min left)
- [03:51:10] Time budget tight after 123 generations — stopping
- [03:51:10] Time budget exhausted after 62 problems, 123 generations
- [03:51:10] 
✓ Phase 1 step 1 complete: 123 generations from 62 problems in 180.5 min
- [03:51:10]   Average: 88.1 sec/generation
- [03:51:10]   Output: /home/anonymous/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit/data/generations/generations.jsonl
- [03:51:51] 
## Phase 1 Step 2: Filter generations & build pos/neg/control sets

- [03:51:51] ============================================================
- [03:51:51] Phase 1 step 2 started at 2026-05-28 03:51:51
- [03:51:51] Loaded 123 generations from /home/anonymous/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit/data/generations/generations.jsonl
- [03:51:51] 
Classification done:
- [03:51:51]   Positives (contain self-correction marker): 121
- [03:51:51]   Negatives (no self-correction): 2
- [03:51:51] 
Marker frequency in positives:
- [03:51:51]     'hmm': 112
- [03:51:51]     'wait': 8
- [03:51:51]     'actually': 1
- [03:51:51] 
Loading model for tokenization (needed for position alignment)...
- [03:52:37]   Loaded deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [03:52:37] 
Computing token positions for positives...
- [03:52:37]   Positives with valid token positions: 121
- [03:52:37] 
Matching negative positions by length distribution...
- [03:52:37]   Negatives with tokenization: 2
- [03:52:37]   Matched negatives: 2
- [03:52:37]   Control (post-correction) examples: 121
- [03:52:37] 
## Sanity Checks

- [03:52:37] SC1 (balance pos/neg = 60.50): FAIL
- [03:52:37] SC3 (marker coverage): FAIL — 3 unique markers, top freq 92.6%
- [03:52:37] SC4 (no leakage): PASS
- [03:52:37] SC6 (yield ≥60 positives): PASS
- [03:52:37] 
✓ Phase 1 step 2 complete. Stats saved to /home/anonymous/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit/data/stats.json
- [03:52:37]   Positives: 121
- [03:52:37]   Negatives: 2
- [03:52:37]   Controls: 121
- [03:52:38] 
## Phase 2: Activation extraction

- [03:52:38] ============================================================
- [03:52:38] Phase 2 started at 2026-05-28 03:52:38
- [03:52:38] Loading model...
- [03:53:24]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [03:53:24]   Layers: 28, d_model: 3584, n_heads: 28, d_head: 128
- [03:53:24] 
Extracting activations for positive (121 records)...
- [03:53:29]   positive: 20/121 done (3.7/s, est total: 0.5 min)
- [03:53:34]   positive: 40/121 done (3.8/s, est total: 0.5 min)
- [03:53:40]   positive: 60/121 done (3.8/s, est total: 0.5 min)
- [03:53:45]   positive: 80/121 done (3.9/s, est total: 0.5 min)
- [03:53:50]   positive: 100/121 done (3.9/s, est total: 0.5 min)
- [03:53:55]   positive: 120/121 done (3.9/s, est total: 0.5 min)
- [03:53:55]   positive done: 121 kept, 0 skipped, 0 with NaN, in 0.5 min
- [03:53:55]   Saved positive activations to /home/anonymous/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit/activations/positive.pt
- [03:53:55] 
Extracting activations for negative (2 records)...
- [03:53:56]   negative done: 2 kept, 0 skipped, 0 with NaN, in 0.0 min
- [03:53:56]   Saved negative activations to /home/anonymous/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit/activations/negative.pt
- [03:53:56] 
Extracting activations for control (121 records)...
- [03:54:01]   control: 20/121 done (3.8/s, est total: 0.5 min)
- [03:54:13]   control: 40/121 done (2.4/s, est total: 0.9 min)
- [03:54:23]   control: 60/121 done (2.2/s, est total: 0.9 min)
- [03:54:31]   control: 80/121 done (2.3/s, est total: 0.9 min)
- [03:54:36]   control: 100/121 done (2.5/s, est total: 0.8 min)
- [03:54:42]   control: 120/121 done (2.6/s, est total: 0.8 min)
- [03:54:42]   control done: 121 kept, 0 skipped, 0 with NaN, in 0.8 min
- [03:54:42]   Saved control activations to /home/anonymous/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit/activations/control.pt
- [03:54:42] 
✓ Phase 2 complete. Summary: {
  "positive": {
    "total": 121,
    "skipped": 0,
    "nan": 0
  },
  "negative": {
    "total": 2,
    "skipped": 0,
    "nan": 0
  },
  "control": {
    "total": 121,
    "skipped": 0,
    "nan": 0
  }
}
- [03:54:43] 
## Phase 3: Probe training

- [03:54:43] ============================================================
- [03:54:43] Phase 3 started at 2026-05-28 03:54:43
- [03:54:46] Loading activations...
- [03:54:47]   Positives: torch.Size([121, 28, 6, 3584])
- [03:54:47]   Negatives: torch.Size([2, 28, 6, 3584])
- [03:54:47]   Offsets: [-3, -2, -1, 0, 1, -5]
- [03:54:47]   N pos: 121, N neg: 2
- [03:54:47] ⚠️ Too few examples for stable probing
- [09:34:50] 
## Phase 3: Probe training

- [09:34:50] ============================================================
- [09:34:50] Phase 3 started at 2026-05-28 09:34:50
- [09:34:52] Loading activations...
- [09:34:59]   Positives: torch.Size([121, 28, 6, 3584])
- [09:34:59]   Negatives (raw): torch.Size([2, 28, 6, 3584])
- [09:34:59]   Offsets: [-3, -2, -1, 0, 1, -5]
- [09:34:59]   N pos: 121, N neg (raw): 2
- [09:34:59] ⚠️ Only 2 whole-generation negatives (SC rate too high).
- [09:34:59]   Falling back to post-correction CONTROL set as negatives.
- [09:34:59]   Using 121 post-correction positions as negatives.
- [09:34:59]   NOTE: probe now contrasts 'about-to-correct' vs 'post-correction' positions.
- [09:34:59] 
Training probes per (layer, offset)...
- [09:35:01]   Layer 0: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 0.0 min elapsed
- [09:35:09]   Layer 1: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 0.2 min elapsed
- [09:35:16]   Layer 2: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 0.3 min elapsed
- [09:35:23]   Layer 3: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 0.4 min elapsed
- [09:35:31]   Layer 4: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 0.5 min elapsed
- [09:35:36]   Layer 5: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 0.6 min elapsed
- [09:35:38]   Layer 6: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 0.7 min elapsed
- [09:35:40]   Layer 7: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 0.7 min elapsed
- [09:35:43]   Layer 8: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 0.7 min elapsed
- [09:35:45]   Layer 9: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 0.8 min elapsed
- [09:35:49]   Layer 10: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 0.8 min elapsed
- [09:35:51]   Layer 11: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 0.9 min elapsed
- [09:35:53]   Layer 12: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 0.9 min elapsed
- [09:35:55]   Layer 13: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 0.9 min elapsed
- [09:35:56]   Layer 14: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 1.0 min elapsed
- [09:35:58]   Layer 15: best offset AUROC = 1.000 (offset_idx=0, real offset=-3) — 1.0 min elapsed
- [09:36:00]   Layer 16: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.0 min elapsed
- [09:36:02]   Layer 17: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.0 min elapsed
- [09:36:04]   Layer 18: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.1 min elapsed
- [09:36:06]   Layer 19: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.1 min elapsed
- [09:36:08]   Layer 20: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.1 min elapsed
- [09:36:10]   Layer 21: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.2 min elapsed
- [09:36:11]   Layer 22: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.2 min elapsed
- [09:36:14]   Layer 23: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.2 min elapsed
- [09:36:15]   Layer 24: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.3 min elapsed
- [09:36:19]   Layer 25: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.3 min elapsed
- [09:36:21]   Layer 26: best offset AUROC = 1.000 (offset_idx=1, real offset=-2) — 1.4 min elapsed
- [09:36:23]   Layer 27: best offset AUROC = 1.000 (offset_idx=2, real offset=-1) — 1.4 min elapsed
- [09:36:23] 
PEAK: layer=0, offset=-2, AUROC=1.000
- [09:36:23] 
SC9: random baseline at peak cell...
- [09:36:23]   Random-labels AUROC at peak: 0.497 (should be ~0.5)
- [09:36:23] 
✓ Phase 3 complete. Probes saved to /home/anonymous/CAISc/autovoila/all-spikes/paper1-self-monitor-circuit/probes
- [09:36:23]   Peak AUROC: 1.000 at (L=0, offset=-2)
- [09:36:27] 
## Phase 5: Per-head attribution

- [09:36:27] ============================================================
- [09:36:27] Phase 5 started at 2026-05-28 09:36:27
- [09:36:32] Peak (from Phase 3): L=0, offset_idx=1, offset_value=-2, AUROC=1.000
- [09:36:32] Loading head_z activations...
- [09:36:32] ⚠️ Too few whole-gen negatives — using post-correction control set.
- [09:36:32]   Positives head_z: torch.Size([121, 28, 6, 28, 128])
- [09:36:32]   Negatives head_z: torch.Size([121, 28, 6, 28, 128])
- [09:36:32] Layers to analyze: [0, 1]
- [09:36:32] 
Layer 0:
- [09:36:32]   median=0.974, max=1.000
- [09:36:32]   top-3 heads at L=0: [(1, '1.000'), (5, '1.000'), (11, '1.000')]
- [09:36:32] 
Layer 1:
- [09:36:32]   median=0.964, max=1.000
- [09:36:32]   top-3 heads at L=1: [(0, '1.000'), (10, '1.000'), (25, '1.000')]
- [09:36:32] 
SC13 (head AUROC distribution skewness at peak layer): -1.00
- [09:36:32]   PASS' if right-skewed (>0.5): WARN
- [09:36:32]   Median: 0.974, Top-10% mean: 1.000
- [09:36:32] 
Top-k joint probe analysis at peak layer...
- [09:36:33]   Top-3: heads [27, 25, 11], joint AUROC = 1.000
- [09:36:33]   Top-5: heads [27, 25, 11, 12, 5], joint AUROC = 1.000
- [09:36:33]   Top-10: heads [27, 25, 11, 12, 5, 1, 24, 26, 23, 3], joint AUROC = 1.000
- [09:36:33] 
SC14 (top-5 joint AUROC - best single head): +0.000
- [09:36:33]   PASS if gain >= 0.02: WARN
- [09:36:34] 
✓ Phase 5 complete. Top heads at peak layer 0: [27, 25, 11, 12, 5]
- [09:36:36] 
## Phase 6: Causal ablation

- [09:36:36] ============================================================
- [09:36:36] Phase 6 started at 2026-05-28 09:36:36
- [09:36:37] Peak layer: 0
- [09:36:37] Top-10 heads at peak: [27, 25, 11, 12, 5, 1, 24, 26, 23, 3]
- [09:36:37] Loading held-out GSM8K problems...
- [09:36:45] Held-out problem count: 50
- [09:36:45] Loading model...
- [09:37:31]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [09:37:31] 
  Generating baseline (50 problems, ablating 0 heads)...
- [09:50:54]     baseline: 10/50, sc=10, rate=100.00%, 13.4 min
- [10:04:10]     baseline: 20/50, sc=17, rate=85.00%, 26.7 min
- [10:11:20] 
## Phase 6: Causal ablation

- [10:11:20] ============================================================
- [10:11:20] Phase 6 started at 2026-05-28 10:11:20
- [10:11:20] Peak layer: 0
- [10:11:20] Top-10 heads at peak: [27, 25, 11, 12, 5, 1, 24, 26, 23, 3]
- [10:11:20] Loading held-out GSM8K problems...
- [10:11:25] Held-out problem count: 20
- [10:11:25] Loading model...
- [10:12:08]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [10:12:08] 
  Generating baseline (20 problems, ablating 0 heads)...
- [10:26:26]     baseline: 10/20, sc=9, rate=90.00%, 14.3 min
- [10:42:06]     baseline: 20/20, sc=19, rate=95.00%, 30.0 min
- [10:42:06]   baseline DONE: 19/20 (95.00%) self-corrections in 30.0 min
- [10:42:06] 
Baseline self-correction rate: 95.00%
- [10:42:06] 
  Generating top3_ablation (20 problems, ablating 3 heads)...
- [10:55:12]     top3_ablation: 10/20, sc=7, rate=70.00%, 13.1 min
- [11:09:51]     top3_ablation: 20/20, sc=15, rate=75.00%, 27.8 min
- [11:09:51]   top3_ablation DONE: 15/20 (75.00%) self-corrections in 27.8 min
- [11:09:51] 
  Generating top5_ablation (20 problems, ablating 5 heads)...
- [11:24:30]     top5_ablation: 10/20, sc=9, rate=90.00%, 14.7 min
- [11:47:40]     top5_ablation: 20/20, sc=18, rate=90.00%, 37.8 min
- [11:47:40]   top5_ablation DONE: 18/20 (90.00%) self-corrections in 37.8 min
- [11:47:40] 
  Generating top10_ablation (20 problems, ablating 10 heads)...
- [13:21:23]     top10_ablation: 10/20, sc=4, rate=40.00%, 93.7 min
- [14:29:24]     top10_ablation: 20/20, sc=8, rate=40.00%, 161.7 min
- [14:29:24]   top10_ablation DONE: 8/20 (40.00%) self-corrections in 161.7 min
- [14:29:24] 
  Generating random3_seed0 (20 problems, ablating 3 heads)...
- [15:04:32]     random3_seed0: 10/20, sc=10, rate=100.00%, 35.1 min
- [15:40:31]     random3_seed0: 20/20, sc=20, rate=100.00%, 71.1 min
- [15:40:31]   random3_seed0 DONE: 20/20 (100.00%) self-corrections in 71.1 min
- [15:40:31] 
  Generating random5_seed0 (20 problems, ablating 5 heads)...
- [15:54:36] 
## Phase 6: Causal ablation

- [15:54:36] ============================================================
- [15:54:36] Phase 6 started at 2026-05-28 15:54:36
- [15:54:37] Peak layer: 0
- [15:54:37] Top-10 heads at peak: [27, 25, 11, 12, 5, 1, 24, 26, 23, 3]
- [15:54:37] Loading held-out GSM8K problems...
- [15:54:42] Held-out problem count: 50
- [15:54:42] Loading model...
- [16:14:51] 
## Phase 6: Causal ablation

- [16:14:51] ============================================================
- [16:14:51] Phase 6 started at 2026-05-28 16:14:51
- [16:14:52] Peak layer: 0
- [16:14:52] Top-10 heads at peak: [27, 25, 11, 12, 5, 1, 24, 26, 23, 3]
- [16:14:52] Loading held-out GSM8K problems...
- [16:14:56] Held-out problem count: 50
- [16:14:56] Loading model...
- [16:15:41]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [16:15:41] 
  Generating baseline (50 problems, ablating 0 heads)...
- [16:18:47] 
## Phase 6: Causal ablation

- [16:18:47] ============================================================
- [16:18:47] Phase 6 started at 2026-05-28 16:18:47
- [16:18:47] Peak layer: 0
- [16:18:47] Top-10 heads at peak: [27, 25, 11, 12, 5, 1, 24, 26, 23, 3]
- [16:18:47] Loading held-out GSM8K problems...
- [16:18:52] Held-out problem count: 50
- [16:18:52] Loading model...
- [16:19:39]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [16:19:39] 
  Generating baseline (50 problems, ablating 0 heads)...
- [16:51:56]     baseline: 10/50, sc=9, rate=90.00%, 32.3 min
- [17:25:42]     baseline: 20/50, sc=18, rate=90.00%, 66.0 min
- [17:54:27]     baseline: 30/50, sc=27, rate=90.00%, 94.8 min
- [18:24:54]     baseline: 40/50, sc=35, rate=87.50%, 125.2 min
- [18:57:48]     baseline: 50/50, sc=44, rate=88.00%, 158.1 min
- [18:57:48]   baseline DONE: 44/50 (88.00%) self-corrections in 158.1 min
- [18:57:48] 
Baseline self-correction rate: 88.00%
- [18:57:48] 
  Generating top3_ablation (50 problems, ablating 3 heads)...
- [19:33:49]     top3_ablation: 10/50, sc=10, rate=100.00%, 36.0 min
- [20:09:23]     top3_ablation: 20/50, sc=19, rate=95.00%, 71.6 min
- [20:44:53]     top3_ablation: 30/50, sc=29, rate=96.67%, 107.1 min
- [21:16:57]     top3_ablation: 40/50, sc=36, rate=90.00%, 139.1 min
- [21:50:21]     top3_ablation: 50/50, sc=42, rate=84.00%, 172.6 min
- [21:50:21]   top3_ablation DONE: 42/50 (84.00%) self-corrections in 172.6 min
- [21:50:21] 
  Generating top5_ablation (50 problems, ablating 5 heads)...
- [22:23:31]     top5_ablation: 10/50, sc=10, rate=100.00%, 33.2 min
- [22:56:41]     top5_ablation: 20/50, sc=17, rate=85.00%, 66.3 min
- [23:29:27]     top5_ablation: 30/50, sc=26, rate=86.67%, 99.1 min
- [23:59:45]     top5_ablation: 40/50, sc=33, rate=82.50%, 129.4 min
- [00:35:47]     top5_ablation: 50/50, sc=42, rate=84.00%, 165.4 min
- [00:35:47]   top5_ablation DONE: 42/50 (84.00%) self-corrections in 165.4 min
- [00:35:47] 
  Generating random3_seed0 (50 problems, ablating 3 heads)...
- [01:07:33]     random3_seed0: 10/50, sc=10, rate=100.00%, 31.8 min
- [01:44:01]     random3_seed0: 20/50, sc=20, rate=100.00%, 68.2 min
- [02:19:13]     random3_seed0: 30/50, sc=30, rate=100.00%, 103.4 min
- [02:54:06]     random3_seed0: 40/50, sc=40, rate=100.00%, 138.3 min
- [03:26:54]     random3_seed0: 50/50, sc=50, rate=100.00%, 171.1 min
- [03:26:54]   random3_seed0 DONE: 50/50 (100.00%) self-corrections in 171.1 min
- [03:26:54] 
  Generating random5_seed0 (50 problems, ablating 5 heads)...
- [04:02:03]     random5_seed0: 10/50, sc=10, rate=100.00%, 35.1 min
- [04:35:33]     random5_seed0: 20/50, sc=20, rate=100.00%, 68.6 min
- [05:07:39]     random5_seed0: 30/50, sc=30, rate=100.00%, 100.7 min
- [05:43:11]     random5_seed0: 40/50, sc=40, rate=100.00%, 136.3 min
- [06:18:33]     random5_seed0: 50/50, sc=50, rate=100.00%, 171.7 min
- [06:18:33]   random5_seed0 DONE: 50/50 (100.00%) self-corrections in 171.7 min
- [06:18:33] 
✓ Phase 6 complete. Summary:
- [06:18:33]   top3: 84.00% | random3: 100.00%±0.00% | baseline: 88.00%
- [06:18:33]   top5: 84.00% | random5: 100.00%±0.00% | baseline: 88.00%
- [06:19:31] 
## Phase 6.5: Steering experiment

- [06:19:31] ============================================================
- [06:19:31] Phase 6.5 started at 2026-05-29 06:19:31
- [06:19:31] Peak layer: 0, offset_idx: 1, peak AUROC: 1.000
- [06:19:31] Probe direction norm: 0.6028 → normalized
- [06:19:31] Loading model...
- [06:20:16]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [10:32:11] 
## Phase 6.5: Steering experiment

- [10:32:11] ============================================================
- [10:32:11] Phase 6.5 started at 2026-05-29 10:32:11
- [10:32:11] Peak layer: 0, offset_idx: 1, peak AUROC: 1.000
- [10:32:11] Probe direction norm: 0.6028 → normalized
- [10:32:11] Loading model...
- [10:32:56]   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- [10:32:57] α_unit (mean resid norm at peak layer): 25.92
- [10:33:00] Held-out steering problems: 20
- [10:54:40]   α=-2.0×unit: SC=0/20 (0.00%)
- [11:16:20]   α=-1.0×unit: SC=20/20 (100.00%)
- [11:38:01]   α=+0.0×unit: SC=20/20 (100.00%)
- [11:59:21]   α=+1.0×unit: SC=18/20 (90.00%)
- [12:21:00]   α=+2.0×unit: SC=0/20 (0.00%)
- [12:21:00] 
SC21 (α=0 SC rate): 100.00%
- [12:21:00] SC22: mean SC at α>0 = 45.00%, mean at α<0 = 50.00%
- [12:42:28]   random_dir_0: SC=13/20 (65.00%)
- [13:03:49]   random_dir_1: SC=15/20 (75.00%)
- [13:25:29]   random_dir_2: SC=20/20 (100.00%)
- [13:25:29] 
SC23: random dir SC rate at α=+1: 80.00%±14.72%
- [13:25:29] SC23: probe dir SC rate at α=+1: 90.00%
- [13:25:29] SC23 WARN: probe effect > random
- [13:25:29] 
✓ Phase 6.5 complete. Steering curve saved.
- [13:25:29]   α=-2: 0.00% → α=0: 100.00% → α=+2: 0.00%
