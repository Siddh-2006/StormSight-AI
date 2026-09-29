# Training on HPC

**TODO (pending):** train the real model(s) on HPC once available, then integrate the resulting checkpoint(s) back into this repo. The local dev machine (8GB RAM, no GPU) is fine for building/testing the pipeline but not for real training — see `NEXT_STEPS.md` and `docs/evaluation_results.md` for why the current checkpoint is weak (undertrained, loses to the optical-flow baseline).

## What to train

The architecture (`backend/app/ml/model.py`) and training script (`backend/app/ml/train.py`) don't need to change — just run with a much bigger budget than was feasible locally:

- More epochs (local: 8 → try 50-100+)
- More synthetic episodes per epoch (local: 96 → try 500-2000+)
- Larger batch size (local: 4, memory-constrained → try 32-64 on a GPU)
- Optionally a bigger model via `--hidden-channels 32,64` or `64,128` (default is `16,32`)

If you also want to try **real data** instead of only the synthetic generator (see `docs/research_notes.md` §3 for why **SEVIR** — MIT Lincoln Lab's radar+satellite+GLM-lightning dataset — is the natural real-data upgrade), that's a separate task: write a new `Dataset` in `backend/app/ml/dataset.py` that yields the same `(input_seq, radar_target, lightning_target)` shapes from SEVIR tiles instead of the synthetic generator, and everything else (model, training loop, evaluation) stays the same.

## Workflow

1. **Sync the repo to the HPC** (exclude the local venv and build artifacts):
   ```bash
   rsync -avz --exclude='.venv' --exclude='node_modules' --exclude='*.pyc' \
       --exclude='cpp_module/build*' \
       ./ your-username@hpc-login-node:/path/to/thunderstorm_alert/
   ```
   Or just `git push` here and `git clone`/`git pull` on the HPC side (the repo is at `https://github.com/bhscodz/thunderstorm_alert`) — usually cleaner than rsync if the HPC has outbound git access.

2. **Edit `scripts/train_hpc.sbatch`** — fill in the `#SBATCH` placeholders (partition, account, GPU type, walltime) for your specific cluster, and the module-load lines in the environment-setup section. This repo has no way to know your cluster's scheduler config, so these are left as clearly marked edits.

3. **Submit the job**:
   ```bash
   sbatch scripts/train_hpc.sbatch
   ```
   (If your cluster uses PBS/Torque instead of SLURM, translate the `#SBATCH` directives to `#PBS` equivalents — same script body otherwise; ask if you want a PBS variant written out.)

4. **Retrieve the trained checkpoint** once the job finishes:
   ```bash
   scp your-username@hpc-login-node:/path/to/thunderstorm_alert/models/checkpoints/convlstm.pt \
       models/checkpoints/convlstm.pt
   ```

5. **Re-run evaluation locally** to confirm the bigger model actually closes the gap with the baseline:
   ```bash
   cd backend && ../.venv/Scripts/python ../scripts/evaluate.py
   ```
   Check `docs/evaluation_results.md` — the goal is ConvLSTM CSI/HSS beating the optical-flow baseline at most/all lead times (it currently doesn't, see the committed results).

6. **No other code changes needed** — `backend/app/ml/infer.py` reads the model's `hidden_channels` from the checkpoint automatically, so a bigger HPC-trained model drops in without touching the API, frontend, or anything else.
