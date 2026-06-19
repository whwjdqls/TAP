# Tracker-Aware Kinematic Motion Planning for Humanoid Whole-Body Control
## 0. WARNING
Never you login node to do heavy, memory exaustive computation.
If you need memory heavy cpu work, srun on a CPU and use it. After you are finished always scancel the srun. 
Also, if you need to debug with GPU, dont sbatch jobs until you are sure that it will work on a GPU. Always srun to a GPU and debug, and then scancel the SRUN and SBATCH it for the real run. If you think you are going to debug on SRUN for several tasks, you can leave the SRUN open. But not too long. MAX 2 hours. 
## Repository Setup (vendored dependencies)

This repo (`TAP/`) holds **only** the research scaffolding — launch wrappers in
`scripts/`, the `tmr_g1/` package, and `docs/`. The four upstream repos it
builds on are **not committed** (they're large, LFS-backed, and have their own
git history — see `.gitignore`). Clone them as **siblings inside `TAP/`** with
the exact directory names below; the scripts and `CLAUDE.md` paths assume them.

Conda envs live at `/nfsdata/home/jungbin.cho/miniconda3/envs/`. See `CLAUDE.md`
for full env package lists and `docs/notes.md` for gotchas referenced below.

```bash
cd /nfsdata/home/jungbin.cho/TAP   # clone all four as siblings here
```

### 1) GR00T-WholeBodyControl — SONIC whole-body tracker

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl
cd GR00T-WholeBodyControl
git lfs install && git lfs pull            # LFS assets (git-lfs lives in `base` env)
# env: env_isaaclab (Python 3.11, torch 2.7.0+cu128, Isaac Lab 2.3.0, Isaac Sim 5.1)
source /nfsdata/home/jungbin.cho/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
pip install -e ./gear_sonic                # editable; pulls the [training] extras
pip install open3d vector_quantize_pytorch # runtime deps missing from pyproject
python download_from_hf.py --sample        # released checkpoint + 2 sample motions
                                           #  (--training for the full Bones-SEED lib + SMPL)
python check_environment.py --training     # pre-flight check (needs a GPU node)
```

SONIC runs need a GPU — submit via Slurm, never the login node. Entry points:
`gear_sonic/{train_agent_trl,eval_agent_trl}.py`. See `CLAUDE.md` → "SONIC".

### 2) GMT — general motion tracker (inference only)

```bash
git clone https://github.com/zixuan417/humanoid-general-motion-tracking GMT
# env: gmt (Python 3.8, torch 2.4.1+cu121, MuJoCo 3.2.3, numpy 1.23.0,
#            mujoco-python-viewer 0.1.4)
conda activate gmt
```

Upstream ships the pretrained policy (`assets/pretrained_checkpoints/pretrained.pt`)
and 8 example motions — **no training/retargeter code released yet**, so this is
inference-only. **CPU-only on this cluster** (the pinned torch 2.4.1 has no
Blackwell sm_120 kernels). For headless runs use `scripts/gmt_smoke.py` (it stubs
the GLFW viewer that `sim2sim.py` otherwise requires). See `docs/notes.md` → GMT.

### 3) kimodo — kinematic planner + motion-rep library

```bash
git clone https://github.com/nv-tlabs/kimodo
conda activate kimodo_soma                 # shared env for kimodo + SOMA-X + TMR-G1
pip install -e ./kimodo
```

> **Namespace-shadowing trap** (`docs/notes.md` §6): `kimodo/` has both
> `kimodo/assets.py` and a `kimodo/assets/` dir, so `import kimodo` breaks when
> the cwd contains a `kimodo/` directory. **Always launch Python from a cwd
> without one** (e.g. `cd /tmp`). The TMR-G1 sbatch scripts already `cd /tmp`.

### 4) SOMA-X — SOMA parameter generation / retargeting

```bash
git clone https://github.com/NVlabs/SOMA-X
conda activate kimodo_soma
pip install -e .                           # see SOMA-X/requirements.txt
```

Used to turn Kimodo-generated SOMA parameters into motions for SONIC tracking
and for the side-by-side render comparisons in `scripts/`.

---

## 1. Core Research Idea

Recent humanoid motion systems increasingly follow a two-stage paradigm:

> **Kinematic Motion Planner → Whole-Body Tracker**

In this pipeline, a kinematic motion planner, often based on diffusion or MDM-like generative models, produces a reference human motion. A whole-body tracker, often trained with reinforcement learning, then tracks this reference on a humanoid robot.

This paradigm is attractive because it separates high-level motion generation from low-level physical control. The planner can generate diverse, semantic, and long-horizon motions, while the tracker handles dynamics, balance, contacts, and robot-specific constraints.

However, these two modules are usually trained separately. In many existing pipelines, the planner is trained only to generate realistic human motion, while the tracker is later trained or fine-tuned to imitate whatever reference motion the planner produces. This raises an important question:

> **When a planner–tracker pipeline fails, should we adapt the motion planner, the tracker, or both?**

This project investigates whether kinematic motion planners should be explicitly optimized for downstream trackability under a whole-body humanoid tracker.

---

## 2. Main Thesis

> **Kinematic humanoid motion planners should not be evaluated only by motion realism, diversity, or text alignment. They should also be optimized for downstream trackability under a real whole-body controller.**

Current systems often place most of the burden on the tracker. The tracker is expected to absorb all mismatch between human motion data and robot-executable motion. However, many generated motions may be realistic in a kinematic sense but difficult or impossible for a humanoid robot to execute due to balance, contact, torque, joint-limit, or morphology constraints.

Instead of only making the tracker more robust, this work proposes to make the planner **tracker-aware**. The planner should learn to generate reference motions that preserve semantic intent while being easier, safer, and more stable for a whole-body tracker to execute.

---

## 3. Research Question

The central research question is:

> **Who should adapt in a kinematic-planner-to-whole-body-tracker pipeline: the planner, the tracker, or both?**

More specifically:

1. Is tracker fine-tuning sufficient for handling untrackable reference motions?
2. Can planner fine-tuning improve downstream robot execution while preserving motion quality and semantic intent?
3. When is planner adaptation better than tracker adaptation?
4. When is joint planner–tracker adaptation necessary?
5. Can we define a general trackability objective for kinematic motion generation?

---

## 4. Proposed Paper Direction

A possible paper direction is:

> **Tracker-Aware Kinematic Motion Planning for Humanoid Whole-Body Control**

The paper would introduce a framework for adapting kinematic motion planners using rollout-derived trackability signals from a whole-body tracker.

Instead of treating the planner and tracker as independent modules, the system explicitly aligns the planner's output distribution with the tracker's executable motion manifold.

---

## 5. Proposed Method

### 5.1 System Overview

The system consists of three components:

1. **Kinematic Motion Planner (Kimodo, MDM, Closd's DiP)**

   * MDM-like diffusion model, Kimodo-like controllable motion generator, or another motion prior.
   * Inputs may include text, root trajectory, keyframes, end-effector constraints, or scene constraints.
   * Outputs reference motion in a kinematic representation.

2. **Whole-Body Tracker (SONIC, GMT-General motion Tracker)**

   * RL-based humanoid tracking policy.
   * Tracks the reference motion on a humanoid robot in simulation.
   * Produces physically executed motion.

3. **Trackability Model or Trackability Objective**

   * Evaluates whether the generated reference motion is easy or difficult for the tracker to execute.
   * Provides a learning signal to adapt the planner.

The key loop is:

```text
Text / Constraint / Scene Input
        ↓
Kinematic Motion Planner
        ↓
Reference Motion
        ↓
Whole-Body Tracker Rollout
        ↓
Execution Metrics / Trackability Signal
        ↓
Planner Fine-Tuning
```

---

### 5.2 Trackability Signal

For each generated reference motion, the frozen tracker attempts to execute it in simulation. From the rollout, we compute a trackability signal based on metrics such as:

* tracking error,
* fall rate,
* foot slip,
* contact mismatch,
* ground penetration,
* torque usage,
* joint-limit violation,
* center-of-mass stability,
* task success,
* upper-body motion preservation,
* residual correction magnitude.

This signal measures whether a reference motion is merely kinematically plausible or actually executable by the robot.

---

### 5.3 Trackability Critic

One possible implementation is to train a trackability critic:

```text
C_phi(reference_motion, initial_state, robot_context) → trackability_score
```

The critic predicts how well a given reference motion can be tracked by the current whole-body controller.

The critic can be trained using rollout data from the tracker. Once trained, it provides a differentiable or sample-efficient reward signal for planner fine-tuning.

---

### 5.4 Planner Fine-Tuning Objective

The planner should not simply collapse into safe but boring motions. Therefore, the fine-tuning objective must preserve the original motion prior while improving trackability.

A possible objective is:

```text
L = L_motion + λ_text L_text + λ_constraint L_constraint + λ_track L_trackability + λ_reg L_regularization
```

Where:

* `L_motion` preserves motion realism.
* `L_text` preserves semantic alignment with the input text.
* `L_constraint` preserves control constraints such as root trajectory or end-effector targets.
* `L_trackability` encourages motions that the tracker can execute.
* `L_regularization` prevents the planner from drifting too far from the pretrained motion prior.

Possible regularization strategies include:

* LoRA-based fine-tuning,
* KL regularization to the original planner,
* reconstruction loss on real motion data,
* diversity preservation loss,
* text-motion alignment loss,
* contact consistency loss,
* smoothness and acceleration penalties.

---

## 6. Experimental Design

### 6.1 Main Comparison

The most important experiment is a controlled comparison of where adaptation happens.

| Setting | Planner                      | Tracker    | Purpose                                       |
| ------- | ---------------------------- | ---------- | --------------------------------------------- |
| A       | Frozen                       | Frozen     | Baseline pipeline                             |
| B       | Frozen                       | Fine-tuned | Conventional tracker adaptation               |
| C       | Fine-tuned                   | Frozen     | Proposed planner-side adaptation              |
| D       | Fine-tuned                   | Fine-tuned | Joint adaptation / upper bound                |
| E       | Heuristic filtering          | Frozen     | Compare against simple feasibility filtering  |
| F       | Contact/smoothness loss only | Frozen     | Test whether trackability critic is necessary |

The key comparison is between **B** and **C**.

If planner fine-tuning with a frozen tracker improves execution quality while preserving semantic motion quality, then the paper has a strong argument:

> **Planner-side adaptation can reduce the burden on the tracker and produce more executable reference motions.**

---

### 6.2 Candidate Tasks

The tasks should expose the mismatch between kinematic realism and physical executability.

Good task candidates include:

#### Text-Conditioned Whole-Body Motion

Examples:

* “walk forward and wave the right hand,”
* “turn around and sit down,”
* “walk while crouching,”
* “step over a low obstacle,”
* “crawl forward,”
* “walk to the chair, turn around, sit, and raise the right hand.”

#### Constraint-Conditioned Motion

Examples:

* root trajectory following,
* wrist target reaching,
* sitting keypose generation,
* footstep or contact timing constraints,
* whole-body keyframe constraints.

#### Terrain- or Object-Aware Motion

Examples:

* stepping over obstacles,
* sitting on a chair,
* reaching toward an object on a table,
* navigating around a simple obstacle,
* transitioning from locomotion to manipulation.

#### Long-Horizon Motion

Long prompts are especially useful because tracker-only adaptation may struggle when local tracking errors accumulate over time.

---

## 7. Evaluation Metrics

The paper should evaluate both motion generation quality and robot execution quality.

### 7.1 Motion Generation Metrics

* FID,
* diversity,
* multimodality,
* R-precision,
* text-motion alignment,
* MPJPE against reference when available,
* foot skating,
* contact consistency,
* smoothness.

### 7.2 Robot Execution Metrics

* task success rate,
* fall rate,
* tracking error,
* root tracking error,
* end-effector tracking error,
* foot slip,
* contact violation,
* ground penetration,
* torque usage,
* energy consumption,
* joint-limit violation,
* center-of-mass stability,
* residual action magnitude,
* recovery rate after perturbation.

### 7.3 Semantic Preservation Metrics

* text-action classifier score,
* constraint satisfaction rate,
* keyframe satisfaction error,
* object interaction success,
* upper-body gesture preservation,
* long-horizon stage completion.

### 7.4 Pareto Analysis

A strong analysis would plot:

```text
Motion Naturalness / Diversity  vs.  Robot Execution Success
```

The goal is to show that tracker-aware planner fine-tuning improves the Pareto frontier between motion quality and physical executability.

---

## 8. Expected Findings

The expected findings are:

1. **Frozen planner + frozen tracker** fails when the generated motion is realistic but dynamically difficult to track.
2. **Tracker-only fine-tuning** improves robustness but may distort semantic motion or require significant retraining.
3. **Planner-only fine-tuning** can improve execution quality without changing the tracker.
4. **Joint fine-tuning** may achieve the best performance but requires careful regularization.
5. **Trackability-aware planner fine-tuning** is especially useful for long-horizon, contact-rich, or constraint-conditioned motions.

---

## 9. Key Failure Modes to Analyze

To make the paper stronger, it should include a taxonomy of planner–tracker failure modes.

### 9.1 Kinematically Plausible but Dynamically Infeasible Motion

The generated human motion looks natural, but the humanoid cannot execute it due to balance or torque constraints.

### 9.2 Contact Mismatch

The planner generates unrealistic foot contacts, foot sliding, or incorrect contact timing, making tracking unstable.

### 9.3 Morphology Mismatch

The motion is generated for a human body but executed by a humanoid robot with different proportions, joint limits, or actuation limits.

### 9.4 Tracker Compensation Destroys Semantics

The tracker avoids falling but changes the intended motion, such as reducing arm gestures or changing the body posture.

### 9.5 Long-Horizon Error Accumulation

Small tracking errors accumulate over long motions, causing drift, instability, or task failure.

### 9.6 Overly Conservative Adaptation

If fine-tuning is not regularized, the planner may collapse toward safe but low-diversity motions.

---

## 10. Possible Contributions

The paper can claim the following contributions:

1. **Problem Formulation**

   * Formulate planner–tracker alignment as a key problem in humanoid whole-body motion generation.
   * Argue that motion planners should be evaluated not only by kinematic realism but also by downstream trackability.

2. **Tracker-Aware Planner Fine-Tuning**

   * Introduce a method for adapting kinematic motion planners using rollout-derived trackability signals from a whole-body tracker.

3. **Trackability Critic or Trackability Objective**

   * Train a model or define a loss that predicts whether a generated reference motion is executable by a humanoid tracker.

4. **Systematic Adaptation Study**

   * Compare planner-only, tracker-only, and joint adaptation in a controlled humanoid control pipeline.

5. **Execution-Aware Evaluation**

   * Propose evaluation metrics that connect motion generation quality with robot execution quality.

6. **Failure Mode Analysis**

   * Analyze when tracker-only adaptation fails and when planner-side adaptation is beneficial.

---

## 11. Why This Is a Good ICRA 2027 Topic

This topic is timely because humanoid research is moving toward modular systems that combine:

* generative motion priors,
* whole-body tracking policies,
* vision-language-action systems,
* text- or scene-conditioned motion generation,
* long-horizon autonomous humanoid behaviors.

However, the interface between kinematic motion generation and physical control is still underexplored.

Most existing systems ask:

> “How can we train a better tracker?”

This project asks a different question:

> “How can we train the planner so that the tracker does not need to compensate for impossible references?”

This is a strong robotics question because it directly connects generative modeling with physical execution.

---

## 12. Paper Positioning

A possible positioning statement:

> Recent humanoid systems increasingly combine kinematic motion generators with whole-body tracking policies, but the two modules are typically trained separately and later reconciled by fine-tuning the tracker. We argue that this places the burden of physical feasibility entirely on the controller, while leaving the planner unaware of the robot's executable motion manifold. We introduce a tracker-aware fine-tuning framework that adapts the kinematic planner using rollout-derived trackability signals, producing references that preserve semantic motion intent while being easier and safer for humanoid trackers to execute.

---

## 13. Possible Paper Titles

### Clear and Direct

* **Tracker-Aware Kinematic Motion Planning for Humanoid Whole-Body Control**
* **Learning Tracker-Aware Motion Priors for Humanoid Whole-Body Control**
* **Trackability-Aware Motion Generation for Humanoid Robots**

### More Question-Driven

* **Who Should Adapt? Planner–Tracker Alignment for Humanoid Whole-Body Motion Generation**
* **Should the Planner or the Tracker Adapt? A Study of Humanoid Motion Generation Pipelines**

### More Robotics-Oriented

* **Executable Motion Priors for Humanoid Whole-Body Tracking**
* **From Human Motion Priors to Humanoid-Executable Motion Plans**
* **Aligning Kinematic Motion Generators with Humanoid Whole-Body Trackers**

### Recommended Title

> **Tracker-Aware Kinematic Motion Planning for Humanoid Whole-Body Control**

This title is clear, technical, and well aligned with ICRA.

---

## 14. One-Sentence Pitch

> We propose to make kinematic humanoid motion planners aware of downstream whole-body tracking, so that generated motions are not only realistic and semantically aligned, but also physically executable by a humanoid robot.

---

## 15. Short Abstract Draft

Recent humanoid control systems increasingly use a two-stage pipeline in which a kinematic motion generator produces reference motions and a whole-body tracking policy executes them on a robot. While this separation enables diverse and semantic motion generation, the planner and tracker are typically trained independently, leaving the tracker to compensate for reference motions that may be realistic for humans but difficult for humanoid robots to execute. In this work, we study the planner–tracker adaptation problem: should we fine-tune the planner, the tracker, or both? We propose a tracker-aware motion planning framework that adapts a kinematic motion generator using rollout-derived trackability signals from a whole-body tracker. By aligning the planner's output distribution with the tracker's executable motion manifold, our method generates references that preserve semantic intent while improving physical execution. We evaluate planner-only, tracker-only, and joint adaptation across text-conditioned, constraint-conditioned, and contact-rich humanoid motion tasks, demonstrating that tracker-aware planner fine-tuning reduces tracking burden, improves stability, and better preserves high-level motion intent.

---

## 16. Main Takeaway

The strongest version of this project is not simply:

> “Should we fine-tune the planner or the tracker?”

The stronger framing is:

> **A kinematic motion planner for humanoids should be trained not only as a human motion generator, but as an execution-aware reference generator for a downstream whole-body tracker.**

This makes the work more than an ablation study. It becomes a paper about the interface between generative motion models and physically embodied humanoid control.
