# Prompt for Code Generation Model: SAGA-SR Reproduction - Phase 2 (Implementation, Revised)

## Current Progress & **Revision**

Phase 1 (scaffolding) is complete. **Revision**: We will now use the T5 text encoder that comes bundled with the Stable Audio model (`stable-audio-open-1.0`), removing the dependency on a separate Flan-T5 model. This simplifies the architecture.

The file structure is `data.py`, `conditioning.py`, `blocks.py`, `model_sagasr.py`, and `train.py`. The core components are now all loaded from the Stable Audio pipeline, but the model's internal logic, training objective, and inference pipeline still need to be correctly implemented.

## Overall Goal for Phase 2

Your task is to **implement the core logic** to make the SAGA-SR model fully functional, runnable, and faithful to the paper's architecture, using the **integrated T5 encoder**. This involves correcting the DiT's forward pass, implementing the proper conditioning fusion, defining the correct training objective (CFM), and building the inference sampler.

---

## Task 1: Correct DiT Invocation and Conditioning Fusion (`model_sagasr.py`)

**Problem**: The DiT call is incorrect, and the conditioning logic needs to be updated to use the pipeline's internal T5 encoder.

**Detailed Instructions**:

1.  **Modify `SagaSR.__init__`**:
    *   Load the entire `StableAudioPipeline` from the local path.
    *   Assign `self.vae`, `self.dit`, **`self.text_encoder`**, and **`self.tokenizer`** from the loaded pipeline.
    *   Ensure the text encoder is in eval mode (`.eval()`).
    *   Remove the `t5_path` argument from the constructor.

2.  **Rewrite `SagaSR._compute_condition`**:
    *   This method should no longer call an external `encode_text_t5` function.
    *   **New Logic**:
        1.  Get `captions` from `caption_with_qwen`.
        2.  Use `self.tokenizer` to convert captions into `input_ids`.
        3.  Use `self.text_encoder` to get the `last_hidden_state` from the `input_ids`. This is your `text_embedding`.
        4.  Project the `text_embedding` to the DiT's `cross_attention_dim` as before.
    *   The roll-off conditioning logic remains the same.

---

## Task 2: Implement Conditional Flow Matching (CFM) Loss (`train.py`)

**Problem**: The current `F.l1_loss` is a placeholder. SAGA-SR uses CFM.

**Detailed Instructions**:

1.  **In `SagaSR.forward`**:
    *   Calculate the CFM target:
        *   `t = torch.rand(batch_size)`
        *   `z_t = t * z_h + (1 - t) * z_l`
        *   `target_velocity = z_h - z_l`
    *   Return the DiT's `prediction` and the `target_velocity`.

2.  **In `train.py`'s main loop**:
    *   The loss function should be `F.mse_loss(outputs['prediction'], outputs['target'])`.
    *   Remove the `--t5` argument from `parse_args()` and update the `SagaSR` instantiation.

---

## Task 3: Build the Inference Script (`infer.py`)

**Problem**: `infer.py` does not exist. We need a script to run the trained model.

**Detailed Instructions**:

1.  **Create `infer.py`**:
    *   Remove the `--t5` argument from the CLI parser.
    *   Update the `SagaSR` model loading to reflect the new constructor signature.

2.  **Implement Inference Logic**:
    *   The logic inside remains largely the same, but the text embedding will now be generated internally by the `SagaSR` model's `prepare_condition` method, which should be using the integrated T5 encoder.

---

## Task 4: Refine Ancillary Modules

**Detailed Instructions**:

1.  **In `conditioning.py`**:
    *   Remove the `_load_t5` and `encode_text_t5` functions as they are now obsolete. The text encoding is handled directly within the `SagaSR` class.

2.  **In `data.py`'s `AudioSuperResDataset`**:
    *   (No change from before) Add an optional step to apply a random low-pass filter before resampling to better simulate real-world degradation.

3.  **In `conditioning.py`'s `caption_with_qwen`**:
    *   (No change from before) Verify the output of Qwen-Audio. If it returns token IDs, use the processor's `batch_decode` method to get strings.

---

## Final Checklist

-   [ ] **`model_sagasr.py`**: Does `__init__` load the tokenizer and text encoder from `StableAudioPipeline`?
-   [ ] **`model_sagasr.py`**: Does `_compute_condition` use the internal tokenizer and text encoder?
-   [ ] **`conditioning.py`**: Have `_load_t5` and `encode_text_t5` been removed?
-   [ ] **`train.py` & `infer.py`**: Have the `--t5` CLI arguments been removed and model instantiation updated?
-   [ ] **`train.py`**: Is the loss function now CFM-based (`mse_loss`)?
-   [ ] **`infer.py`**: Does the script exist and implement a full ODE sampling loop with CFG?
-   [ ] **`data.py`**: Has a more realistic degradation (e.g., low-pass filter) been added?
