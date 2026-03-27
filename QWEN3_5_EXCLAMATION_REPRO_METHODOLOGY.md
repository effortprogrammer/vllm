# Qwen3.5 `!!!` Reproduction Methodology

## Goal

Define a test plan to reproduce and classify intermittent Qwen3.5 failures where
responses collapse into repeated exclamation marks or similar degenerate output.

This note is intentionally about **reproduction methodology**, not a final root
cause claim.

## Working model of the failure

Based on the current code and issue review, the most likely high-level classes
of failure are:

1. **Hybrid state contamination**
   - `ssm_state` corruption in the GDN / recurrent path
   - stale state reuse across requests
2. **Attention KV corruption**
   - full-attention KV cache block contamination
   - stale block reuse / block-table mismatch
3. **Backend numeric corruption without persistent state contamination**
   - kernel writes invalid values directly into outputs or state
4. **Configuration / precision failures**
   - FP8 scale issues, quantization-specific corruption, or long-context numeric drift
5. **Lower-priority stateless causes**
   - tokenizer / sampling / formatting issues

The observed symptom pattern of:

- normal outputs at startup,
- then some requests returning `!!!`,
- then the frequency suddenly increasing,

fits **stateful contamination / stale reuse** better than a purely stateless bug.

## Why Qwen3.5 needs a separate plan

Qwen3.5 is a hybrid model in vLLM.

There are two distinct state families to consider:

1. **Full-attention KV cache**
2. **GDN recurrent state** (`ssm_state`)

The relevant Qwen3.5 GDN path reads and writes:

- `self.kv_cache[0]` as convolution state
- `self.kv_cache[1]` as `ssm_state`

and indexes recurrent state through:

- `spec_state_indices_tensor`
- `non_spec_state_indices_tensor`

So a Qwen3.5 `!!!` failure can plausibly come from:

- attention KV corruption,
- `ssm_state` corruption,
- or stale index / stale block reuse.

## Test philosophy

The plan should answer three questions separately:

1. **What state can produce `!!!` if corrupted?**
2. **Which state gets corrupted first in real runs?**
3. **What runtime condition creates that corruption?**

That means fault injection and organic reproduction should both be used.

## Phase 1: Establish a clean baseline

Before any injection or stress run:

1. restart the server / worker process
2. fix model, prompt set, backend, and sampling parameters
3. collect a baseline batch of requests
4. confirm that `!!!` does not appear initially

### Baseline controls

Keep these fixed during each experiment:

- model revision
- tensor parallel / data parallel settings
- backend (ROCm AITER / Triton / FlashInfer, etc.)
- quantization mode
- `temperature`, `top_p`, `top_k`, `max_tokens`
- prefix caching on/off
- speculative decoding on/off
- CUDA graph on/off

## Phase 2: Fault-injection tests

These tests do **not** prove the original root cause.
They determine which state is sufficient to trigger the symptom.

### Test group A: `ssm_state` contamination

**Purpose**: Determine whether corrupting Qwen3.5 recurrent state is sufficient
to create `!!!` and cross-request degradation.

**Injection target**:

- `gdn_linear_attn.py` path using `self.kv_cache[1]`

**Method**:

1. run a normal request to populate state
2. pick one or more `state_idx` entries
3. overwrite a small slice of `ssm_state[state_idx]` with `NaN`
4. send follow-up requests that are likely to reuse the same state slots

**Observe**:

- whether only the affected requests fail
- whether failures look like `!!!`, punctuation spam, or general gibberish
- whether the corruption spreads to additional state indices over time

### Test group B: attention KV contamination

**Purpose**: Determine whether corrupting full-attention KV cache is sufficient
to create the same symptom.

**Injection target**:

- one KV cache block or one slot range in the full-attention group

**Method**:

1. run a normal request to populate attention KV
2. overwrite a targeted K/V slice with `NaN`
3. issue follow-up requests that reuse the affected block(s)

**Observe**:

- whether `!!!` appears
- whether the symptom profile matches the `ssm_state` injection case
- whether corruption stays local or spreads

### Test group C: stale-index / stale-block simulation

**Purpose**: Check whether wrong reuse alone is enough, even without explicit
manual NaN writes.

**Method**:

1. stress request creation / finish / reuse cycles
2. keep prefix caching and batching enabled
3. track block-table row reuse and request teardown timing
4. look for stale state/block references that survive request completion

**Observe**:

- whether bad outputs begin only after reuse churn
- whether the first failing request correlates with a recently freed slot/block

## Phase 3: Organic reproduction matrix

After fault injection shows what is sufficient, run a matrix to determine what
runtime conditions make failures appear without manual corruption.

### Matrix axes

#### 1. State path

- prefill-heavy
- decode-heavy
- mixed prefill + decode
- speculative decode

#### 2. Cache / reuse behavior

- prefix caching off
- prefix caching on
- low concurrency
- high concurrency
- short-lived requests
- churn-heavy request turnover

#### 3. Backend

- ROCm / AITER
- Triton / FLA-derived path
- FlashInfer path if applicable

#### 4. Numeric configuration

- BF16 baseline
- FP8 or quantized model variant
- `--calculate-kv-scales` off / on when relevant
- short context vs long context

## Phase 4: First-corruption detection

Once a real run reproduces the bug, the next goal is not “observe `!!!` again”
but “identify what became invalid first.”

### Required logging

For each step / request batch, log enough information to answer:

1. which request ids were active
2. which block ids / state indices they used
3. whether any tracked KV blocks contained `NaN`
4. whether any tracked `ssm_state` entries contained `NaN`
5. whether the first corrupted output happened after a specific reuse event

### Success criterion

A strong signal would look like:

1. all tracked state clean at startup
2. one request or kernel path introduces first invalid values
3. later requests reusing the same slot/index begin failing
4. failure frequency rises as reuse expands

## Proposed test list

### Tier 1: Must-run

1. **`ssm_state` NaN injection**
   - Goal: prove or reject recurrent-state sufficiency
2. **Attention KV NaN injection**
   - Goal: prove or reject KV-cache sufficiency
3. **Long-running reuse stress on ROCm**
   - Goal: match the real “starts clean, then degrades” pattern
4. **Prefix caching on/off comparison**
   - Goal: determine how much reuse contributes

### Tier 2: Strong follow-up

1. **Spec decode on/off**
2. **CUDA graph on/off**
3. **Long-context vs short-context**
4. **Quantized vs non-quantized**

### Tier 3: Deeper narrowing

1. **Packed decode path isolation**
2. **AITER vs non-AITER backend comparison**
3. **Block-table churn test around request completion**

## What each test should record

Every experiment should log:

- exact model id
- exact commit / branch
- backend
- cache mode
- prompt length
- decode length
- concurrency
- whether `!!!` appeared
- whether output degraded in another way
- whether `NaN` was detected in KV cache
- whether `NaN` was detected in `ssm_state`

## Interpretation rules

### If only `ssm_state` injection reproduces the symptom

Most likely direction:

- GDN recurrent path corruption
- stale state reuse
- bad `ssm_state_indices` usage

### If only KV injection reproduces the symptom

Most likely direction:

- full-attention cache corruption
- stale block reuse
- block-table cleanup / allocation bug

### If both reproduce the symptom

Interpretation:

- `!!!` is a generic collapse symptom for invalid persistent state
- more instrumentation is needed to find the **first** corrupted state in real runs

### If neither reproduces the symptom

Interpretation:

- the true issue may be direct output/logit corruption
- or a very specific runtime path not covered by the injection setup

## Practical next step

Implement the experiments in this order:

1. `ssm_state` NaN injection
2. KV cache NaN injection
3. reuse churn / stale block stress
4. backend and config matrix

This order is optimized for Qwen3.5 specifically, where hybrid recurrent state
appears to be the most suspicious component.
