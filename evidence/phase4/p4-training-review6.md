## P4B review — NOT ACCEPTED

Reviewed exact commit: **`b91cc5c9404445d8c644136f404ed7e5d07de6c9`**

Accepted base: `8562a4b0b6fe5c3baf0e463c9ab213e00aae1dcd`  
Latest-fix parent: `2711fbf0a73365633090eedeaa6133004c140071`

### Findings

1. **High — cache consumers do not bind the exact classifier tokenizer/configuration identity**

   Locations: [evaluation.py:1961](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review6/src/mmdc_clip_f/research/view_risk/evaluation.py:1961), [cli.py:493](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review6/src/mmdc_clip_f/research/view_risk/cli.py:493), [cli.py:1036](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review6/src/mmdc_clip_f/research/view_risk/cli.py:1036), [training.py:1811](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review6/src/mmdc_clip_f/research/view_risk/training.py:1811)

   Evaluation now compares checkpoint, backbone, prompts, preprocessing and fusion, but never compares `CacheProvenance.text_input_sha256` with the exact token IDs persisted by the selected classifier’s public-initialization artifact. Learned fitting, tune selection and analytic-control loading are weaker still: they primarily compare only `checkpoint_sha256`.

   Concrete synthetic reproductions:

   - An evaluation cache used the selected weights and prompt strings but different token IDs. `validate_evaluation_cache_provenance()` accepted it:

     ```text
     accepted_realization= True
     prompts_match= True
     tokenization_matches_frozen_source= False
     ```

   - A tune-control cache used the selected checkpoint but reversed class prompts. The exact `_control_cache_index()` → `_scalar_rows_from_control_caches()` path used by `view-risk-fit-control` accepted it:

     ```text
     accepted_rows= ('exam-a',)
     cache_prompts_match_selected= False
     checkpoint_matches= True
     ```

   Impact: authoritative classifier predictions, confidence fitting, tune selection, or controls can be derived from different text embeddings/class semantics while appearing bound to the selected classifier. This breaks the shared-frozen-classifier requirement.

   Minimal correction: persist the exact `FrozenEncoderIdentity`, including `text_input_sha256`, with the selected classifier/plan and use one shared validator for every cache consumer. Compare checkpoint, HF model/revision, dimensions, preprocessing, normalization, prompt order, token hash, and dataset fusion tree.

2. **High — tune and analytic-control caches are not constrained to their prescribed realizations**

   Locations: [cli.py:514](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review6/src/mmdc_clip_f/research/view_risk/cli.py:514), [cli.py:796](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review6/src/mmdc_clip_f/research/view_risk/cli.py:796), [cli.py:1351](/home/tianqini/research/.ivr-p4-worktrees/p4-training-review6/src/mmdc_clip_f/research/view_risk/cli.py:1351)

   `_validate_tune_cache_realization()` checks only family/severity/target labels. It does not require the intended four-view tune input or derive the exact realization seed/spec from the authorized record and frozen configuration. Control loading checks the cohort and checkpoint but has no method-specific clean/matched-draw contract.

   Concrete reproductions:

   - A three-view Gaussian-noise tune cache with realization seed `37` was accepted for the nominal tune cell:

     ```text
     accepted=True
     observed_views= ('L_CC', 'R_CC', 'R_MLO')
     realization_seed= 37
     ```

   - The same kind of stressed, masked tune cache was accepted as temperature-scaling input:

     ```text
     accepted_rows= ('exam-a',)
     observed_views= ('L_CC', 'R_CC', 'R_MLO')
     R_CC_family= gaussian_noise
     ```

   Impact: checkpoint selection can use a different mask/realization than the frozen balanced tune panel; TS or scalar calibration can be fitted on non-clean data; DS fitting can use arbitrary unmatched draws. The resulting artifacts correctly rehash the wrong evidence, so reload does not repair the scientific mismatch.

   Minimal correction: derive and validate the complete expected tune specification—including full mask, seed, target, realized parameters and clean non-target views—from the authorized record/config. Give control indexes method-specific schemas: clean full-view tune caches for TS/calibration, and the exact deterministic confidence-fit/tune schedule, masks and exposure budget for DS. Bind the resulting realization/index identities across matched selections.

### Review4 finding disposition

- **Referenced tune-cache bytes unchecked:** fixed. Selection records bind every metadata/tensor pair and reject changed bytes on reload.
- **Evaluation cells unbound to realized stresses / standalone authority:** stress family, severity, variant, mask, cohort and cache bytes are now validated, and standalone prediction JSON was removed. Still incomplete because exact tokenizer identity is not checked—finding 1.
- **Standalone analytic-control inputs:** standalone feature JSON was removed and cache files are rehashed. Still incomplete because exact classifier identity and clean/matched realization contracts are not enforced—findings 1 and 2.

### Earlier four-finding disposition

- **Synthetic provenance promotion:** fixed; production plan reload reconstructs initialization, readiness, fit, tune and checkpoint evidence.
- **Conflicting authoritative prediction sets:** fixed; the plan owns one exclusive registry path and scoring reloads that identity. Its source-classifier validity remains affected by finding 1.
- **Incomplete learned/control artifacts:** learned-selection cache-byte verification is fixed; control validity remains incomplete as above.
- **Disconnected production workflows:** functionally connected through public commands and cache-derived scoring/fitting, but the invalid cache boundaries prevent acceptance.

### Validation

- Focused cache/evaluation/artifact/production suite: **68 passed**
- Full research suite: **258 passed**
- Ruff over `src` and `tests/research`: **passed**
- Compileall: **passed**
- Source/test latest-fix `git diff --check`: **passed**
- Full latest-fix diff reports one nonfunctional trailing space in copied `p4-training-review4.md:5`.
- Both required commits are ancestors of the reviewed commit.
- Worktree remained clean.

No real medical data, model downloads, public-weight loading, real training, pilot outcomes, or locked-test access were used. P4C smoke/cost/CI and P5 resource execution were not assessed.