// BL-05 debug driver (not committed): mirror the C# spec flow exactly.
#include "pvxinference.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    pvx_set_log_level(PVX_LOG_DEBUG);
    pvx_model_params_t mp = {}; mp.model_path = getenv("PVX_MODEL_PATH"); mp.gpu_layers = 0; mp.flash_attn = true;
    pvx_model_t* m = NULL;
    if (pvx_load_model(&mp, &m) != PVX_OK) { printf("load failed\n"); return 1; }
    pvx_spec_capabilities_t caps{};
    pvx_model_spec_capabilities(m, &caps);
    printf("caps types=0x%x mtp=%d draft=%d\n", caps.types, caps.mtp, caps.draft);
    pvx_context_params_t cp = {}; cp.context_size = 1024; cp.batch_size = 512; cp.seq_max = 1;
    pvx_context_t* ctx = NULL;
    if (pvx_create_context(m, &cp, &ctx) != PVX_OK) { printf("ctx failed: %s\n", pvx_last_error(ctx)); return 1; }
    pvx_executor_t* exec = NULL;
    if (pvx_executor_create(ctx, &exec) != PVX_OK) return 1;
    pvx_spec_params_t sp = {}; sp.mode = PVX_SPEC_DRAFT; sp.draft_model_path = getenv("PVX_DRAFT_MODEL_PATH"); sp.draft_max = 4;
    pvx_result_t r = pvx_executor_spec_init(exec, &sp);
    printf("spec_init: %d err=%s\n", (int) r, pvx_last_error(ctx));
    if (r != PVX_OK) return 1;
    if (pvx_executor_prompt(exec, "List three colors:") != PVX_OK) { printf("prompt failed\n"); return 1; }
    if (pvx_executor_infer(exec) != PVX_DECODE_OK) { printf("infer failed: %s\n", pvx_last_error(ctx)); return 1; }
    pvx_sampling_params_t samp = {}; samp.temperature = 0.0f; samp.top_k = 1; samp.top_p = 1.0f; samp.repeat_penalty = 1.0f;
    int n = 0;
    r = pvx_executor_generate_spec(exec, &samp, 12, NULL, NULL, &n);
    printf("generate_spec: r=%d n=%d err=%s\n", (int) r, n, pvx_last_error(ctx));
    return 0;
}
