// pvx_tts.cpp — ABI v8 wave 3: speech output / TTS (mtmd gen-audio).
//
// Pattern-port of the vendored llama.cpp/tools/tts/tts.cpp driver loop via
// the PUBLIC mtmd-helper gen-audio surface (mtmd_helper_gen_audio_* — never
// the common/ runtimes): set_input → step_prompt* → {sample backbone token,
// step_gen} loop → get_output (raw PCM F32). The backbone sampler is our own
// deterministic top-k/top-p/greedy pipeline (the common_sampler chain is
// frozen-tree runtime code — off-limits).
//
// KV-neutrality (owner constraint, argued): the TTS session owns a SEPARATE
// llama context on the backbone speech model — no conversation context, no
// pool, no pvx KV surface is ever touched. Generation is fully isolated.

#include "pvx_internal.h"
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <mutex>

struct pvx_tts_session_s {
    pvx_model_t* model = nullptr;        // backbone speech model (non-owning)
    llama_context* lctx = nullptr;       // OWN context (never a conversation's)
    mtmd_context* mctx = nullptr;        // gen-audio mmproj (own — not shared
                                         // with the model's vision mtmd_ctx)
    int sample_rate = 0;
    uint32_t n_ctx = 0;
    std::mutex mtx;                       // serialize pvx_tts_generate calls
    std::string last_error;
    bool consumed = false;                // SINGLE-SHOT: see pvx_tts_generate
};

// ── Deterministic backbone sampler (vendored tts.cpp sample_semantic_code
//    pattern, our own pipeline): greedy when top_k == 1; otherwise top-k +
//    top-p renormalized softmax over the logits row, drawn from a
//    per-call RNG seeded by params (UINT32_MAX = random device). Identical
//    (text, seed) pairs replay identically — the determinism gate. ──
static llama_token tts_sample_backbone(llama_context* lctx, const llama_vocab* vocab,
                                       int top_k, float top_p, float temp,
                                       std::mt19937& rng) {
    const float* logits = llama_get_logits_ith(lctx, -1);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> cands;
    cands.reserve((size_t) n_vocab);
    for (int i = 0; i < n_vocab; i++) cands.push_back({ (llama_token) i, logits[i], 0.0f });

    auto by_logit = [](const llama_token_data& a, const llama_token_data& b) {
        return a.logit > b.logit;
    };
    std::sort(cands.begin(), cands.end(), by_logit);
    if (top_k > 0 && (size_t) top_k < cands.size())
        cands.resize((size_t) top_k);

    if (cands.size() == 1) return cands[0].id;  // greedy

    // top-p cut (keep the smallest prefix with mass >= top_p)
    if (top_p > 0.0f && top_p < 1.0f) {
        float maxl = cands[0].logit;
        double sum = 0.0;
        for (auto& c : cands) c.logit -= maxl;  // numerical safety
        for (auto& c : cands) { c.p = (float) std::exp((double) c.logit / (double) (temp > 0 ? temp : 1.0f)); sum += c.p; }
        double cum = 0.0; size_t keep = cands.size();
        for (size_t i = 0; i < cands.size(); i++) {
            cum += cands[i].p / sum;
            if (cum >= top_p) { keep = i + 1; break; }
        }
        cands.resize(keep);
    }

    // Softmax over the survivors (temp applied; temp<=0 → 1.0) then CDF draw.
    const float t = temp > 0 ? temp : 1.0f;
    double maxl = cands[0].logit;  // still shifted; re-shift each below
    double sum = 0.0;
    for (auto& c : cands) { c.p = (float) std::exp(((double) c.logit - maxl) / t); sum += c.p; }
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    double r = unif(rng) * sum;
    for (const auto& c : cands) {
        r -= (double) c.p;
        if (r <= 0.0) return c.id;
    }
    return cands.back().id;
}

pvx_result_t pvx_tts_session_create(pvx_model_t* backbone_model,
                                    const char* tts_mmproj_path,
                                    uint32_t context_size,
                                    pvx_tts_session_t** out_session) {
    if (!backbone_model || !tts_mmproj_path || !out_session) return PVX_ERR_INVALID_ARG;
    *out_session = nullptr;

    // Gen-audio mmproj against the backbone model — same public init the
    // vision path uses; a dedicated context (never shared with model->mtmd_ctx).
    mtmd_context_params mparams = mtmd_context_params_default();
    mtmd_context* mctx = mtmd_init_from_file(tts_mmproj_path, backbone_model->model, mparams);
    if (!mctx) {
        backbone_model->last_error = "tts/mmproj: mtmd_init_from_file failed for '" +
                                     std::string(tts_mmproj_path) + "'";
        return PVX_ERR_LOAD_FAILED;
    }

    // Capability truth: refuse BEFORE any context work when the pipeline is
    // absent (boundary: tts/gen-audio-capability).
    const mtmd_gen_audio_info info = mtmd_gen_audio_get_info(mctx);
    if (info.type == MTMD_GEN_AUDIO_TYPE_NONE) {
        mtmd_free(mctx);
        backbone_model->last_error =
            "tts/gen-audio-capability: this mmproj has no gen-audio pipeline "
            "(MTMD_GEN_AUDIO_TYPE_NONE; a QWEN3TTS/POCKETTTS-capable mmproj is required)";
        return PVX_ERR_NOT_SUPPORTED;
    }

    // OWN decode context on the backbone — isolated from every conversation.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = context_size > 0 ? context_size : 0;  // 0 → model default
    cp.n_seq_max = 1;
    // Decision-family backbone (qwen3tts): semantic codes are exposed via the
    // EMBEDDINGS output, not logits — mirror the vendored tts.cpp context
    // (common_init_from_params auto-enables embeddings for decision archs).
    // The hidden state for step_gen comes from llama_get_embeddings_ith(-1).
    cp.embeddings = true;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.flash_attn_type = backbone_model->flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                                    : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (backbone_model->threads > 0) cp.n_threads = backbone_model->threads;
    llama_context* lctx = llama_new_context_with_model(backbone_model->model, cp);
    if (!lctx) {
        mtmd_free(mctx);
        backbone_model->last_error = "tts/context: llama_new_context_with_model failed";
        return PVX_ERR_LOAD_FAILED;
    }

    auto* session = new pvx_tts_session_s();
    session->model = backbone_model;
    session->lctx = lctx;
    session->mctx = mctx;
    session->sample_rate = info.sample_rate;
    session->n_ctx = llama_n_ctx(lctx);
    *out_session = session;
    return PVX_OK;
}

void pvx_tts_session_free(pvx_tts_session_t* session) {
    if (!session) return;
    // Destruction order: context (GPU buffers) before the shared model device
    // goes away — same gate reasoning as pvx_free_context (the session's model
    // pointer is non-owning; the model's own refcount discipline is untouched).
    if (session->lctx) llama_free(session->lctx);
    if (session->mctx) mtmd_free(session->mctx);
    delete session;
}

pvx_result_t pvx_tts_generate(pvx_tts_session_t* session,
                              const pvx_tts_params_t* params,
                              const char* text,
                              float** out_pcm,
                              int* out_n_samples,
                              int* out_sample_rate) {
    if (!session || !params || !text || !out_pcm || !out_n_samples || !out_sample_rate)
        return PVX_ERR_INVALID_ARG;
    *out_pcm = nullptr; *out_n_samples = 0; *out_sample_rate = 0;
    const size_t tlen = strlen(text);
    if (tlen == 0) {
        session->last_error = "tts/text: empty prompt";
        return PVX_ERR_INVALID_ARG;
    }

    std::lock_guard<std::mutex> lock(session->mtx);

    // SINGLE-SHOT contract (upstream pipeline semantics): the gen-audio
    // pipeline carries cross-call state inside the mtmd context (code-predictor
    // accumulators), so a SECOND generate on the same session cannot be kept
    // exact — repeated generations on a live session are a typed refusal,
    // never a silently degraded/shifted utterance. Identical (text, seed)
    // pairs replay BYTE-IDENTICALLY on fresh sessions (gated). Fail closed.
    if (session->consumed) {
        session->last_error =
            "tts/session-consumed: single-shot session — the gen-audio pipeline "
            "carries cross-call state; create a fresh session per utterance";
        return PVX_ERR_NOT_SUPPORTED;
    }

    // Per-model defaults from the public gen-audio surface; params override
    // only what they set (no env vars).
    const mtmd_gen_inp def = mtmd_gen_inp_default(session->mctx);
    const int top_k = params->top_k > 0 ? params->top_k : def.top_k;
    const float top_p = params->top_p > 0.0f ? params->top_p : def.top_p;
    const float temp = params->temp > 0.0f ? params->temp : def.temp;
    const int max_frames = params->max_frames > 0 ? params->max_frames : 512;

    // Deterministic RNG per call: fixed seed replays identically (gate);
    // UINT32_MAX = random draw.
    std::mt19937 rng;
    if (params->seed == UINT32_MAX)
        rng.seed(std::random_device{}());
    else
        rng.seed(params->seed ? params->seed : 1234u);

    // Fresh pipeline per call (stateless session across calls: the backbone
    // context's seq 0 is cleared so nothing bleeds between generations).
    llama_memory_t mem = llama_get_memory(session->lctx);
    llama_memory_seq_rm(mem, 0, 0, -1);

    mtmd_helper_gen_audio* gen = mtmd_helper_gen_audio_init(session->lctx, session->mctx);
    if (!gen) {
        session->last_error = "tts/gen-init: mtmd_helper_gen_audio_init failed";
        return PVX_ERR_INTERNAL;
    }
    struct GenGuard {
        mtmd_helper_gen_audio* p;
        ~GenGuard() { if (p) mtmd_helper_gen_audio_free(p); }
    } gen_guard{ gen };

    mtmd_helper_gen_audio_inp inp = {};
    inp.seq_id = 0;
    inp.prompt = text;
    inp.prompt_len = tlen;
    inp.speaker_ref = nullptr;             // speaker-reference files stay out of
    inp.lang = (params->lang && params->lang[0]) ? params->lang : nullptr;
    inp.top_k = top_k;
    inp.top_p = top_p;
    inp.seed = params->seed ? params->seed : def.seed;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;  // raw PCM F32
    int32_t si = mtmd_helper_gen_audio_set_input(gen, &inp);
    if (si != 0) {
        pvx_log(PVX_LOG_ERROR, "TTS", "gen_audio set_input failed rc=%d", si);
        session->last_error = "tts/prompt-stage: set_input failed (rc=" +
                              std::to_string(si) + "; lang='" +
                              std::string(inp.lang ? inp.lang : "") + "')";
        return PVX_ERR_DECODE_FAILED;
    }

    // Stage 1: prompt processing (chunked by n_batch, upstream loop shape).
    const int32_t n_batch = 512;
    for (;;) {
        int32_t ret = mtmd_helper_gen_audio_step_prompt(gen, n_batch);
        if (ret < 0) {
            pvx_log(PVX_LOG_ERROR, "TTS", "gen_audio step_prompt failed rc=%d", ret);
            session->last_error = "tts/prompt-stage: step_prompt failed (rc=" +
                                  std::to_string(ret) + ")";
            return PVX_ERR_DECODE_FAILED;
        }
        if (ret == 0) break;
    }

    // Stage 2+3: { sample backbone code, step_gen } until EOS or frame cap.
    const llama_vocab* vocab = llama_model_get_vocab(session->model->model);
    llama_token sampled = tts_sample_backbone(session->lctx, vocab, top_k, top_p, temp, rng);
    const float* h_state = llama_get_embeddings_ith(session->lctx, -1);
    if (!h_state) {
        session->last_error = "tts/prompt-stage: no backbone hidden state after prompt";
        return PVX_ERR_DECODE_FAILED;
    }

    int n_frames = 0;
    bool stop = false;
    while (!stop && n_frames < max_frames) {
        const float* h_next = nullptr;
        int32_t gret = mtmd_helper_gen_audio_step_gen(gen, sampled, h_state, &h_next, &stop);
        if (gret != 0) {
            pvx_log(PVX_LOG_ERROR, "TTS", "gen_audio step_gen failed rc=%d at frame %d", gret, n_frames);
            session->last_error = "tts/gen-stage: step_gen failed at frame " +
                                 std::to_string(n_frames) + " (rc=" + std::to_string(gret) + ")";
            return PVX_ERR_DECODE_FAILED;
        }
        if (!h_next) break;  // stopped without generating a frame
        n_frames++;
        h_state = h_next;
        sampled = tts_sample_backbone(session->lctx, vocab, top_k, top_p, temp, rng);
    }

    // Stage 4: collect PCM.
    int32_t sample_rate = 0;
    const char* data = nullptr;
    size_t data_len = 0;
    int64_t n_samples_i = 0;
    if (mtmd_helper_gen_audio_get_output(gen, &sample_rate, &data, &data_len, &n_samples_i) != 0) {
        session->last_error = "tts/wav-stage: get_output failed";
        return PVX_ERR_DECODE_FAILED;
    }
    if (!data || data_len == 0 || n_samples_i <= 0) {
        session->last_error = "tts/wav-stage: pipeline produced no audio (frame cap "
                              "hit before any frame? text too short?)";
        return PVX_ERR_DECODE_FAILED;
    }

    const size_t n_samples = (size_t) n_samples_i;
    auto* pcm = (float*) malloc(n_samples * sizeof(float));
    if (!pcm) return PVX_ERR_INTERNAL;
    memcpy(pcm, data, data_len < n_samples * sizeof(float) ? data_len : n_samples * sizeof(float));

    llama_memory_seq_rm(mem, 0, 0, -1);  // leave the isolated context clean

    *out_pcm = pcm;
    *out_n_samples = (int) n_samples;
    *out_sample_rate = (int) sample_rate;
    session->consumed = true;
    return PVX_OK;
}

void pvx_free_audio_samples(float* pcm) { free(pcm); }
