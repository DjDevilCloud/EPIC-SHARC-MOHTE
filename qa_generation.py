"""Answer-independent generation profiles shared by model, API and CLI."""
PROFILES = ('legacy', 'native_qa_eval_v1')
NATIVE_QA_EVAL = dict(max_new_tokens=128, min_new_tokens=0, top_k=1, top_p=1.,
                      temperature=0., repetition_penalty=1., no_repeat_ngram_size=0,
                      beam_size=1, use_speculative_decoding=False)
MODEL_LEGACY = dict(max_new_tokens=64, min_new_tokens=1, top_k=0, top_p=1.,
                    temperature=1., repetition_penalty=1.08, no_repeat_ngram_size=4,
                    beam_size=1, use_speculative_decoding=None)
API_LEGACY = dict(MODEL_LEGACY, use_speculative_decoding=False)
CLI_LEGACY = dict(MODEL_LEGACY, top_k=8, top_p=.92, temperature=.9,
                  repetition_penalty=1.1)


def resolve_generation_options(profile, supplied, legacy_defaults=None):
    if profile not in PROFILES:
        raise ValueError('Unknown QA generation profile')
    defaults = dict((legacy_defaults or MODEL_LEGACY) if profile=='legacy' else NATIVE_QA_EVAL)
    unknown = set(supplied)-set(MODEL_LEGACY)
    if unknown:
        raise ValueError(f'Unknown generation options: {sorted(unknown)}')
    explicit = {k:v for k,v in supplied.items() if v is not None}
    options = dict(defaults, **explicit)
    changed = {k:v for k,v in options.items() if profile!='legacy' and v!=NATIVE_QA_EVAL[k]}
    record = dict(profile=profile, canonical=profile!='legacy' and not changed,
                  overrides=changed, options=dict(options))
    return options,record
