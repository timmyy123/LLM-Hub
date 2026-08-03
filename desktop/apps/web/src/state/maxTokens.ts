import type { AppConfig } from '../types';
import litellmData from './litellm-models.json';

// Per-model output cap, used to default `max_tokens` so users on supported
// models don't have to find Settings to avoid mid-stream truncation.
//
// Source of truth: vendored slice of BerriAI/litellm's
// model_prices_and_context_window.json (MIT). Regenerate with:
//   node --experimental-strip-types scripts/sync-litellm-models.ts
//
// Anything LiteLLM doesn't track (or where its value is wrong for our
// usage) goes in OVERRIDES; unknown models fall through to FALLBACK.
export const FALLBACK_MAX_TOKENS = 8192;

// Bounds the user can express via the Settings override. Source of truth
// for both the UI input attributes and runtime validation in
// `effectiveMaxTokens`, so a stale or hand-edited localStorage value
// can't sneak past the UI's promise.
export const MIN_MAX_TOKENS = 1024;
export const MAX_MAX_TOKENS = 200000;

const LITELLM_MODELS = litellmData.models as Record<string, number>;

const OVERRIDES: Record<string, number> = {
  // Ollama Cloud models. LiteLLM keys this set under `ollama/`-prefixed
  // ids (many with `-cloud` suffixes), so the bare model-id lookups never
  // match. Add overrides so chat doesn't silently clip at 8192 tokens.
  // 131072 (128k) is a safe floor for all Ollama Cloud models.
  // Exception: cogito-2.1:671b caps at 65536 (confirmed via upstream 400).
  'cogito-2.1:671b': 65536,
  'devstral-2:123b': 131072,
  'devstral-small-2:24b': 131072,
  'gemini-3-flash-preview': 131072,
  'gemma3:4b': 131072,
  'gemma3:12b': 131072,
  'gemma3:27b': 131072,
  'gemma4:31b': 131072,

  'gpt-oss:20b': 131072,
  'gpt-oss:120b': 131072,
  'ministral-3:3b': 131072,
  'ministral-3:8b': 131072,
  'ministral-3:14b': 131072,
  'mistral-large-3:675b': 131072,
  'nemotron-3-nano:30b': 131072,
  'nemotron-3-super': 131072,
  'nemotron-3-ultra': 131072,
  'rnj-1:8b': 131072,
};

export function modelMaxTokensDefault(model: string): number {
  return OVERRIDES[model] ?? LITELLM_MODELS[model] ?? FALLBACK_MAX_TOKENS;
}

function isValidOverride(value: number | undefined): value is number {
  return (
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= MIN_MAX_TOKENS &&
    value <= MAX_MAX_TOKENS
  );
}

export function effectiveMaxTokens(cfg: Pick<AppConfig, 'maxTokens' | 'model'>): number {
  // Out-of-range or non-integer overrides (stale localStorage, hand-edited
  // config, future schema drift) fall back to the model default rather
  // than silently shipping an invalid `max_tokens` upstream.
  if (isValidOverride(cfg.maxTokens)) return cfg.maxTokens;
  return modelMaxTokensDefault(cfg.model);
}
