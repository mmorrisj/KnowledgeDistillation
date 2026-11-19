export function fullParams(dIn: number, dOut: number): number {
  return dIn * dOut;
}

/** Weights in the two LoRA matrices: A is rank x dIn, B is dOut x rank. */
export function loraParams(dIn: number, dOut: number, rank: number): number {
  return rank * (dIn + dOut);
}

export function loraShare(dIn: number, dOut: number, rank: number): number {
  return loraParams(dIn, dOut, rank) / fullParams(dIn, dOut);
}

export type Method = "full" | "lora" | "qlora";

// Rough bytes of GPU memory per model weight while training.
// full: 32-bit weights, gradients and two optimizer values per weight.
// lora: frozen 16-bit weights; the adapters are too small to matter.
// qlora: frozen 4-bit weights plus some overhead.
const BYTES_PER_WEIGHT: Record<Method, number> = { full: 16, lora: 2, qlora: 0.7 };

/** Rough training memory for a model of `billions` weights, before working memory. */
export function trainingMemoryGb(billions: number, method: Method): number {
  return billions * BYTES_PER_WEIGHT[method];
}
