/** Raw space for stored embeddings: one 4-byte number per dimension per text. */
export function vectorStorageGb(count: number, dimensions: number, bytesPerNumber = 4): number {
  return (count * dimensions * bytesPerNumber) / 1024 ** 3;
}

/** Share of the first k results that two ranked lists have in common. */
export function overlapAtK(first: string[], second: string[], k: number): number {
  const a = first.slice(0, k);
  const b = new Set(second.slice(0, k));
  return a.filter((item) => b.has(item)).length / k;
}
