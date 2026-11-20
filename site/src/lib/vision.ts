/** How a vision transformer cuts a square image into square patches. */
export function patchGrid(imageSize: number, patchSize: number): { perSide: number; patches: number; tokens: number } {
  const perSide = Math.floor(imageSize / patchSize);
  const patches = perSide * perSide;
  // One extra token is added to stand for the whole image.
  return { perSide, patches, tokens: patches + 1 };
}
