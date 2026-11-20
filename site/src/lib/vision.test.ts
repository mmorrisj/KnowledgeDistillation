import { patchGrid } from "./vision";

describe("patchGrid", () => {
  it("counts the patches of a 224-pixel image cut into 16-pixel squares", () => {
    expect(patchGrid(224, 16)).toEqual({ perSide: 14, patches: 196, tokens: 197 });
  });
  it("quadruples the patch count when the patch size halves", () => {
    expect(patchGrid(224, 8).patches).toBe(784);
  });
  it("drops the part of the image that does not fill a whole patch", () => {
    expect(patchGrid(224, 32).perSide).toBe(7);
    expect(patchGrid(100, 16).perSide).toBe(6);
  });
});
