import { resolveTheme } from "./theme";

describe("resolveTheme", () => {
  it("uses an explicit stored choice over the system setting", () => {
    expect(resolveTheme("dark", false)).toBe("dark");
    expect(resolveTheme("light", true)).toBe("light");
  });
  it("follows the system setting when nothing is stored", () => {
    expect(resolveTheme(null, true)).toBe("dark");
    expect(resolveTheme(null, false)).toBe("light");
  });
  it("ignores unrecognized stored values", () => {
    expect(resolveTheme("garbage", false)).toBe("light");
  });
});
