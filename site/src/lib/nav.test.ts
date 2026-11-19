import { adjacent, buildNav, type ModuleInfo } from "./nav";

const modules: ModuleInfo[] = [
  { slug: "01-a", title: "A", status: "available" },
  { slug: "02-b", title: "B", status: "available" },
  { slug: "03-c", title: "C", status: "planned" },
];
const lessons = [
  { id: "01-a/02-second", title: "Second", order: 2 },
  { id: "01-a/01-first", title: "First", order: 1 },
  { id: "02-b/01-only", title: "Only", order: 1 },
  { id: "99-ghost/01-x", title: "Ghost", order: 1 },
];

describe("buildNav", () => {
  it("sorts lessons by order inside their module", () => {
    const nav = buildNav(lessons, modules);
    expect(nav[0].lessons.map((l) => l.id)).toEqual(["01-a/01-first", "01-a/02-second"]);
  });
  it("keeps planned modules with an empty lesson list", () => {
    const nav = buildNav(lessons, modules);
    expect(nav[2]).toMatchObject({ slug: "03-c", status: "planned", lessons: [] });
  });
  it("builds hrefs as /modules/<module>/<lesson>/", () => {
    const nav = buildNav(lessons, modules);
    expect(nav[1].lessons[0].href).toBe("/modules/02-b/01-only/");
  });
  it("ignores lessons whose module is not in MODULES", () => {
    const ids = buildNav(lessons, modules).flatMap((m) => m.lessons.map((l) => l.id));
    expect(ids).not.toContain("99-ghost/01-x");
  });
});

describe("adjacent", () => {
  const nav = buildNav(lessons, modules);
  it("crosses module boundaries", () => {
    const { prev, next } = adjacent(nav, "01-a/02-second");
    expect(prev?.id).toBe("01-a/01-first");
    expect(next?.id).toBe("02-b/01-only");
  });
  it("is undefined at the ends", () => {
    expect(adjacent(nav, "01-a/01-first").prev).toBeUndefined();
    expect(adjacent(nav, "02-b/01-only").next).toBeUndefined();
  });
  it("returns nothing for an unknown lesson", () => {
    expect(adjacent(nav, "nope/nope")).toEqual({});
  });
});
