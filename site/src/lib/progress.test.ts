import { LocalProgressStore, percentComplete, PROGRESS_KEY, type Progress } from "./progress";

function throwingStorage(): Storage {
  return {
    getItem: () => {
      throw new Error("blocked");
    },
    setItem: () => {
      throw new Error("blocked");
    },
    removeItem: () => {},
    clear: () => {},
    key: () => null,
    length: 0,
  };
}

describe("LocalProgressStore", () => {
  it("starts empty", async () => {
    expect(await new LocalProgressStore().load()).toEqual({ lessons: {}, quizzes: {} });
  });

  it("persists a completed lesson across store instances", async () => {
    await new LocalProgressStore().completeLesson("m/a");
    const loaded = await new LocalProgressStore().load();
    expect(Object.keys(loaded.lessons)).toEqual(["m/a"]);
  });

  it("keeps the first completedAt when a lesson is completed twice", async () => {
    const store = new LocalProgressStore();
    await store.completeLesson("m/a");
    const first = (await store.load()).lessons["m/a"].completedAt;
    await store.completeLesson("m/a");
    expect((await store.load()).lessons["m/a"].completedAt).toBe(first);
  });

  it("counts quiz attempts and stores the last answer", async () => {
    const store = new LocalProgressStore();
    await store.recordQuiz("m/a", "q1", "first", false);
    await store.recordQuiz("m/a", "q1", "second", true);
    expect((await store.load()).quizzes["m/a#q1"]).toEqual({
      attempts: 2,
      correct: true,
      lastAnswer: "second",
    });
  });

  it("keeps correct once earned, even after a later wrong answer", async () => {
    const store = new LocalProgressStore();
    await store.recordQuiz("m/a", "q1", "right", true);
    await store.recordQuiz("m/a", "q1", "wrong", false);
    expect((await store.load()).quizzes["m/a#q1"].correct).toBe(true);
  });

  it("notifies subscribers until they unsubscribe", async () => {
    const store = new LocalProgressStore();
    const seen: Progress[] = [];
    const unsubscribe = store.subscribe((p) => seen.push(p));
    await store.completeLesson("m/a");
    unsubscribe();
    await store.completeLesson("m/b");
    expect(seen).toHaveLength(1);
    expect(seen[0].lessons["m/a"]).toBeDefined();
  });

  it("falls back to memory when storage throws", async () => {
    const store = new LocalProgressStore(throwingStorage());
    await store.completeLesson("m/a");
    expect((await store.load()).lessons["m/a"]).toBeDefined();
  });

  it("treats corrupt JSON as empty", async () => {
    localStorage.setItem(PROGRESS_KEY, "{not json");
    expect(await new LocalProgressStore().load()).toEqual({ lessons: {}, quizzes: {} });
  });

  it("ignores stored data of the wrong shape", async () => {
    localStorage.setItem(PROGRESS_KEY, JSON.stringify({ lessons: [1, 2], quizzes: "x" }));
    expect(await new LocalProgressStore().load()).toEqual({ lessons: {}, quizzes: {} });
    localStorage.setItem(PROGRESS_KEY, "null");
    expect(await new LocalProgressStore().load()).toEqual({ lessons: {}, quizzes: {} });
  });
});

describe("percentComplete", () => {
  const progress: Progress = { lessons: { "m/a": { completedAt: "x" }, "old/gone": { completedAt: "x" } }, quizzes: {} };
  it("rounds the share of listed lessons that are complete", () => {
    expect(percentComplete(progress, ["m/a", "m/b", "m/c", "m/d"])).toBe(25);
  });
  it("is 0 for an empty lesson list", () => {
    expect(percentComplete(progress, [])).toBe(0);
  });
});
