export type QuizRecord = { attempts: number; correct: boolean; lastAnswer: string };
export type Progress = {
  lessons: Record<string, { completedAt: string }>;
  quizzes: Record<string, QuizRecord>;
};

/** Where progress is kept. Lessons and components only ever talk to this interface. */
export interface ProgressStore {
  load(): Promise<Progress>;
  completeLesson(lessonId: string): Promise<void>;
  recordQuiz(lessonId: string, quizId: string, answer: string, correct: boolean): Promise<void>;
  subscribe(fn: (progress: Progress) => void): () => void;
}

export const PROGRESS_KEY = "kd-course:progress:v1";

const empty = (): Progress => ({ lessons: {}, quizzes: {} });

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parse(raw: string | null): Progress {
  if (!raw) return empty();
  try {
    const data: unknown = JSON.parse(raw);
    if (!isRecord(data) || !isRecord(data.lessons) || !isRecord(data.quizzes)) return empty();
    return { lessons: data.lessons, quizzes: data.quizzes } as Progress;
  } catch {
    return empty();
  }
}

/** Keeps progress in the browser. Falls back to memory when storage is unavailable. */
export class LocalProgressStore implements ProgressStore {
  private memory: Progress | null = null;
  private listeners = new Set<(progress: Progress) => void>();
  private storage: Storage | null;

  constructor(storage?: Storage) {
    this.storage = storage ?? (typeof localStorage === "undefined" ? null : localStorage);
  }

  private read(): Progress {
    if (this.memory) return this.memory;
    try {
      return parse(this.storage?.getItem(PROGRESS_KEY) ?? null);
    } catch {
      return empty();
    }
  }

  private write(progress: Progress): void {
    try {
      if (!this.storage) throw new Error("no storage");
      this.storage.setItem(PROGRESS_KEY, JSON.stringify(progress));
    } catch {
      // Storage is blocked or full: keep progress for this page view.
      this.memory = progress;
    }
    this.listeners.forEach((fn) => fn(progress));
  }

  async load(): Promise<Progress> {
    return this.read();
  }

  async completeLesson(lessonId: string): Promise<void> {
    const progress = this.read();
    if (progress.lessons[lessonId]) return;
    this.write({
      ...progress,
      lessons: { ...progress.lessons, [lessonId]: { completedAt: new Date().toISOString() } },
    });
  }

  async recordQuiz(lessonId: string, quizId: string, answer: string, correct: boolean): Promise<void> {
    const progress = this.read();
    const key = `${lessonId}#${quizId}`;
    const previous = progress.quizzes[key];
    this.write({
      ...progress,
      quizzes: {
        ...progress.quizzes,
        [key]: {
          attempts: (previous?.attempts ?? 0) + 1,
          correct: correct || (previous?.correct ?? false),
          lastAnswer: answer,
        },
      },
    });
  }

  subscribe(fn: (progress: Progress) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}

let shared: ProgressStore | null = null;

/** The store every component shares. The tracking plan adds a server-backed branch here. */
export function getProgressStore(): ProgressStore {
  shared ??= new LocalProgressStore();
  return shared;
}

/** Test hook: swap or reset the shared store. */
export function setProgressStore(store: ProgressStore | null): void {
  shared = store;
}

export function percentComplete(progress: Progress, lessonIds: string[]): number {
  if (lessonIds.length === 0) return 0;
  const done = lessonIds.filter((id) => progress.lessons[id]).length;
  return Math.round((done / lessonIds.length) * 100);
}
