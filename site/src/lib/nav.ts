export type ModuleInfo = { slug: string; title: string; status: "available" | "planned" };
export type LessonInfo = { id: string; title: string; order: number };
export type NavLesson = { id: string; title: string; href: string };
export type NavModule = ModuleInfo & { lessons: NavLesson[] };

/** Group lessons under their module, in module order then lesson order. */
export function buildNav(lessons: LessonInfo[], modules: ModuleInfo[]): NavModule[] {
  return modules.map((module) => ({
    ...module,
    lessons: lessons
      .filter((lesson) => lesson.id.split("/")[0] === module.slug)
      .sort((a, b) => a.order - b.order)
      .map(({ id, title }) => ({ id, title, href: `/modules/${id}/` })),
  }));
}

/** The lessons before and after `id` in reading order, across module boundaries. */
export function adjacent(nav: NavModule[], id: string): { prev?: NavLesson; next?: NavLesson } {
  const flat = nav.flatMap((module) => module.lessons);
  const index = flat.findIndex((lesson) => lesson.id === id);
  if (index === -1) return {};
  return { prev: flat[index - 1], next: flat[index + 1] };
}
