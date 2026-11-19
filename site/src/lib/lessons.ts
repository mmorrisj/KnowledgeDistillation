import { getCollection } from "astro:content";
import { MODULES } from "../content/modules";
import { buildNav, type NavModule } from "./nav";

/** Navigation for the whole course, built from the lesson files on disk. */
export async function getNav(): Promise<NavModule[]> {
  const entries = await getCollection("lessons");
  return buildNav(
    entries.map((entry) => ({ id: entry.id, title: entry.data.title, order: entry.data.order })),
    MODULES,
  );
}
