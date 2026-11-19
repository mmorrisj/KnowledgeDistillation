import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const lessons = defineCollection({
  loader: glob({ pattern: "**/*.mdx", base: "./src/content/lessons" }),
  schema: z.object({
    title: z.string().min(1),
    order: z.number().int().positive(),
    minutes: z.number().int().positive(),
    objectives: z.array(z.string().min(1)).min(1).max(5),
    lab: z
      .object({
        title: z.string().min(1),
        path: z.string().min(1),
        minutes: z.number().int().positive(),
      })
      .optional(),
  }),
});

export const collections = { lessons };
