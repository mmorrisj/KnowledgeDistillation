import type { ModuleInfo } from "../lib/nav";

/** Course modules in reading order. Lessons attach to a module by folder name. */
export const MODULES: ModuleInfo[] = [
  { slug: "00-sample", title: "Component sampler", status: "available", hidden: true },
  { slug: "01-foundations", title: "Foundations", status: "planned" },
  { slug: "02-baselines", title: "Baselines and evaluation", status: "planned" },
  { slug: "03-white-box", title: "White-box distillation", status: "planned" },
  { slug: "04-black-box", title: "Black-box distillation", status: "planned" },
  { slug: "05-lora", title: "LoRA and QLoRA", status: "planned" },
  { slug: "06-generative", title: "Distilling generative models", status: "planned" },
  { slug: "07-serve", title: "Compress and serve", status: "planned" },
  { slug: "08-vision", title: "Vision and multimodal", status: "planned" },
];
