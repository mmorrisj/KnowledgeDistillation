# Module 8: Distilling Vision Models — Design Note

**Spec:** `docs/superpowers/specs/2026-11-19-kd-course-platform-design.md`

## Decisions

- **Scope chosen by the course owner: image classification** ("option A"), with a LoRA run. Vision-language models that write about images are not covered. The module was renamed from "Vision and multimodal" to "Distilling vision models" to match.
- **Purpose:** show that the method transfers. The lab calls the Module 3 `kd_loss` unchanged.
- **Data:** Oxford-IIIT Pet (`timm/oxford-iiit-pet`, CC BY-SA 4.0), 37 breeds. Scarce-label splits with seed 13: 148 gold training (4 per breed), 74 gold validation, about 3,450 pool, 1,000 test.
- **Teacher:** CLIP ViT-L/14, used zero-shot with the sentence "a photo of a {breed}, a type of pet". It gives a probability for every class in one pass. Its Hub page carries no licence tag; lesson 8.2 says so and the teacher id is a setting.
- **Students:** DeiT-tiny (5.7M weights, Apache 2.0), fully fine-tuned; ViT-base (86M, Apache 2.0) with LoRA.
- **Images are converted to tensors once** per student model and kept in memory.
- **Augmentation is a left-right mirror only**, which keeps saved teacher scores valid. Lesson 8.3 explains why stronger augmentation would need the teacher online.
- **LoRA on a classifier trains the new output layer in full** (`modules_to_save`). Attention layer names are looked up from the model, because they differ between library versions (found while testing: the installed version uses `q_proj`/`v_proj`, older ones `query`/`value`).
- **Results are saved in `results/vision/`**, apart from the banking results.

## Lessons

8.1 From text to pixels (patch explorer) · 8.2 A teacher that reads class names · 8.3 Distilling into a small vision model · 8.4 LoRA for vision models · 8.5 Lab

## Known limits

- Not run on a GPU. Teacher accuracy, timings and memory use are unmeasured.
- The LoRA run changes model size and training method together; lesson 8.5 says the comparison does not isolate LoRA.
- No lab or lesson on vision-language models, by the owner's choice.
