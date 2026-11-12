# Sources, data boundaries, and code provenance

## Research context

The original study records Teresa Ristow and Ivan Hernandez as authors. The public companion summarizes that research and reuses five pure analysis functions from source revision `b2432201221d617dcff4d96a43e2ea74dadc924f`.

- `safe_divide`, `wilson_interval`, `confusion_metrics`, and `cohens_h` come from the research metrics module.
- `paired_bootstrap_accuracy_difference` comes from the research publication-comparison module.

The selected function bodies are preserved unchanged. Imports and package organization are reduced for this standalone companion. The input-validation wrapper, synthetic generator, example scripts, tests, and documentation are companion additions prepared with AI assistance. [Source hashes](code_provenance.json) record the extraction without distributing the private source inventory or linking to private Drive files.

This repository shares an analysis companion, not the original manuscript or full research workspace. It does not assert individual contribution roles beyond the existing authorship record, and it does not claim that all code was manually written.

## Benchmark and model sources

- [FairFace](https://github.com/joojs/fairface): dataset and source annotations. Its dataset page lists CC BY 4.0.
- [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ): dataset context and the non-commercial research agreement, including redistribution restrictions.
- [CLIP model](https://huggingface.co/sentence-transformers/clip-ViT-L-14): the fixed image/text model used in the research.
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B): the separately seeded prompt generator used in the research.

Dataset annotations do not establish personal identity, biological categories, or affect. The study evaluates differential model performance against those labels.

## What is distributed

The committed numeric example is generated from invented categorical outcomes. It contains no images, image identifiers, source annotations, transcripts, person-level predictions, private prompts, execution bundles, credentials, or model weights. The original benchmark output tables are not distributed. The qualitative findings are separated from the synthetic example throughout the documentation.

Access to any original dataset or model must follow its publisher's terms independently. The availability of this companion does not grant rights to those materials. No new blanket software or dataset license is asserted here; original attribution and third-party terms remain distinct.
