# Study design and findings

The study asks whether averaging varied text prompts improves zero-shot image classification and reduces variation in performance across dataset-annotated groups.

## Comparisons

| Condition | Design |
| --- | --- |
| Single | One fixed conventional prompt for each label |
| Human | Five manuscript seed prompts for each label |
| BLOOM | Original saved prompt lists, preserved as historical artifacts |
| Qwen3 | Five separately seeded prompt sets, each with 100 generated prompts per label |

The fixed image encoder is `sentence-transformers/clip-ViT-L-14`, revision `1b4b2e899178706d1b9905460ac21de1e0ba86a5`. The prompt generator is `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, with non-thinking generation and seeds 1701–1705. The comparison averages text embeddings within a label and chooses the label with the highest cosine similarity to the image embedding.

FairFace supplies the age-label and gender-label tasks. The age contrast is limited to the source annotations `20–29` and `50–59`. CelebAMask-HQ supplies the smiling-label task. Group comparisons use inherited annotations, not independently verified identity or biological categories. A smiling label does not measure a person's underlying affect.

## Completed research and interpretation

Saved run artifacts document eight conditions, 102,624 task-image cases, and five Qwen prompt-set replications. The research comparison uses 10,000 paired bootstrap replications within task-label strata. This companion was prepared from the completed saved run; it does not repeat model inference.

The findings are mixed. Every Qwen set improved smiling-label accuracy against the single, human, and BLOOM baselines. Age-label results varied by generated set; BLOOM was strongest. All Qwen sets fell below the single and BLOOM baselines on the gender-label task. Subgroup gaps were not consistently smallest for Qwen.

Image cases are paired across conditions. Repeated prompt sets measure a different source of variability from image sampling, so the study reports the five sets separately rather than pooling them as independent image observations or selecting a favorable seed.

## Limits

- The results apply to the fixed benchmarks, label definitions, model revisions, and prompts used here.
- Many task and subgroup contrasts are examined. Intervals and directional patterns do not establish a universal improvement claim.
- Five generated sets give a limited view of prompt-generation variability.
- The original prompt audit flagged problematic lines; those prompts were retained rather than removed after inspecting outcomes.
- Dataset labels are imperfect measurement choices. Model performance against them does not establish identity, validity for workplace decisions, or employment adverse impact.
- Independent review and manuscript preparation remain separate from this public companion. No journal publication or new held-out experiment is claimed.

The original outputs, manuscript, prompts, and source inventory remain in the preserved research workspace. This companion shares the design and qualitative findings, selected analysis code, and a distinct synthetic demonstration.
