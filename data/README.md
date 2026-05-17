# Data

This repository uses two data sources in the DUCX paper workflow:

- **MIMIC-FairnessVQA**: provided by this DUCX paper release. The file in this repository, [mimic-fairnessVQA_example.jsonl](mimic-fairnessVQA_example.jsonl), is a small format example. The full release should place the complete MIMIC-FairnessVQA question file under `data/mimic/`.
- **ChestAgentBench / EuroRAD metadata**: provided by MedRAX. Download the MedRAX data files from https://github.com/bowang-lab/MedRAX/tree/main/data and place them under the paths described below.

This repository does not distribute MIMIC-CXR images, protected health information, model weights, or private experiment logs. Users must obtain source images and any restricted metadata from their official providers and follow the relevant access terms.

## Expected Layout

```text
data/
  chestagentbench/
    metadata.jsonl
  eurorad_metadata.json
  mimic/
    medrax_input_all_2000.jsonl
    mimic_sample_400.csv
  mimic-fairnessVQA_example.jsonl
```

Image files may live outside `data/`, but the paths listed in each question JSONL must resolve on the local machine.

## MIMIC-FairnessVQA

DUCX provides the MIMIC-FairnessVQA question data introduced in the paper. For local runs, place the complete file at:

```text
data/mimic/medrax_input_all_2000.jsonl
```

The case-level demographic CSV should be placed at:

```text
data/mimic/mimic_sample_400.csv
```

The CSV should include one identifier column, preferably one of:

- `case_id`
- `dicom_id`
- `id`

Demographic columns used by DUCX:

- `gender`
- `age` or `anchor_age`
- optionally `age_group`

Age is binarized at 60 for the paper setting: `young < 60` and `old >= 60`.

## ChestAgentBench / MedRAX Data

ChestAgentBench and the EuroRAD metadata are from MedRAX. Download the data from:

https://github.com/bowang-lab/MedRAX/tree/main/data

The DUCX scripts expect:

```text
data/chestagentbench/metadata.jsonl
data/eurorad_metadata.json
```

`analysis/fairness_posthoc.py` joins question rows to case metadata through `case_id`.

## Question JSONL Schema

Agent evaluation expects a JSONL file where each line is one multiple-choice chest X-ray question.

Required fields:

- `question_id`: stable unique identifier.
- `case_id`: case or study identifier used to join demographic metadata.
- `question`: question text with answer options.
- `answer`: gold answer letter, usually `A` through `F`.
- `images`: list of local image paths.

Optional fields:

- `full_question_id`
- `type`
- `categories`
- `sections`
- `explanation` or `explaination`
- `image_source_urls`

See [mimic-fairnessVQA_example.jsonl](mimic-fairnessVQA_example.jsonl) for an example schema.

## Image Paths

`launch_over_chexbench.py` passes image paths to the agent tools exactly as listed in the JSONL. The paths should be valid from the repository root or absolute paths on the local machine.
