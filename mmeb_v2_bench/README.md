---
license: apache-2.0
task_categories:
- text-retrieval
- text-classification
- token-classification
language:
- en
tags:
- multimodal
pretty_name: MMEB-V2
size_categories:
- 1M<n<10M
viewer: false
---

# MMEB-V2 (Massive Multimodal Embedding Benchmark)

[**Website**](https://tiger-ai-lab.github.io/VLM2Vec/) |[**Github**](https://github.com/TIGER-AI-Lab/VLM2Vec) | [**🏆Leaderboard**](https://huggingface.co/spaces/TIGER-Lab/MMEB) | [**📖MMEB-V2/VLM2Vec-V2 Paper**](https://arxiv.org/abs/2507.04590) | | [**📖MMEB-V1/VLM2Vec-V1 Paper**](https://arxiv.org/abs/2410.05160) |


## Introduction

Building upon on our original [**MMEB**](https://arxiv.org/abs/2410.05160), **MMEB-V2** expands the evaluation scope to include five new tasks: four video-based tasks — Video Retrieval, Moment Retrieval, Video Classification, and Video Question Answering — and one task focused on visual documents, Visual Document Retrieval. This comprehensive suite enables robust evaluation of multimodal embedding models across static, temporal, and structured visual data settings.

**This Hugging Face repository contains the image and video frames used in MMEB-V2, which need to be downloaded in advance.**


## Guide to All MMEB-V2 Data
**Please review this section carefully for all MMEB-V2–related data.**

- **Image/Video Frames** – Available in this repository.  
- **Test File** – Loaded during evaluation from Hugging Face automatically. A comprehensive list of HF paths can be found [here](https://github.com/TIGER-AI-Lab/VLM2Vec/blob/main/src/data/dataset_hf_path.py).  
- **Raw Video Files** – In most cases, the video frames are all you need for MMEB evaluation. However, we also provide the raw video files [here](https://huggingface.co/datasets/TIGER-Lab/MMEB_Raw_Video) in case they are needed for specific use cases. Since these files are very large, please download and use them only if necessary.  

For this local benchmark sandbox, HF annotation parquet files are persisted under:

```text
mmeb_v2_bench/cache/annotations
```

You can override this with:

```bash
--annotation-cache-dir /your/persistent/path
```


## 🚀 What's New
- **\[2025.07\]** Release [tech report](https://arxiv.org/abs/2507.04590).
- **\[2025.05\]** Initial release of MMEB-V2/VLM2Vec-V2.


## Dataset Overview

We present an overview of the MMEB-V2 dataset below:
<img width="900" alt="abs" src="overview.png">


## Dataset Structure

The directory structure of this Hugging Face repository is shown below. 
For video tasks, we provide sampled frames in this repo. For image tasks, we provide the raw images.
Files from each meta-task are zipped together, resulting in six files. For example, ``video_cls.tar.gz`` contains the sampled frames for the video classification task.

```

→ video-tasks/
├── frames/
│   ├── video_cls.tar.gz
│   ├── video_qa.tar.gz
│   ├── video_ret.tar.gz
│   └── video_mret.tar.gz

→ image-tasks/
├── mmeb_v1.tar.gz
└── visdoc.tar.gz

```

After downloading and unzipping these files locally, you can organize them as shown below. (You may choose to use ``Git LFS`` or ``wget`` for downloading.)
Then, simply specify the correct file path in the configuration file used by your code.

```

→ MMEB
├── video-tasks/
│   └── frames/
│       ├── video_cls/
│       │   ├── UCF101/
│       │   │   └── video_1/              # video ID
│       │   │       ├── frame1.png        # frame from video_1
│       │   │       ├── frame2.png
│       │   │       └── ...
│       │   ├── HMDB51/
│       │   ├── Breakfast/
│       │   └── ...                       # other datasets from video classification category
│       ├── video_qa/
│       │   └── ...                       # video QA datasets
│       ├── video_ret/
│       │   └── ...                       # video retrieval datasets
│       └── video_mret/
│           └── ...                       # moment retrieval datasets
├── image-tasks/
│   ├── mmeb_v1/
│   │   ├── OK-VQA/
│   │   │   ├── image1.png
│   │   │   ├── image2.png
│   │   │   └── ...
│   │   ├── ImageNet-1K/
│   │   └── ...                           # other datasets from MMEB-V1 category
│   └── visdoc/
│       └── ...                           # visual document retrieval datasets


```
