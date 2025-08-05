<p align="left">
        中文</a>&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>
<br>
<div align="center">
<img src="figs/logo.jpg" width="300px">

**OceanGPT (沧渊): 沧渊海洋基础大模型**

<p align="center">
    <a href="https://github.com/zjunlp/OceanGPT">项目</a> •
    <a href="https://arxiv.org/abs/2310.02031">论文</a> •
    <a href="https://huggingface.co/collections/zjunlp/oceangpt-664cc106358fdd9f09aa5157">模型</a> •
    <a href="http://oceangpt.zjukg.cn/">网站</a> •
    <a href="#概述">概述</a> •
    <a href="#快速开始">快速开始</a> •
    <a href="#引用">引用</a>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/PRs-Welcome-red)  <a href='https://hyper.ai/datasets/32992'><img src='https://img.shields.io/badge/Dataset-HyperAI超神经-pink'></a> 


</div>

## Table of Contents

- <a href="#最新动态">最新动态</a>
- <a href="#概述">概述</a>
- <a href="#快速开始">快速开始</a>
- <a href="#与我们的Gradio演示对话"> 🤗与我们的Gradio演示对话</a>
- <a href="#推理">推理</a>
- <a href="#模型">模型</a>
- <a href="#使用llama.cpp, ollama, vLLM进行高效推理">使用llama.cpp, ollama, vLLM进行高效推理</a>
- <a href="#引用">引用</a>

## 🔔最新动态
- **2024-07-04，我们发布了OceanGPT-Basic-14B/2B以及更新版本的OceanGPT-Basic-7B。**
- **2024-06-04，OceanGPT 被ACL 2024接收。🎉🎉**
- **2023-10-04，我们发布了论文"OceanGPT: A Large Language Model for Ocean Science Tasks "并基于LLaMA2发布了OceanGPT-Basic-7B。**
- **2023-05-01，我们启动了OceanGPT (沧渊) 项目。**
---

## 🌟概述

这是OceanGPT (沧渊) 项目，旨在为海洋科学任务构建大语言模型。

- ❗**免责声明：本项目纯属学术探索，并非产品应用（本项目仅为学术探索并非产品应用）。请注意，由于大型语言模型的固有局限性，可能会出现幻觉等问题。**

<div align="center">
<img src="figs/overview.png" width="60%">
<img src="figs/vedio.gif" width="60%">
</div>


## ⏩快速开始

```
conda create -n py3.11 python=3.11
conda activate py3.11
pip install -r requirements.txt
```

### 下载模型
#### 从HuggingFace下载
```shell
git lfs install
git clone https://huggingface.co/zjunlp/OceanGPT-14B-v0.1
```
或
```
huggingface-cli download --resume-download zjunlp/OceanGPT-14B-v0.1 --local-dir OceanGPT-14B-v0.1 --local-dir-use-symlinks False
```
#### 从WiseModel下载
```shell
git lfs install
git clone https://www.wisemodel.cn/zjunlp/OceanGPT-14B-v0.1.git
```
#### 从ModelScope下载
```shell
git lfs install
git clone https://www.modelscope.cn/ZJUNLP/OceanGPT-14B-v0.1.git
```

### 推理
#### 使用HuggingFace进行推理
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"
path = 'YOUR-MODEL-PATH'

model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(path)

prompt = "Which is the largest ocean in the world?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
#### 使用vllm进行推理
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

path = 'YOUR-MODEL-PATH'

tokenizer = AutoTokenizer.from_pretrained(path)

prompt = "Which is the largest ocean in the world?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

sampling_params = SamplingParams(temperature=0.8, top_k=50)
llm = LLM(model=path)

response = llm.generate(text, sampling_params)
```

## 🤗与我们的Gradio演示对话

### 在线演示 <!-- omit in toc -->

我们为用户提供了可通过网络访问的交互式Gradio演示

### 本地WebUI演示
You can easily deploy the interactive interface locally using the code we provide.

```python
python app.py
```
在浏览器中打开 `https://localhost:7860/` 并享受与OceanGPT的互动。

## 📌推理

### 模型

| 模型名称        | HuggingFace                                                          | WiseModel                                                                 | ModelScope                                                                |
|-------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| OceanGPT-Basic-14B (based on Qwen) | <a href="https://huggingface.co/zjunlp/OceanGPT-14B-v0.1" target="_blank">14B</a> | <a href="https://wisemodel.cn/models/zjunlp/OceanGPT-14B-v0.1" target="_blank">14B</a> | <a href="https://modelscope.cn/models/ZJUNLP/OceanGPT-14B-v0.1" target="_blank">14B</a> |
| OceanGPT-Basic-7B (based on Qwen) | <a href="https://huggingface.co/zjunlp/OceanGPT-7b-v0.2" target="_blank">7B</a>   | <a href="https://wisemodel.cn/models/zjunlp/OceanGPT-7b-v0.2" target="_blank">7B</a>   | <a href="https://modelscope.cn/models/ZJUNLP/OceanGPT-7b-v0.2" target="_blank">7B</a>   |
| OceanGPT-Basic-2B (based on MiniCPM) | <a href="https://huggingface.co/zjunlp/OceanGPT-2B-v0.1" target="_blank">2B</a>   | <a href="https://wisemodel.cn/models/zjunlp/OceanGPT-2b-v0.1" target="_blank">2B</a>   | <a href="https://modelscope.cn/models/ZJUNLP/OceanGPT-2B-v0.1" target="_blank">2B</a>   |
| OceanGPT-Omni-7B  | 即将发布                                                                    | 即将发布                                                                         | 即将发布                                                                          |
| OceanGPT-Coder-7B  | 即将发布                                                                    | 即将发布                                                                         | 即将发布                                                                          |
---

### 使用llama.cpp、ollama、vLLM进行高效推理

<details> 
<summary>llama.cpp现在正式支持基于Qwen2.5-hf转换为gguf的模型。点击展开查看。</summary>

从huggingface下载OceanGPT PyTorch模型到“OceanGPT”文件夹。

克隆llama.cpp并编译：
```shell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
make llama-cli
```

然后将PyTorch模型转换为gguf文件：
```shell
python convert-hf-to-gguf.py OceanGPT --outfile OceanGPT.gguf
```

运行模型：
```shell
./llama-cli -m OceanGPT.gguf \
    -co -cnv -p "Your prompt" \
    -fa -ngl 80 -n 512
```
  </details>

<details> 
<summary>ollama现在正式支持基于Qwen2.5的模型。点击展开查看。</summary>

创建一个名为`Modelfile`的文件：
```shell
FROM ./OceanGPT.gguf
TEMPLATE "[INST] {{ .Prompt }} [/INST]"
```

在Ollama中创建模型：
```shell
ollama create example -f Modelfile
```

运行模型：
```shell
ollama run example "What is your favourite condiment?"
```
  </details>

<details>
<summary> vLLM现在正式支持基于Qwen2.5-VL和Qwen2.5的模型。点击展开查看。</summary>

1. 下载 vLLM(>=0.7.3):
```shell
pip install vllm
```

2. 运行示例:
* [MLLM](https://docs.vllm.ai/en/latest/getting_started/examples/vision_language.html) 
* [LLM](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) 
  </details>


## 🌻致谢

OceanGPT (沧渊) 基于开源大语言模型训练，包括[Qwen](https://huggingface.co/Qwen), [MiniCPM](https://huggingface.co/collections/openbmb/minicpm-2b-65d48bf958302b9fd25b698f), [LLaMA](https://huggingface.co/meta-llama)。感谢他们的杰出贡献！

## 局限性

- 模型可能存在幻觉问题。
- 我们未对身份信息进行优化，模型可能会生成类似于Qwen/MiniCPM/LLaMA/GPT系列模型的身份信息。
- 模型输出受提示词影响，可能导致多次尝试结果不一致。
- 模型需要包含特定模拟器代码指令进行训练才能具备模拟具身智能能力（模拟器受版权限制，暂无法公开），其当前能力非常有限。

### 🚩引用

如果您在工作中使用了OceanGPT，请引用以下论文。

```bibtex
@article{bi2024oceangpt,
  title={OceanGPT: A Large Language Model for Ocean Science Tasks},
  author={Bi, Zhen and Zhang, Ningyu and Xue, Yida and Ou, Yixin and Ji, Daxiong and Zheng, Guozhou and Chen, Huajun},
  journal={arXiv preprint arXiv:2310.02031},
  year={2024}
}

```

---
