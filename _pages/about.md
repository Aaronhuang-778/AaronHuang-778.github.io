---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

Hi, I am Wei Huang(黄炜)’s website! I am currently a Ph.D candidate in [Computer Vision and Machine Intelligence Lab (CVMI Lab) @ HKU](https://xjqi.github.io/cvmi.html) from September 2023, advised by [Prof.Xiaojuan Qi](https://xjqi.github.io/). Previously I obtained my bachelor's degree in computer science (Jun 2023) from Beihang University where I was advised by [Prof.Si Liu](https://colalab.net/people).I am also working closely with [Dr.Haotong Qin](https://htqin.github.io/) from ETH Zürich, [Dr.Yue Liao](https://liaoyue.net/) from The Chinese University of HongKong.

I’m currently conducting some research in efficient deep learning, including:
🚀 **Efficient AI**: The efficiency of the Large Language/Vision-Language Model and Diffusion Model (e.g. model quantization/binarization).
🔥 **Brain-mimic Computing**: Neuromorphic computing and hardware acceleration (e.g. spiking neural network-SNN).
⌚ **Edged AI**: Edged AI for wearable and digital health.

**I‘m actively seeking internship and visiting opportunities. If you have any opportunities available, I would greatly appreciate it if you could reach out to me. Thank you!**


# 🔥 News
- *2024.04*: &nbsp;🎉🎉 one co-author paper is accepted by ICCAD'24! 
- *2024.05*: &nbsp;Release *SliM-LLM*, a plug-and-play group-wise mixed-precision quantizaion framework for 2-bit LLMs. Please check our [paper](https://arxiv.org/abs/2405.14917), [code](https://github.com/Aaronhuang-778/SliM-LLM) and [huggingface](https://huggingface.co/AaronHuangWei)!
- *2024.04*: &nbsp;Release *An Empirical Study of LLaMA3 Quantization: From LLMs to MLLMs*, an emperical study on the performance of low-bit quantized LLM/MLLM based on LLaMA-3. Please check our [paper](https://arxiv.org/abs/2404.14047), [code](https://github.com/Macaronlin/LLaMA3-Quantization) and [huggingface](https://huggingface.co/LLMQ)!
- *2024.04*: &nbsp;🎉🎉 *BiLLM* is accepted by ICML'24! 
- *2024.02*: &nbsp;Release BiLLM*, the first post-training quantization work pushing the LLMs to nearly 1-bit. Please check our [paper](https://arxiv.org/abs/2402.04291) and [code](https://github.com/Aaronhuang-778/BiLLM)!
- *2023.09*: &nbsp;Start my Ph.D. in HKU.
- *2023.09*: &nbsp;Release *OHQ*, the on-chip hardware-aware mixed-precision quantization work. Please check our [paper](https://arxiv.org/abs/2402.04291)[https://arxiv.org/abs/2309.01945]!
- *2023.06*: &nbsp;Graduate from Beihang University. Thanks to my supervisors and all my friends in Beihang University.
- *2022.10*: &nbsp;Release *VLSNR*, the multi-modal news recommendation system work. Please check our [paper](https://arxiv.org/abs/2210.02946) and [code](https://github.com/Aaronhuang-778/V-MIND)!

# 💬 Invited Talks & Reporter

- *2024.05*: *BiLLM* was reported bt **IEEE Spectrum**. Please see the [link](https://spectrum.ieee.org/1-bit-llm). 
- *2024.05*: AI-Time online talk on *BiLLM*. Please see the [video](https://www.bilibili.com/video/BV1XM4m1z7RU/?share_source=copy_web&vd_source=c680cccdae8e0fd2e453769e2e789b78). 
- *2024.04*: Our emperical study *How Good Are Low-bit Quantized LLaMA3 Models? An Empirical Study* (new version: *[An Empirical Study of LLaMA3 Quantization: From LLMs to MLLMs](https://arxiv.org/abs/2404.14047)*) was reported by QbitAI (量子位). Please see the [link](https://m.thepaper.cn/newsDetail_forward_27189727).
  
# 📝 Publications 

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv</div><img src='https://github.com/Aaronhuang-778/SliM-LLM/blob/main/imgs/WX20240527-155305%402x.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

[**SliM-LLM: Salience-Driven Mixed-Precision Quantization for Large Language Models**](https://arxiv.org/abs/2405.14917) <img src='https://img.shields.io/github/stars/Aaronhuang-778/SliM-LLM.svg?style=social&label=Star' alt="sym" height="100%">

**Wei Huang**, Haotong Qin, Yangdong Liu, Yawei Li, Xianglong Liu, Luca Benini, Michele Magno, Xiaojuan Qi

- A novel scheme that observes and proves the structure-clustering of salient elements in LLMs weight matrix.
- The first group-wise mixed-precision quantization framework for LLMs.
- Serve as a plug-and-play approach to GPTQ/Omniquant/..., improving the inference-friendly method under low-bit quantization.

<div style="display: inline">
    <a href="https://arxiv.org/abs/2405.14917"> <strong>[paper]</strong></a>
    <a href="https://github.com/Aaronhuang-778/SliM-LLM"> <strong>[code]</strong></a>
    <a class="fakelink" onclick="$(this).siblings('.abstract').slideToggle()" ><strong>[abstract]</strong></a>
    <div class="abstract"  style="overflow: hidden; display: none;">  
        <p> Large language models (LLMs) achieve remarkable performance in natural language understanding but require substantial computation and memory resources. Post-training quantization (PTQ) is a powerful compression technique extensively investigated in LLMs. However, existing PTQ methods are still not ideal in terms of accuracy and efficiency, especially with below 4 bit-widths. Standard PTQ methods using group-wise quantization suffer difficulties in quantizing LLMs accurately to such low-bit, but advanced methods remaining high-precision weights element-wisely are hard to realize their theoretical hardware efficiency. This paper presents a Salience-Driven Mixed-Precision Quantization scheme for LLMs, namely SliM-LLM. The scheme exploits the salience distribution of weights to determine optimal bit-width and quantizers for accurate LLM quantization, while aligning bit-width partition to groups for compact memory usage and fast integer inference. Specifically, the proposed SliM-LLM mainly relies on two novel techniques: (1) Salience-Determined Bit Allocation utilizes the clustering characteristics of salience distribution to allocate the bit-widths of each group, increasing the accuracy of quantized LLMs and maintaining the inference efficiency; (2) Salience-Weighted Quantizer Calibration optimizes the parameters of the quantizer by considering the element-wise salience within the group, balancing the maintenance of salient information and minimization of errors. Comprehensive experiments show that SliM-LLM significantly improves the accuracy of LLMs at ultra-low bits, e.g., 2-bit LLaMA-7B achieves a 5.5-times memory-saving than original model on NVIDIA A800 GPUs, and 48% decrease of perplexity compared to the state-of-the-art gradient-free PTQ method. Moreover, SliM-LLM+, which is integrated from the extension of SliM-LLM with gradient-based quantizers, further reduces perplexity by 35.1%. </p>
    </div>
</div>

</div>
</div>


# 📖 Educations
- *2023.09 - (now)*, Ph.D. Student in Department of Electrical Electronic Engineering, The University of HongKong.
- *2019.09 - 2023.06*, B.Eng. in Computer Science, School of Computer Science and Engineering, Beihang University.
  

# 🗒️ Academic Services

- Conference: Neurips'2024, ICML'2024, ECCV'2024. 
- Program Committee member for Practical Deep Learning Workshop, IEEE CAI 2024.


# 🎖 Honors and Awards 
- *2019-2023(B.Eng.)*: 
Outstanding Graduate(2022), Beihang University (2023).
Outstanding Project of the 15th National College Student Innovation and Entrepreneurship Competition, China (2022).
Second-class of the Social Practice Scholarship, Beihang University (2022).
Third-class of the Innovation and Entreprenuership Scholarship, Beihang University (2021).
Second-class of the Subject Competition Scholarship, Beihang University (2022), 3rd Prize of the 32st “Feng Ru Cup” Competition (2022).
Second-class scholarship, Beihang University (2022).
3rd “Lan Qiao Cup” programming competation(Python), Beijing (2022).
Second-class of the Social Practice Scholarship, Beihang University (2021).
Second-class of the Subject Competition Scholarship, Beihang University (2021).
Outstanding Teaching Assistant, Beihang University (2021).
2nd Prize of the 31st “Feng Ru Cup” Competition (2020).
First-class scholarship, Beihang University (2020). 


# 💻 Internships & Teaching Services
- *2022.09 - 2023.01*, AI algorithm internship on model inference acceleration, [Enflame](https://www.linkedin.com/company/enflame/), China.
- *2022.08 - 2023.01*, TA for **Frontiers in Artificial Intelligence**, Beihang University.
- *2022.08 - 2023.01*, TA for **Computer Hardware Basics**, the head of TA team, Beihang University.
- *2021.08 - 2022.01*, TA for **Computer Hardware Basics**, the head of TA team, Beihang University.
- *2021.03 - 2021.06*, TA for **Discrete Mathematics**, the head of TA team, Beihang University.
