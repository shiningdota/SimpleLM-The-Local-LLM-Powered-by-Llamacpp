# SimpleLM Chatbot and RAG system with Streamlit UI + Llamacpp
![SimpleLM Screenshot](https://github.com/user-attachments/assets/31efc84e-105a-4030-a11d-33d39de04af7)

## Overview
I make simple ui to use a Chatbot and also had RAG (Retrieval Augemted Generation) feature. This project using Llamacpp+Vulkan as the main backend for LLM, Streamlit for the UI, Combining the SentenceTransformers for embedding, and also support the OpenAI API

## Install
1. Install some package
```bash
pip install streamlit openai sentence-transformers nltk numpy PyPDF2 python-docx
```

2. To install this project you have 2 options:
- You can download the package from release page and extract them
- Or using 'git clone':

```bash
git clone https://github.com/shiningdota/SimpleLM-The-Local-LLM-Powered-by-Llamacpp.git
```

## How to use
1. You need LLM models, and you can download it from website like Huggingface. Example about the models:
- `Llama-3.2-1B-Instruct(gguf)`
[Unsloth](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF)
- `Llama-3.2-3B-Instruct(gguf)`
[Unsloth](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF)
- `Llama-3.1-8B-Instruct(gguf)`[Unsloth](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
2. Open the `launch-llm.bat`, and you choose if you want to use vulkan with GPU or CPU to run
3. Load the LLM Model
```
NOTE/REMINDER: 
Compiling vulkan take some minutes (maybe around 1-3minutes) just for the first time launch. So for the next launch, you no longer to wait and ready to use.
```

## About model quantization
There is a lot of option for the quantization. The smaller model like 1B or 3B you can use the `Q8_0` option, and the bigger model like 7B, 8B, 12B++ you can start from `Q4_K_M` option. 
`IMPORTANT: Do not use lower quant for small model (example: Q4_K_M on Llama-3.2-1B or Llama-3.2-3B). Because the results will be pretty bad, it's not recommended to use.`

You can read more about quantization below:
```
https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/quantize.cpp#L19
https://mlabonne.github.io/blog/posts/Quantize_Llama_2_models_using_ggml.html
https://github.com/unslothai/unsloth/wiki#gguf-quantization-options

{
    "f32"     : "Not recommended. Retains 100% accuracy, but super slow and memory hungry.",
    "f16"     : "Fastest conversion + retains 100% accuracy. Slow and memory hungry.",
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q4_k"    : "alias for q4_k_m",
    "q5_k"    : "alias for q5_k_m",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
    "iq2_xxs" : "2.06 bpw quantization",
    "iq2_xs"  : "2.31 bpw quantization",
    "iq3_xxs" : "3.06 bpw quantization",
    "q3_k_xs" : "3-bit extra small quantization",
}
```

## Changelog
- V1.0
```
- First release!!!
- Llamacpp version B4539
- Support Mistral model, Llama model, the new powerfull model Deepseek R1 Distill models, and many more
- Implemented Local LLM powered by Llamacpp + OpenAI API
- Implemented Local RAG Document Chatbot with combination of Sentence Transformer
- Customizeable LLM Settings like Prompt for LLM, Temperature, Max tokens (currently only support up to 4096), etc. 
```

## Resource
- [Llamacpp](https://github.com/ggerganov/llama.cpp)
- [Streamlit](https://streamlit.io/)
- [Unsloth](https://unsloth.ai/)
