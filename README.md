## 📖 Introduction
This project allows you to use images or just prompts that convert to the Ghibli drawing style locally.

## 📄 References
<p>
Base Model: 
<a href="https://huggingface.co/black-forest-labs/FLUX.1-dev" target="_blank">black-forest-labs/FLUX.1-dev</a>
</p>
<p>
Base Lora:
<a href="https://huggingface.co/openfree/flux-chatgpt-ghibli-lora" target="_blank">https://huggingface.co/openfree/flux-chatgpt-ghibli-lora</a> 
</p>
<p>
Space Reference:
<a href="https://huggingface.co/spaces/seawolf2357/Ghibli-Multilingual-Text-rendering" target="_blank">https://huggingface.co/spaces/seawolf2357/Ghibli-Multilingual-Text-rendering</a>
</p>

## ⚡️ Quick Start

### 🔧 Requirements and Installation

Install the requirements
```bash
conda create -n local_ghibli python=3.10 -y
conda activate local_ghibli
pip install -r requirements.txt
```

### 🌟 Gradio Run

```bash
python app.py
```

**For low vmemory usage**, please pass the `--load-in-4bit` or `--load-in-8bit` args. The peak memory usage will be something about 16GB or 20GB.
Enough to run on the RTX 3090.

```bash
python app.py --load-in-4bit
```

### 📌 Examples

### Image / Prompt to Image

<img src="https://raw.githubusercontent.com/Llocus/local_ghibli/refs/heads/main/examples/example1.png" width="1200" />  
</br></br>
<img src="https://raw.githubusercontent.com/Llocus/local_ghibli/refs/heads/main/examples/example2.png" width="1200" />  
</br></br>
<img src="https://raw.githubusercontent.com/Llocus/local_ghibli/refs/heads/main/examples/example3.png" width="1200" />  

### Prompt to Image

<img src="https://raw.githubusercontent.com/Llocus/local_ghibli/refs/heads/main/examples/example4.png" width="1200" />  
</br></br>
<img src="https://raw.githubusercontent.com/Llocus/local_ghibli/refs/heads/main/examples/example5.png" width="1200" />  
</br></br>
<img src="https://raw.githubusercontent.com/Llocus/local_ghibli/refs/heads/main/examples/example6.png" width="1200" />  