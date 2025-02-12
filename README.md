# LoRA

Computer Science Project that aims to demonstrate the steps necessary to fine tune an LLM using LoRA.


## Sources


### Datasets

|Dataset | Source |
|---- | ---- |
| SmolTalk v1.0| https://huggingface.co/datasets/HuggingFaceTB/smoltalk |
| | |

## Dependencies


- Python 3.11
- Pytorch 2.6.0+ 
- transformers
- accelerate
- datasets
- torchao
- huggingface_hub
- argparse

---

Fine tuning libraries:
- peft
- torchtune (only for Llama and Mistral)
- unsloth (only on linux or windows systems with CUDA >= 12)

For unsloth on windows use this:
`pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-2.1.0-cp311-cp311-win_amd64.whl` nvm