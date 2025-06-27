import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline

nf4_config =BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_use_double_quant= True,
    bnb_4bit_compute_dtype= torch.bfloat16
)

def get_hf_llm(model_name: str ="Qwen/Qwen2-1.5B-Instruct",
               max_new_tokens = 1024,
               **kwargs
               ):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = nf4_config,
        low_cpu_mem_usage= True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer= tokenizer,
        max_new_tokens = max_new_tokens,
        pad_token_id = tokenizer.eos_token_id,
        device_map="auto"
    )

    if 'max_new_token' in kwargs:
        kwargs['max_new_tokens'] = kwargs.pop('max_new_token')

    llm = HuggingFacePipeline(
        pipeline= model_pipeline,
        model_kwargs= kwargs
    )
    return llm

