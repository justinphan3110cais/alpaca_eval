import fire
import json
import csv
from vllm import LLM, SamplingParams
import os
import random
import ray
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import ray
from vllm.distributed import destroy_model_parallel
from chat_templates import get_template


def main(model_path, output_path="tmp.json", split="test", tensor_parallel_size=1, chat_template=None):
    # model
    print(f"Loading model from: {model_path}")
    if tensor_parallel_size > 1:
        _init_ray(8)
    model = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    
    if chat_template is None or chat_template == "none":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        template = [{'role': 'user', 'content': '{instruction}'}]
        prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
        if tokenizer.bos_token:
            prompt = prompt.replace(tokenizer.bos_token, "")
        chat_template = prompt
    else:
        chat_template = get_template(chat_template)['prompt']

    print("chat_template=", chat_template)
    with open("./alpaca_eval_questions.json") as file:
        data = json.load(file)
    inputs = [chat_template.format(instruction=t['instruction']) for t in data]
    outputs = model.generate(inputs, sampling_params)
    outputs = [o.outputs[0].text.strip() for o in outputs]
    for d, o in zip(data, outputs):
        d['output'] = o
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

def _init_ray(num_cpus=16, reinit=False):
    from transformers.dynamic_module_utils import init_hf_modules

    # check if ray already started
    if ('RAY_ADDRESS' in os.environ or ray.is_initialized()) and not reinit:
        return
    # Start RAY
    # config different ports for ray head and ray workers to avoid conflict when running multiple jobs on one machine/cluster
    # docs: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-networking-caveats
    num_cpus = min([os.cpu_count(), num_cpus])

    os.environ['RAY_DEDUP_LOGS'] = '0'
    RAY_PORT = random.randint(0, 999) + 6000 # Random port in 6xxx zone
    RAY_MIN_PORT = random.randint(0, 489) * 100 + 10002 
    RAY_MAX_PORT = RAY_MIN_PORT + 99 # Random port ranges zone
    
    os.environ['RAY_ADDRESS'] = f"127.0.0.1:{RAY_PORT}"
    ray_start_command = f"ray start --head --num-cpus={num_cpus} --port {RAY_PORT} --min-worker-port={RAY_MIN_PORT} --max-worker-port={RAY_MAX_PORT} --disable-usage-stats --include-dashboard=False"
    
    print(f"Starting Ray with command: {ray_start_command}")
    os.system(ray_start_command)

    init_hf_modules()  # Needed to avoid import error: https://github.com/vllm-project/vllm/pull/871
    ray.init(ignore_reinit_error=True)

if __name__ == "__main__":
    fire.Fire(main)
