from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import filter_code, run_eval, fix_indents
import os
import torch

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = ""


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]
    inputs = tokenizer(input_batch, return_token_type_ids=False, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )

    return [filter_code(fix_indents(completion)) for completion in batch_completions]

def test_model(name="Linksoul-llama2-7b",model_path="/home/Linksoul-llama2-7b"):
    num_samples_per_task = 10
    out_path = f"results/{name}/eval.jsonl"
    os.makedirs(f"results/{name}", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )

    model = torch.compile(
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        .eval()
    )

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        True,
    )

if __name__ == "__main__":
    # model_path="/home/Linksoul-llama2-7b"
    # name="Linksoul-llama2-7b"
    # test_model(name,model_path)

    # model_path="codellama/CodeLlama-7b-hf"
    # name="CodeLlama-7b-hf"
    # test_model(name,model_path)

    # model_path="codellama/CodeLlama-7b-Python-hf"
    # name="CodeLlama-7b-Python-hf"
    # test_model(name,model_path)

    model_path="/mnt/SFT_store/LLM/llama-65b-hf"
    name="llama-65b-hf"
    test_model(name,model_path)