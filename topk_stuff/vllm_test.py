from vllm import LLM, SamplingParams
import argparse
import datasets
import torch

from create_dataset import create_dataset


def main(args):
    print(f"\nCreating dataset of {args.num_examples} examples with {args.num_tokens} tokens per example.")
    args_obj = argparse.Namespace(
        num_tokens=args.num_tokens,
        num_examples=args.num_examples,
        num_paragraphs=args.num_paragraphs,
        zeroshot=args.zeroshot,
        name=args.custom_name,
        tokenizer_model=args.model,
    )
    dataset_path = create_dataset(args_obj)
    raw_data = datasets.load_dataset("json", data_files={"eval": dataset_path})["eval"]

    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
    prompts = raw_data["prompt"]
    sampling_params = SamplingParams(temperature=0, min_tokens=20, max_tokens=200)

    llm = LLM(model=args.model,
              max_model_len=36000,
            #   gpu_memory_utilization=0.97,
            #   quantization="bitsandbytes",
            #   load_format="bitsandbytes",
            #   dtype=torch.float16,
              )

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gradientai/Llama-3-8B-Instruct-262k", type=str, choices=[
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        "meta-llama/Llama-2-7b-chat-hf", 
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        "gradientai/Llama-3-8B-Instruct-262k",
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        ])    
    parser.add_argument("--num_tokens", default=1000, type=int, help="Number of tokens in each long context window. Ignored if --dataset is provided.")
    parser.add_argument("--num_examples", default=1, type=int, help="Number of these long context examples to include in the created dataset.")
    parser.add_argument("--num_paragraphs", default=2, type=int, help="Number of paragraphs to include from each article. Set really large (10,000?) to just get the whole article.")
    parser.add_argument("--zeroshot", action="store_true", help="Make a dataset with no articles text, just the instruction.")
    parser.add_argument("--custom_name", default=None, type=str, help="Optional custom name for the results file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)