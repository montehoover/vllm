from datasets import Dataset
import datasets
from transformers import AutoTokenizer
import argparse
import torch


def get_article_content(args, article):
    title = article['title']
    paragraphs = article["text"].split("\n\n")
    target_paragraph = paragraphs[0]
    content = []
    content.append(f"Title: {title}")
    content.append(f"URL: {article['url']}")
    content.append(f"Article:")
    paragraphs = article["text"].split("\n\n")
    content.append("\n\n".join(paragraphs[:args.num_paragraphs]))
    content = "\n".join(content)
    metadata = {"title": {title}, "target_paragraph": target_paragraph}
    return content, metadata


def create_long_context_example(args, dataset_iter, tokenizer):
    tokens_added = 0
    long_context_window = []
    all_metadata = []
    while tokens_added < args.num_tokens:
        article = next(dataset_iter)
        paragraphs = article["text"].split("\n\n")
        target_paragraph = paragraphs[0]
        # Only use an article if the first paragraph is long enough to be a good label.
        if len(target_paragraph.split()) > 35:
            new_content, metadata = get_article_content(args, article)
            long_context_window.append(new_content)
            all_metadata.append(metadata)
            tokens_added += len(tokenizer.tokenize(new_content))
    # Randomly select an article to be the label
    i = torch.randint(0, len(all_metadata), (1,)).item()
    title = all_metadata[i]["title"]
    target = all_metadata[i]["target_paragraph"]
    intro = "Remember the following Wikipedia articles:\n"
    instruction = f"Instruction: Write the first paragraph of the Wikipedia article on \"{title}\"."
    long_context_window = "\n\n\n".join(long_context_window)
    if args.zeroshot:
        prompt = instruction
    else:
        prompt = "\n".join([intro, long_context_window, instruction])
        prefix_prompt = "\n".join([intro, long_context_window])
        suffix_prompt = instruction
    token_count = len(tokenizer.tokenize(prompt))
    prefix_token_count = len(tokenizer.tokenize(prefix_prompt)) if not args.zeroshot else 0
    return {"prompt": prompt, "target": target, "token_count": token_count, "prefix_token_count": prefix_token_count}


def create_dataset(args):
    # torch.manual_seed(42)
    dataset = datasets.load_dataset('wikipedia', "20220301.en", split='train', streaming=True)
    dataset = dataset.shuffle(seed=42)
    # turn into an iterator so we can pop off articles one at a time without having to know ahead of time how many we need to reach our target token count.
    dataset_iter = iter(dataset)  
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
    new_dataset = []
    for i in range(args.num_examples):
        new_dataset.append(create_long_context_example(args, dataset_iter, tokenizer))
    new_dataset = Dataset.from_list(new_dataset)

    # Print information about the token counts
    token_counts = [len(tokenizer.tokenize(prompt)) for prompt in new_dataset["prompt"]]
    print(f"Max token count: {max(token_counts)}")
    print(f"Median token count: {torch.tensor(token_counts).median().item()}")

    # Save dataset to disk
    if args.name is not None:
        name = f"longcontext_wiki_{args.name}"
    elif args.zeroshot:
        name = "longcontext_wiki_zeroshot"
    else:
        name = f"longcontext_wiki_{args.num_tokens}"
    dataset_path = f"datasets/{name}.json"
    new_dataset.to_json(dataset_path, orient="records", lines=False, indent=True)
    return dataset_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", default=4000, type=int, help="Number of tokens in each long context window.")
    parser.add_argument("--num_examples", default=100, type=int, help="Number of these long context examples to include in the dataset.")
    parser.add_argument("--num_paragraphs", default=2, type=int, help="Number of paragraphs to include from each article. Set really large (10,000?) to just get the whole article.")
    parser.add_argument("--zeroshot", action="store_true", help="Make a dataset with no articles text, just the instruction.")
    parser.add_argument("--name", default=None, type=str, help="Name of the dataset file.")
    parser.add_argument("--tokenizer_model", default="meta-llama/Llama-2-7b-chat-hf", type=str, help="Huggingface model name to use for tokenizer to get token count.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(args)
