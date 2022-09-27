"""
This script generates instances for the synthetic efficiency scenario by extracting a substring
from an input text file (e.g., a book) that has the right number of tokens. This script uses the
respective tokenizer service to ensure that this happens. A separate set of instances is
generated for each distinct tokenizer used.
"""


import os
from typing import Dict, List

from common.general import ensure_directory_exists, ensure_file_downloaded, parse_hocon, write
from common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from proxy.clients.client import Client
from proxy.clients.auto_client import AutoClient
from proxy.services.service import (
    CREDENTIALS_FILE,
    CACHE_DIR,
)
from benchmark.scenarios.synthetic_efficiency_scenario import NUM_INPUT_TOKENS

MAX_ITERS = 5


def _count_prompt_tokens(client: Client, prompt: str, tokenizer: str):
    request: TokenizationRequest = TokenizationRequest(text=prompt, tokenizer=tokenizer)
    result: TokenizationRequestResult = client.tokenize(request)
    return len(result.tokens)


def get_client(base_path: str = "prod_env"):
    credentials_path = os.path.join(base_path, CREDENTIALS_FILE)
    cache_path = os.path.join(base_path, CACHE_DIR)
    ensure_directory_exists(cache_path)
    if os.path.exists(credentials_path):
        with open(credentials_path) as f:
            credentials = parse_hocon(f.read())
    else:
        credentials = {}

    client = AutoClient(credentials, cache_path)

    return client


def tokenize_text(
    client: AutoClient, tokenizer: str, output_path: str = "synthetic_efficiency_instances", base_path: str = "prod_env"
):
    """Tokenizes each book using the requested tokenizer service."""
    sources = {
        "alice": "https://www.gutenberg.org/files/11/11-0.txt",
        "pride_and_prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "monte_cristo": "https://www.gutenberg.org/cache/epub/1184/pg1184.txt",
        "crime_and_punishment": "https://www.gutenberg.org/files/2554/2554-0.txt",
    }

    tokens: Dict = {}
    text_chunks: Dict = {}

    tokenizer_organization: str = tokenizer.split("/")[0]
    ai21_tokenizer: bool = tokenizer_organization == "ai21"

    # Extract tokens from book sources
    seen_tokens = set()
    for book, source_url in sources.items():
        data_path = os.path.join(output_path, f"{book}.txt")
        ensure_file_downloaded(
            source_url=source_url,
            target_path=data_path,
            unpack=False,
        )
        with open(data_path, "r") as f:
            raw_text = f.read()
        batch_size = 2048
        # Skip intro text
        text = raw_text.split(" ")[batch_size:]
        i = 0
        tokens[book] = []
        text_chunks[book] = []
        while i * batch_size < len(text):
            batch = " ".join(text[i * batch_size : (i + 1) * batch_size])
            while True:
                request: TokenizationRequest = TokenizationRequest(
                    text=batch, tokenizer=tokenizer, encode=(not ai21_tokenizer)
                )
                result: TokenizationRequestResult = client.tokenize(request)
                tokens_ = frozenset([token.value for token in result.tokens])
                if tokens_ not in seen_tokens:
                    seen_tokens.add(tokens_)
                    break
            tokens[book] += result.tokens
            if ai21_tokenizer:
                text_chunks[book] += [
                    result.text[token.text_range.start : token.text_range.end] for token in result.tokens
                ]
            i += 1
    return tokens, text_chunks


def generate_synthetic_efficiency_instances(
    tokens: Dict[str, List[int]],
    text_chunks: Dict[str, List[str]],
    client: Client,
    num_instances: int,
    num_prompt_tokens: int,
    tokenizer: str,
    output_path: str = "synthetic_efficiency_instances",
    base_path: str = "prod_env",
):
    """Generates the synthetic efficiency instances given the tokenized book sources."""
    tokenizer_organization: str = tokenizer.split("/")[0]
    ai21_tokenizer: bool = tokenizer_organization == "ai21"
    opt_tokenizer: bool = tokenizer_organization == "facebook"
    yandex_tokenizer: bool = tokenizer_organization == "Yandex"

    books = list(tokens.keys())
    prompts = []
    for i in range(num_instances // len(books)):
        for j in range(len(books)):
            finished = False
            attempt_num = 0
            orig_i = i
            # Select a new span of text to generate a prompt from
            while not finished:
                i = orig_i + attempt_num
                prompt: str = ""

                # Initialize
                if ai21_tokenizer:
                    per_instance_tokens = text_chunks[books[j]][i * num_prompt_tokens : (i + 1) * num_prompt_tokens]
                else:
                    per_instance_tokens = [
                        token.value for token in tokens[books[j]][i * num_prompt_tokens : (i + 1) * num_prompt_tokens]
                    ]

                # Iterate on this span of text until we get the right number of tokens
                num_iters = 0
                while num_iters < MAX_ITERS:
                    if ai21_tokenizer:
                        prompt = "".join(per_instance_tokens)
                    else:
                        decode_request: DecodeRequest = DecodeRequest(tokens=per_instance_tokens)
                        decode_result: DecodeRequestResult = client.decode(decode_request)
                        prompt = decode_result.text

                    if prompt == "":
                        num_generated_tokens = 0
                    else:
                        num_generated_tokens = _count_prompt_tokens(client, prompt, tokenizer)
                    if num_generated_tokens != num_prompt_tokens:
                        temp_num_tokens = num_generated_tokens
                        while temp_num_tokens < num_prompt_tokens:
                            if len(per_instance_tokens) == 0:
                                assert num_prompt_tokens == 1, num_prompt_tokens
                                if ai21_tokenizer:
                                    per_instance_tokens = text_chunks[books[j]][:2]
                                else:
                                    per_instance_tokens = [token.value for token in tokens[books[j]][:2]]
                            else:
                                per_instance_tokens.append(per_instance_tokens[-1])
                            temp_num_tokens += 1
                        while temp_num_tokens > num_prompt_tokens:
                            per_instance_tokens = per_instance_tokens[:-1]
                            temp_num_tokens -= 1
                    else:
                        finished = True
                        break
                    num_iters += 1
                if not finished:
                    print(
                        f"Requested {num_prompt_tokens}, got {num_generated_tokens} for "
                        f"book {books[j]}, instance #{orig_i}, tokenizer={tokenizer}, "
                        "trying again with a new span of text..."
                    )
                    attempt_num += 1
                    continue
                prompts.append(prompt)

    for i, prompt in enumerate(prompts):
        if opt_tokenizer:
            tokenizer = tokenizer.replace("facebook", "meta")
            tokenizer = tokenizer.replace("opt-66b", "opt")
        elif yandex_tokenizer:
            tokenizer = tokenizer.replace("Yandex", "yandex")
        name = f"num_prompt_tokens={num_prompt_tokens}," f"tokenizer={tokenizer.replace('/', '_')}," f"id={i}.txt"
        write(os.path.join(output_path, name), prompt)


if __name__ == "__main__":
    client = get_client()

    for tokenizer in [
        "huggingface/gpt2",
        "ai21/j1",
        "cohere/cohere",
        "bigscience/T0pp",
        "Yandex/yalm",
        "facebook/opt-66b",
        "bigscience/bloom",
        "google/t5-11b",
        "google/ul2",
        "EleutherAI/gpt-neox-20b",
        "EleutherAI/gpt-j-6B",
        "TsinghuaKEG/ice",
    ]:
        tokens, text_chunks = tokenize_text(tokenizer=tokenizer, client=client)
        for num_prompt_tokens in NUM_INPUT_TOKENS:
            generate_synthetic_efficiency_instances(
                tokens=tokens,
                text_chunks=text_chunks,
                client=client,
                num_instances=30,
                num_prompt_tokens=num_prompt_tokens,
                tokenizer=tokenizer,
            )
