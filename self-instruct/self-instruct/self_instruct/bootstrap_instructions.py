import os
import json
import random
import re
import string
import argparse
from multiprocessing import Pool
from functools import partial
from collections import Counter, defaultdict



random.seed(42)


def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def group_seed_tasks(seed_tasks):
    """Group by explicit task labels; IDs and row order have no semantics."""
    groups = defaultdict(list)
    for seed in seed_tasks:
        task = seed.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Seed {seed.get('id')} needs a nonempty task field")
        groups[task].append(seed)
    if not groups:
        raise ValueError("No seed examples available")
    return dict(groups)


class BalancedSeedSampler:
    """Prioritize accepted-generation deficits; break ties by exposure and RNG."""

    def __init__(self, seed_groups, generated_records=(), rng=None):
        if not seed_groups or any(not seeds for seeds in seed_groups.values()):
            raise ValueError("Every task must have at least one seed")
        self.seed_groups = seed_groups
        self.rng = rng if rng is not None else random.Random(42)
        self.generated_counts = Counter({task: 0 for task in seed_groups})
        self.seed_counts = Counter()
        self.pending_counts = Counter()
        for record in generated_records:
            self.record_accepted(record)

    def record_accepted(self, record):
        task = record.get("task")
        if task not in self.seed_groups:
            raise ValueError(
                f"Generated record has missing/unknown task {task!r}; "
                "use a fresh batch directory or annotate existing records before resuming"
            )
        self.generated_counts[task] += 1

    def start_batch(self):
        # Reservations spread simultaneous requests, but never count as accepted.
        self.pending_counts.clear()

    def next_task(self):
        tasks = list(self.seed_groups)
        self.rng.shuffle(tasks)
        task = min(tasks, key=lambda t: self.generated_counts[t] + self.pending_counts[t])
        self.pending_counts[task] += 1
        return task

    def sample_seeds(self, task, n=8):
        """Return full records, one per distinct task, including the request task."""
        if task not in self.seed_groups:
            raise ValueError(f"Unknown task: {task}")
        if not 1 <= n <= len(self.seed_groups):
            raise ValueError("n must be between 1 and the number of seed task groups")
        others = [t for t in self.seed_groups if t != task]
        self.rng.shuffle(others)
        others.sort(key=lambda t: (self.generated_counts[t], self.seed_counts[t]))
        selected = [task] + others[:n - 1]
        seeds = [self.rng.choice(self.seed_groups[t]) for t in selected]
        self.seed_counts.update(selected)
        self.rng.shuffle(seeds)
        return seeds


def build_batch(sampler, generated_records, count, num_prompt_instructions, classification=False):
    """Build legacy numbered prompts with explicit, tracked generation targets."""
    if count < 1 or num_prompt_instructions < 1:
        raise ValueError("Request batch size and prompt instruction count must be positive")
    sampler.start_batch()
    requests = []
    for _ in range(count):
        task = sampler.next_task()
        same_task = [r for r in generated_records if r["task"] == task]
        machine = sampler.rng.sample(same_task, min(2, len(same_task), num_prompt_instructions - 1))
        seed_count = min(num_prompt_instructions - len(machine), len(sampler.seed_groups))
        seeds = sampler.sample_seeds(task, n=seed_count)
        instructions = [r["instruction"] for r in seeds + machine]
        sampler.rng.shuffle(instructions)
        prompt = (
            f"Generate new healthcare insurance instructions only for task: {task}.\n"
            "Examples below span several tasks for diversity; new instructions must "
            "perform the requested task only. Return a numbered list of instructions, "
            "without answers, scenarios, task labels, or commentary.\n"
            + encode_prompt(instructions, classification=classification)
        )
        requests.append({"task": task, "seed_ids": [r["id"] for r in seeds], "prompt": prompt})
    return requests


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)



def post_process_gpt3_response(response):
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print one balanced request batch without loading model/API dependencies or generating data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    seed_groups = group_seed_tasks(seed_tasks)
    print(f"Loaded {len(seed_tasks)} seed examples across {len(seed_groups)} tasks")
    generated_records = []

    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                generated_records.append(instruction_info)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    sampler = BalancedSeedSampler(seed_groups, generated_records)
    if args.request_batch_size < 1 or args.num_prompt_instructions < 1:
        raise ValueError("Request batch size and prompt instruction count must be positive")
    if args.dry_run:
        requests = build_batch(sampler, generated_records, args.request_batch_size,
                               args.num_prompt_instructions, args.use_clf_seed_tasks_only)
        print(json.dumps({"accepted_counts": dict(sampler.generated_counts), "requests": requests}, indent=2))
        raise SystemExit(0)

    # Only actual generation needs the original backend dependencies.
    import tqdm
    import numpy as np
    from rouge_score import rouge_scorer
    from gpt3_api import make_requests as make_gpt3_requests

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            requests = build_batch(
                sampler, generated_records, args.request_batch_size,
                args.num_prompt_instructions, args.use_clf_seed_tasks_only,
            )
            batch_inputs = [request["prompt"] for request in requests]
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1,
                api_key=args.api_key,
                organization=args.organization,
            )
            instructions = []
            all_metadata = []
            if len(results) != len(requests):
                raise ValueError("Backend returned a different number of results than requests")
            all_requests = []
            for result, request in zip(results, requests):
                new_instructions = post_process_gpt3_response(result["response"])
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)
                all_requests += [request] * len(new_instructions)

            for inst, metadata, request in zip(instructions, all_metadata, all_requests):
                if len(machine_instructions) >= args.num_instructions_to_generate:
                    break
                with Pool(4) as p:
                    rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                if max(rouge_scores) > 0.7:
                    continue
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                        all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                    }
                machine_instructions.append(inst)
                record = {
                    "instruction": inst,
                    "task": request["task"],
                    "seed_ids": request["seed_ids"],
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,
                    "request_idx": request_idx
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                generated_records.append(record)
                sampler.record_accepted(record)
                progress_bar.update(1)
            request_idx += 1
