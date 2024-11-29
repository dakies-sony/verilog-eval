from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union, Dict, Tuple, Optional
import itertools
import re

import numpy as np
import tqdm

from verilog_eval.data import read_problems, stream_jsonl, write_jsonl
from verilog_eval.execution import check_correctness, clean_up_simulation

## Added by Daniel
def clean_code_snippet(input_text):
    # Remove any leading backticks and text before the first triple backtick
    cleaned_text = re.sub(r'^.*?```', '', input_text, count=1, flags=re.DOTALL)
    return cleaned_text

def extract_verilog_module(code):

    code = clean_code_snippet(code)
    # Define the regex patterns to find the desired positions
    top_module_pattern = re.compile(r'\bmodule\s+top_module\b')
    module_pattern = re.compile(r'\bmodule\s+(\w+)\s*\(')
    endmodule_pattern = re.compile(r'\bendmodule\b')

    top_module_match = top_module_pattern.search(code)
    # Find the position of "module top_module"
    if not top_module_match:
        module_match = module_pattern.search(code)
        if not module_match:
            start_pos = 0
        else:
            start_pos = module_match.start()  # Start of the substring "module top_module"
            tmp_code = code[start_pos :]
            semicolon_index = tmp_code.find(";")
            if semicolon_index != -1 :
                start_pos = start_pos + semicolon_index + 1
    else:
        start_pos = top_module_match.start()  # Start of the substring "module top_module"
        tmp_code = code[start_pos :]
        semicolon_index = tmp_code.find(";")
        if semicolon_index != -1 :
            start_pos = start_pos + semicolon_index + 1

    # Find the position of the last "endmodule"
    end_pos = -1
    for match in endmodule_pattern.finditer(code):
        end_pos = match.end()

    if end_pos == -1:
        return code  # If "endmodule" is not found, return original string

    # Extract the relevant portion of the code
    extracted_code = code[start_pos:end_pos]
    return extracted_code
#### End of Daniel's code


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def contain_passing_completion(
    problem: Dict,
    completions: List[str],
    n_workers: int = 4,
    timeout: float = 30.0,
    unit_test_length: Optional[int] = None,
    clean_up: bool = True,
) -> Tuple[bool, str]:

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        
        futures = []
        
        for idx, completion in enumerate(completions):
            args = (problem, completion, timeout, idx, unit_test_length)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            
        for future in as_completed(futures):
            result = future.result()
            if result["passed"]:
                return True, completions[result["completion_id"]]
            
    if clean_up:
        clean_up_simulation()
            
    return False, ""
            
def evaluate_functional_correctness(
    sample_file: str,
    problem_file: str,
    k: List[int] = None,
    n_workers: int = 4,
    timeout: float = 30.0,
    unit_test: bool = False,
    clean_up: bool = True,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    if k is None:
        k = [1, 10, 100]
    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            # completion = extract_verilog_module(sample["completion"]) # Daniel's code
            completion = sample["completion"]
            if unit_test:
                args = (problems[task_id], completion, timeout, completion_id[task_id], 100)
            else:
                args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))
    
    if clean_up:
        clean_up_simulation()

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k
