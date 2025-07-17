"""
Script to use the DeepSeek Prover V2 7B model to translate Python to Lean.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)


def get_sequence_():
    """ """
    seq_desc = """ """
    python_src = """ """
    return seq_desc, python_src


def get_sequence_A001222():
    """
    A001222
    """
    seq_desc = """Number of prime divisors of n counted with multiplicity (also called big omega of n, bigomega(n) or Omega(n))."""
    python_src = "from sympy import primeomega\ndef a(n): return primeomega(n)\nprint([a(n) for n in range(1, 112)])"
    return seq_desc, python_src


def get_sequence_A000142():
    """
    A000142
    """
    seq_desc = """Factorial numbers: n! = 1*2*3*4*...*n (order of symmetric group S_n, number of permutations of n letters)"""
    python_src = """for i in range(1, 1000):\n    y = i\n    for j in range(1, i):\n       y *= i - j\n    print(y, "\n")"""
    return seq_desc, python_src


def get_sequence_A001221():
    """
    A001221
    """
    seq_desc = """Number of distinct primes dividing n (also called omega(n))"""
    python_src = """from sympy.ntheory import primefactors\nprint([len(primefactors(n)) for n in range(1, 1001)])"""
    return seq_desc, python_src


def get_sequence_A000720():
    """
    A000720
    """
    seq_desc = """pi(n), the number of primes <= n. Sometimes called PrimePi(n) to distinguish it from the number 3.14159..."""
    python_src = """from sympy.ntheory import primefactors\nprint([len(primefactors(n)) for n in range(1, 1001)])"""
    return seq_desc, python_src


def get_sequence_():
    """ """
    seq_desc = """ """
    python_src = """ """
    return seq_desc, python_src


def get_sequence():
    """
    Return hard-coded sequence data for now. Update this when running for a new sequence.
    """
    # return get_sequence_A001222()
    # return get_sequence_A000142()
    return get_sequence_A000720()
    # return get_sequence_A001221()


def get_examples():
    """
    Returns hard-coded examples, for now.
    """
    python_1 = "def fib(n):\n    x, y = 0, 1\n    for _ in range(n):\n        yield x\n        x, y = y, x+y"
    lean_1 = "def Fibonacci2 (n : ℕ) : ℕ := Id.run do\n  let mut x := 0\n  let mut x_prev := 0\n  let mut y := 1\n  for _ in [0:n] do\n    x_prev := x\n    x := y\n    y := x_prev + y\n  pure x"

    python_2 = "def tri2(n):\n    yield 0\n    x, y = 1, 1\n    for _ in range(n):\n        yield x\n        x, y = x + y + 1, y + 1"
    lean_2 = "def Triangular2 (n : ℕ) : ℕ := Id.run do\n  let mut result : ℕ := 0\n  let mut x := 1\n  let mut y := 1\n  for _ in [1:(n+1)] do\n    result := x\n    x := x + y + 1\n    y := y + 1\n  result"

    python_3 = "A000108 = [1]\nfor n in range(1000):\n    A000108.append(A000108[-1]*(4*n+2)//(n+2))"
    lean_3 = "def Step : ℕ × ℕ → ℕ × ℕ :=\n  fun ⟨i, x⟩ => ⟨i + 1, x * (4 * i + 2) / (i + 2)⟩\n\ndef Catalan₂ (n : ℕ) : ℕ :=\n  Nat.iterate Step n (0, 1) |>.snd"
    return python_1, lean_1, python_2, lean_2, python_3, lean_3


def get_user_msg(
    seq_desc,
    python_src,
    python_ex1,
    lean_ex_1,
    python_ex2,
    lean_ex_2,
    python_ex_3,
    lean_ex_3,
):
    return f"""Here is the source code for a function in Python to compute {seq_desc}. 
    Python source code: {python_src}

    Write an equivalent function in the Lean programming language. Make
    the Lean function computable if possible, and minimize the number of imports
    required to define the function. Finally, return only the source code for 
    the Lean function so that the response can be entered directly into Lean.

    For example, given the Python function {python_ex1}, you should return the Lean
    function {lean_ex_1}.

    Similarly, given the Python function {python_ex2}, you should return the Lean
    function {lean_ex_2}.

    Similarly, given the Python function {python_ex_3}, you should return the Lean 
    function {lean_ex_3}.
    """


def run_model():
    start = time.time()
    python_1, lean_1, python_2, lean_2, python_3, lean_3 = get_examples()
    seq_desc, python_src = get_sequence()  # update above to run for a new sequence.
    user_msg = get_user_msg(
        seq_desc, python_src, python_1, lean_1, python_2, lean_2, python_3, lean_3
    )
    chat = [
        {
            "role": "system",
            "content": "You are a helpful Lean coding assistant. Do not use chain of thought. Only reply with the Lean code",
        },
        {"role": "user", "content": user_msg},
    ]
    inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=8192)
    decoded_output = tokenizer.batch_decode(outputs)
    parts = decoded_output[0].split("<｜Assistant｜>")
    if len(parts) > 1:
        print(f"\nReply:\n{parts[1]}")
        print(f"\n\n*****\n\nFull reply:{decoded_output}")
    else:
        print(f"\nDid not find the expected Assistant prompt; here is the full reply:")
        print(decoded_output)
    print(f"\nTotal runtime: {time.time() - start}")


if __name__ == "__main__":
    run_model()
