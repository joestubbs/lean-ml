"""
Module to generate Lean definitions for oeis sequences that
have a corresponding Python function definition.

This file depends on a file containing OEIS sequences with the
corresponding Python source code.

"""

import json
import os
import timeit

from pantograph import Server
import llms


# API keys and URL for the language model and MCP server

# Public Sambanova project (Llama 3.1-8B) ----
# LLM_API_KEY="e4de26f2-799e-42b8-a154-7229e6589758"
# LLM_BASE_URL="https://tejas.tacc.utexas.edu/v1/b0ffa48b-2509-4bd8-a6d9-5bed9a66f902"
# LLM_MODEL="Meta-Llama-3.1-8B-Instruct"

# Niall's Sambanova Project (DeepSeek and Llama 3.3) ----
# LLM_API_KEY = "4bb30afe-fa9b-4dde-b69b-354c20fcc712"
# LLM_BASE_URL = "https://tejas.tacc.utexas.edu/v1/bf130648-8c40-4a34-a23c-da4f71e90073"
# LLM_MODEL = "DeepSeek-R1-Distill-Llama-70B"
# LLM_MODEL = "Meta-Llama-3.3-70B-Instruct"

# Llama 4
LLM_API_KEY = "f4fda34e-dca7-4297-9bbe-ecff05d8ce71"
LLM_BASE_URL = "https://tejas.tacc.utexas.edu/v1/5ca719f0-5bff-4d87-a84c-e09cdbe08d14"
LLM_MODEL = "Llama-4-Maverick-17B-128E-Instruct"

# not currently used --
LEAN_MCP_API_KEY = os.environ.get(
    "LEAN_MCP_API_KEY", "ZXQgrJ8bb83aRc84KYLQkBPStz7iO-kwnzhoWverOpA"
)

OEIS_RESULTS_FILE = os.environ.get(
    "OEIS_RESULTS_FILE", os.path.expanduser("~/oeis_python_results_more.json")
)

# Path to an output file where this script should write its results
OEIS_LEAN_OUTPUT_FILE = os.environ.get(
    "OEIS_LEAN_OUTPUT_FILE", os.path.expanduser("~/oeis_lean_source_llama-Jul-12.json")
)


# LLM Functions ---------------------------------------------


def get_python_src():
    """
    Load the OEIS sequences that have Python source implementations.
    """
    with open(OEIS_RESULTS_FILE, "r") as f:
        return json.load(f)


def get_lean_server(imports=["Init", "Mathlib"]):
    """
    Return an instance of the pantogram server, importing Mathlib
    into the namespace.
    """
    return Server(project_path=".", imports=imports)


def get_create_messages(seq_desc, python_src):
    """
    Returns system and user messages that can be used to ask for a Lean function
    corresponding to a Python function associated with a sequence.
    """

    template = """Here is the source code for a function in Python to compute {{seq_desc}}. 
    Python source code: {{python_src}}

    Write an equivalent function in the Lean programming language. Make
    the Lean function computable if possible, and minimize the number of imports
    required to define the function. Finally, return only the source code for 
    the Lean function so that the response can be entered directly into Lean.
    """
    return llms.compile_prompt(
        user_template=template,
        system_template=None,
        template_variables=["seq_desc", "python_src"],
        seq_desc=seq_desc,
        python_src=python_src,
    )


def get_repair_messages(seq_desc, python_src, lean_src, error_msg):
    """
    Returns system and user messages that can be used to ask for a Lean function to be
    repaired given that a previous source failed to compile with a
    given error message.
    """

    template = """Thank you but the Lean source code you generated contains an error.
    Here is the code you generated: {{lean_src}} 
    and here is the error reported by Lean: {{error_msg}}
    Here is the original Python function: {{python_src}}
    And the original sequence description: {{seq_desc}}.
    Please fix the error and return a Lean function that computes the same values
    as the Python function. Make
    the Lean function computable if possible, and minimize the number of imports
    required to define the function. Finally, return only the source code for 
    the Lean function so that the response can be entered directly into Lean.
    """
    return llms.compile_prompt(
        user_template=template,
        system_template=None,
        template_variables=["lean_src", "error_msg", "seq_desc", "python_src"],
        lean_src=lean_src,
        error_msg=error_msg,
        seq_desc=seq_desc,
        python_src=python_src,
    )


def generate_lean_function(client, seq_desc, python_src):
    """
    Generate a Lean function corresponding to a Python source function.
    This function invokes an external language model API.
    `client` should be an authenticated OpenAI client object.
    `seq_desc` should be a Python string describing the sequence.
    `python_src` should be the Python source code implementing the function, as a String.
    """

    # first, generate the prompt
    messages = get_create_messages(seq_desc, python_src)
    # send the message to the LLM
    response = llms.send_chat_message(client=client, model=LLM_MODEL, messages=messages)
    try:
        return response.choices[0].message.content
    except Exception as e:
        print(f"Got exception trying to parse the LLM response; error:{e}")
        raise e


def extract_and_remove_imports(lean_src):
    """
    Removes the import lines from `lean_src` and returns them as a list together
    with the modified source code.
    """
    splits = lean_src.split("\n")
    imports = []
    clean_source = ""
    # We assume the lean source code exists in a continuous block, starting with
    # a line that begins with ```lean or ```lean4
    # This variable tracks whether we have started reading lean code or not; we ignore
    # all lines prior.
    lean_started = False
    for line in splits:
        if line.startswith("```lean"):
            lean_started = True
            continue
        elif line.startswith("```"):
            break
        elif line.startswith("import"):
            try:
                imports.append(line.split("import")[1].strip())
            except Exception as e:
                print(f"Got exception trying to parse import: {e}; skipping this line")
                continue
        elif not lean_started:
            continue
        else:
            clean_source += line + "\n"
    return imports, clean_source


def repair_lean_function(client, seq_desc, python_src, lean_src, error_msg):
    """
    Ask LLM to repair a previous Lean function that contains errors.
    This function invokes an external language model API.
    """
    # first, generate the prompt
    messages = get_repair_messages(seq_desc, python_src, lean_src, error_msg)
    # send the message to the LLM
    response = llms.send_chat_message(client=client, model=LLM_MODEL, messages=messages)
    try:
        return response.choices[0].message.content
    except Exception as e:
        print(f"Got exception trying to parse the LLM response; error:{e}")
        raise e


def eval_lean(server, lean_src):
    """
    Use the Lean server to evaluate a Lean source code string, and return
    whether the code type checks, and if not, return the error string
    associated with the code.
    """
    error = False
    error_message = ""
    comp_units = server.load_sorry(lean_src)
    for unit in comp_units:
        for message in unit.messages:
            if "error" in message:
                error = True
                error_message = message
    return error, error_message


def generate_repair_lean_loop(python_src, seq_desc, max_loops=5, send_imports=False):
    """
    High-level function that takes the Python source code, `python_src`,
    implementing a sequence with a description, `seq_desc`, and tries to
    generate a Lean version using an LLM. It does this by asking the LLM to
    generate Lean code, calling pantograph to evaluate the generated Lean,
    checking for errors, and, if there are errors, asking the LLM to repair
    them. It does this loop for a `max_loops` iterations. If `send_imports` is True,
    this function will also send the LLM-generated imports to pantograph, but this
    is turned off by default, as there have been many issues observed with the
    LLM-generated Lean imports.

    """
    client = llms.get_client(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    idx = 0

    # create the initial lean source
    lean_src = generate_lean_function(
        client=client, seq_desc=seq_desc, python_src=python_src
    )
    # print(f"Got initial lean source: {lean_src}")
    imports, clean_src = extract_and_remove_imports(lean_src=lean_src)

    # main loop
    while idx < max_loops:
        print(f"Top of iteration {idx+1}")
        error = None
        error_message = None
        if send_imports:
            server = get_lean_server(imports=imports)
        else:
            server = get_lean_server()
        # evaluate the Lean
        try:
            error, error_message = eval_lean(server=server, lean_src=clean_src)
        except Exception as e:
            print(f"Got error trying to evaluate the Lean source code; e:{e}")
            error = True
            error_message = f"Could not check Lean code; full message: {e}"
        if not error:
            print(f"\n*****The Lean source code type checked!")
            # print(f"Final source: {lean_src}")
            # print("Exiting")
            break
        else:
            # print(f"\nThere was an error with the Lean source. \nError: {error_message}\n Attempting to repair the Lean..")
            pass
        # try to repair the Lean
        lean_src = repair_lean_function(
            client=client,
            seq_desc=seq_desc,
            python_src=python_src,
            lean_src=lean_src,
            error_msg=error_message,
        )
        # print(f"Got repaired lean source: {lean_src}")
        imports, clean_src = extract_and_remove_imports(lean_src=lean_src)
        # print(f"Got imports for repaired: {imports}")
        # print(f"Got clean source for repaired: {clean_src}")
        idx += 1

    # if we had an error, we tried a final repair, so let's evaluate the final
    # source before returning
    if error:
        try:
            error, error_message = eval_lean(server=server, lean_src=clean_src)
        except Exception as e:
            print(f"Got error trying to evaluate the Lean source code; e:{e}")
            error = True
            error_message = f"Could not check Lean code; full message: {e}"
        if not error:
            print(f"\n*****The Lean source code type checked!")
    return lean_src, imports, clean_src, error, error_message


def write_oeis_results_file(results):
    """
    Write the results of the OEIS processing to the output file.
    """
    print(f"Writing final results to: {OEIS_LEAN_OUTPUT_FILE}")
    with open(OEIS_LEAN_OUTPUT_FILE, "w+") as r:
        r.write(json.dumps(results))


def get_oeis_results_from_file():
    """
    Read the results of the OEIS processing from a previous run.
    """
    with open(OEIS_LEAN_OUTPUT_FILE, "r") as f:
        return json.load(f)


def process_oeis_sequences(start_from_prev=False, tot_seqs_limit=1):
    """
    Function to process a set of OEIS sequences from the OEIS_RESULTS_FILE and
    generate Lean source code for each function. This function will process the
    first `tot_seqs_limit` seqeunces; to process the entire file, call with:
        process_oeis_sequences(tot_seqs_limit=20000)
    (there are about 385,000 sequences in the OEIS but less than 20,000 have Python
    functions)
    """
    results = {}
    if start_from_prev:
        results = get_oeis_results_from_file()
    oeis_source = get_python_src()
    idx = 0
    errors = 0
    type_checked = 0
    start = timeit.default_timer()
    for seq_id, value in oeis_source.items():
        idx += 1
        if seq_id in results.keys():
            print(f"Skipping sequence {seq_id} as it was already in the results.")
            if not results[seq_id]["error"]:
                type_checked += 1
            continue
        print(f"Processing sequence {seq_id}, number {idx} of {tot_seqs_limit}")
        try:
            python_src = value["python_src"]
            seq_desc = value["desc"]
        except KeyError as e:
            errors += 1
            print(f"Got Key error for sequence {seq_id}; error: {e}; skipping")
            continue
        lean_src, imports, clean_src, error, error_msg = generate_repair_lean_loop(
            python_src=python_src, seq_desc=seq_desc
        )
        results[seq_id] = {
            "seq_desc": seq_desc,
            "lean_src": lean_src,
            "clean_src": clean_src,
            "imports": imports,
            "error": error,
            "error_msg": error_msg,
        }
        if not error:
            type_checked += 1
        cur = timeit.default_timer()
        tot_sec = cur - start
        print(f"Current run time: {tot_sec} seconds")
        write_oeis_results_file(results=results)
        if idx >= tot_seqs_limit:
            break
    # Report final results:
    print(f"Total sequences processed: {tot_seqs_limit}")
    print(f"Sequences that typed checked: {type_checked}")
    print(f"Total errors: {errors}")


# Main function
if __name__ == "__main__":
    process_oeis_sequences()


# Tests ----------------------


def test_create_lean():
    client = llms.get_client(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    seq_desc = "the number of distinct characteristic polynomials of n X n matrices with elements {0, 1}"
    python_src = """from itertools import product
    from sympy import Matrix
    def A272661(n): return len({tuple(Matrix(n,n,p).charpoly().as_list()) for p in product((0,1),repeat=n**2)}) if n else 1
    """
    lean_src = generate_lean_function(
        client=client, seq_desc=seq_desc, python_src=python_src
    )
    return lean_src


def test_create_and_eval_lean(send_imports=False):
    lean_src = test_create_lean()
    print(f"Got lean source: {lean_src}")
    imports, clean_src = extract_and_remove_imports(lean_src=lean_src)
    print(f"Got imports: {imports}")
    print(f"Got clean source: {clean_src}")
    if send_imports:
        server = get_lean_server(imports=imports)
    else:
        server = get_lean_server()
    error, error_message = eval_lean(server=server, lean_src=clean_src)
    if error:
        print(f"Lean source code had error: {error_message}")
    else:
        print("Lean source code type checked!")


def test_create_eval_repair_lean(send_imports=False, max_loops=5):

    seq_desc = "the number of distinct characteristic polynomials of n X n matrices with elements {0, 1}"
    python_src = """from itertools import product
    from sympy import Matrix
    def A272661(n): return len({tuple(Matrix(n,n,p).charpoly().as_list()) for p in product((0,1),repeat=n**2)}) if n else 1
    """
    return generate_repair_lean_loop(
        python_src=python_src,
        seq_desc=seq_desc,
        send_imports=send_imports,
        max_loops=max_loops,
    )
