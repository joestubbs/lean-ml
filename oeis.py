"""
Module to generate Lean definitions for oeis sequences that 
have a corresponding Python function definition.

This file depends on a file containing OEIS sequences with the 
corresponding Python source code. 
"""

import json
import os 
from pantograph import Server
import llms



# API keys and URL for the language model and MCP server
LLM_API_KEY = os.environ.get("LLM_API_KEY", "e4de26f2-799e-42b8-a154-7229e6589758")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://tejas.tacc.utexas.edu/v1/b0ffa48b-2509-4bd8-a6d9-5bed9a66f902")
LLM_MODEL = os.environ.get("LLM_MODEL", "Meta-Llama-3.1-8B-Instruct")
LEAN_MCP_API_KEY = os.environ.get("LEAN_MCP_API_KEY", "ZXQgrJ8bb83aRc84KYLQkBPStz7iO-kwnzhoWverOpA")


def get_python_src():
    """
    Load the OEIS sequences that have Python source implementations.
    """
    with open(os.path.expanduser("~/oeis_python_results.json"), 'r') as f:
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
    return llms.compile_prompt(user_template=template, 
                               system_template=None, 
                               template_variables=["seq_desc", "python_src"],
                               seq_desc=seq_desc, 
                               python_src=python_src)


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
    return llms.compile_prompt(user_template=template, 
                               system_template=None, 
                               template_variables=["lean_src", "error_msg", "seq_desc", "python_src"],
                               lean_src=lean_src,
                               error_msg=error_msg,
                               seq_desc=seq_desc, 
                               python_src=python_src)

    
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
    for line in splits:
        if line.startswith("```lean"):
            continue
        elif line.startswith("```"):
            break
        elif line.startswith("import"):
            try:
                imports.append(line.split("import")[1].strip())
            except Exception as e:
                print(f"Got exception trying to parse import: {e}; skipping this line")
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


# Tests ----------------------

def test_create_lean():
    client = llms.get_client(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    seq_desc = "the number of distinct characteristic polynomials of n X n matrices with elements {0, 1}"
    python_src = """from itertools import product
    from sympy import Matrix
    def A272661(n): return len({tuple(Matrix(n,n,p).charpoly().as_list()) for p in product((0,1),repeat=n**2)}) if n else 1
    """
    lean_src = generate_lean_function(client=client, 
                                      seq_desc=seq_desc, 
                                      python_src=python_src)
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

    client = llms.get_client(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    seq_desc = "the number of distinct characteristic polynomials of n X n matrices with elements {0, 1}"
    python_src = """from itertools import product
    from sympy import Matrix
    def A272661(n): return len({tuple(Matrix(n,n,p).charpoly().as_list()) for p in product((0,1),repeat=n**2)}) if n else 1
    """
    
    idx = 0

    # create the initial lean source
    lean_src = test_create_lean()
    imports, clean_src = extract_and_remove_imports(lean_src=lean_src)
    
    # main loop 
    while idx < max_loops:
        print(f"Top of iteration {idx+1}")
        print(f"Got lean source: {lean_src}")
        if send_imports:
            server = get_lean_server(imports=imports)
        else:
            server = get_lean_server()
        # evaluate the Lean
        error, error_message = eval_lean(server=server, lean_src=clean_src)
        if not error:
            print(f"The Lean source code type checked! Final source: {lean_src}")
            print("Exiting")
            break
        else:
            print(f"There was an error with the Lean source. Error: {error_message}")
        # try to repair the Lean
        print("Attempting to repair the Lean..")
        lean_src = repair_lean_function(client=client, 
                                        seq_desc=seq_desc, 
                                        python_src=python_src, 
                                        lean_src=lean_src, 
                                        error_msg=error_message)
        imports, clean_src = extract_and_remove_imports(lean_src=lean_src)
        print(f"Got imports for repaired: {imports}")
        print(f"Got clean source for repaired: {clean_src}")
        idx += 1 


