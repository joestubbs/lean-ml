# Some simple examples of using the PyPantogram library
# See the docs: https://centaur.stanford.edu/PyPantograph/setup.html

from pantograph import Server


def get_server(imports=["Init"]):
    """
    Return an instance of the pantogram server, optionally importing a set of Lean packages
    into the namespace. Note that one should likely always import Init, so if you are setting
    the imports parameter, you should include "Init".

    For example, to load a server with Mathlib: 
        server = get_server(imports=["Init", "Mathlib"])
    """
    # this should point to a git project with a Lean installation (and any associated libraries,
    # such as Mathlib, already installed)
    return Server(project_path=".", imports=imports)


def evaluate_lean(server, lean_source):
    """
    Evaluate a block of Lean source code, `lean_source` with `server`. Note that imports 
    should have been set when instantiating `server`.  
    
    Returns a list of `CompilationUnit` objects; 
    cf., https://centaur.stanford.edu/PyPantograph/api-data.html#pantograph.data.CompilationUnit
    """
    return server.load_sorry(lean_source)


def lean_source_ex1():
    """
    An example Lean code generated by an LLM. This code block contains an error in the first rw step:
    'did not find instance of the pattern in the target expression'
    """
    s = """
    example (a b c : ℝ) : a * (b * c) = b * (a * c) := by
    -- We can use the `rw` tactic along with the commutativity of multiplication (`mul_comm`).
    rw [mul_comm a b] -- This changes the goal to `b * a * c = b * (a * c)` [1].
    -- Multiplication in Lean associates to the left, so `b * a * c` is interpreted as `(b * a) * c`.
    -- We can use the associativity of multiplication (`mul_assoc`) to regroup.
    rw [mul_assoc b a c] -- This changes the goal to `(b * a) * c = b * (a * c)` [1].
    -- Now we can use the commutativity again on `a * c`.
    rw [mul_comm a c] -- This changes the goal to `(b * a) * c = b * (c * a)`.
    -- Finally, we use associativity in reverse (`← mul_assoc`).
    rw [← mul_assoc b c a] -- This changes the goal to `b * (c * a) = b * (c * a)`, which is true by reflexivity.
    """
    return s 


def test_1():
    # create the server with Mathlib 
    server = get_server(imports=["Init", "Mathlib"])

    # get the Lean code
    code = lean_source_ex1()

    # evaluate 
    comp_units = evaluate_lean(server=server, lean_source=code)

    # iterate over the units returned and print message
    for u in comp_units: 
        for m in u.messages:
            print(m)

