import json
import pandas as pd
import os


ALL_OEIS_RESULTS_FILE = os.environ.get(
    "ALL_OEIS_RESULTS_FILE", os.path.expanduser("~/oeis_results_more.json")
)

OEIS_PYTHON_RESULTS_FILE = os.environ.get(
    "OEIS_RESULTS_FILE", os.path.expanduser("~/oeis_python_results_more.json")
)

OEIS_EXEC_OUTPUT_FILE = os.environ.get(
    # "OEIS_EXEC_OUTPUT_FILE", os.path.expanduser("~/oeis_exec_output.json")
    "OEIS_EXEC_OUTPUT_FILE",
    os.path.expanduser("~/oeis_exec_output-11-07.json"),
)


def get_all_seq_data():
    with open(ALL_OEIS_RESULTS_FILE, "r") as f:
        data = json.load(f)
        # return pd.DataFrame(data)
        return pd.DataFrame.from_dict(data, orient="index")


def get_all_python_seq_data():
    with open(OEIS_PYTHON_RESULTS_FILE, "r") as f:
        data = json.load(f)
        # return pd.DataFrame(data)
        return pd.DataFrame.from_dict(data, orient="index")


def get_all_exec_data():
    with open(OEIS_EXEC_OUTPUT_FILE, "r") as f:
        data = json.load(f)
        return pd.DataFrame.from_dict(data, orient="index")


def get_empty_python_seq_ids():
    result = []
    # 304 total rows
    empty_rows = all_exec[
        all_exec["python_result"] == "No matching values; Python list was empty"
    ]
    empty_seq_ids = [row[0] for row in empty_rows.iterrows()]
    seqs = get_all_seq_data()
    empty_seqs = []
    for s in empty_seq_ids:
        try:
            empty_seqs.append(seqs.loc[s])
        except KeyError:
            pass
    return empty_seq_ids, empty_seqs


def get_all_python_errors():
    return all_exec[all_exec["python_result"].str.contains("Error")]


def print_m_to_n_python_source(m=0, n=10):
    """
    Pretty print rows m to n of the `empty_seqs` series.
    """
    for idx in range(m, n):
        current_id = empty_seqs[idx].name
        print(f"************  {current_id}  **********")
        print(f"Result: {all_exec.loc[current_id]['python_result']}")
        print(f"Execution time: {all_exec.loc[current_id]['time_for_sequence']}")
        print(empty_seqs[idx].python_src)


all_exec = get_all_exec_data()

all_python_seqs = get_all_python_seq_data()

all_seqs = get_all_seq_data()

empty_seq_ids, empty_seqs = get_empty_python_seq_ids()
