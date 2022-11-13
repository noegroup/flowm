# pointing the training scripts to corresponding datasets
import os

__all__ = ["load_raw_dataset"]


records = {
    "ala2": {
        "file_path": "./ala2_raw_data.npz",
        "entry_mapping": {
            "coords": "coords",
            "forces": "aaFs",
        },
        "shuffle_before": False, # since the dataset was 4 concatenated independent trjectories.
    },
    "cln-des": {
        "file_path": "./DESRES_CLN025.npz",
        "entry_mapping": {
            "coords": "coords",
        },
        "shuffle_before": True,
    },
}

def fetch_raw_dataset(protein="ala2", mode="flow"):
    """
    An interface to load the dataset transformed from all-atom simulations.
    Currently supporting the following entries (more can be added via `records` above):
    - ala2 (capped alanine, with forces)
    - cln-local (home-brew chignolin dataset with forces)
    - cln-des (chignolin dataset without forces from DESRES)
    - trpcage-des
    - bba-des
    - villin-des
    Argument `mode` can be:
    - "flow": for loading coordinates and generated gaussian noise for flow training
    - "cgnet": for loading coordinates and forces for conventional CGnet training
    """
    if protein not in records:
        raise ValueError(f"Unsupported dataset: {protein}.\n"
                         f"Following proteins are supported by the \n"
                         f"Please check the `routine\source_files.py` for details.")
    file_path = records[protein]["file_path"]
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Cannot find data source at `{file_path}`." 
                                 "Please check whether the dataset has been acquired.")
    entry_mapping = records[protein]["entry_mapping"]
    assert "coords" in entry_mapping, (f"Invalid records for {protein}. "
                                        "Please check `source_files.py`.")
    entries = [entry_mapping["coords"]]
    if mode == "flow":
        entries.append("gen_Gaussian_2d") # for the augmented channels
    elif mode == "cgnet":
        assert "forces" in entry_mapping, f"Record for {protein} does not contain forces."
        entries.append(entry_mapping["forces"])
    shuffle_before = records[protein].get("shuffle_before", True)
    return {"protein_data": file_path, 
            "entry_order": entries, 
            "shuffle_before_split": shuffle_before}


