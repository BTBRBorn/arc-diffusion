import json
from pathlib import Path
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_file", type=str)
    parser.add_argument("--solutions_file", type=str, default="")
    parser.add_argument("--target_dir", type=str)

    args = parser.parse_args()

    source_file, target_dir, solutions_file = (
        Path(args.source_file),
        Path(args.target_dir),
        Path(args.solutions_file),
    )

    if target_dir.exists():
        shutil.rmtree(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)
    else:
        target_dir.mkdir(exist_ok=True, parents=True)

    with open(source_file, "r") as fhandle:
        source_json = json.load(fhandle)

    #Attach solutions to the source if exists 
    if args.solutions_file:
        with open(solutions_file, "r") as fhandle:
            solutions_json = json.load(fhandle)

        for key, task in source_json.items():
            for idx, example in enumerate(task['test']):
                example['output'] = solutions_json[key][idx]

    for key in source_json.keys():
        file_name = key + ".json"
        file_path = target_dir / file_name
        with open(file_path, "w") as fhandle:
            json.dump(source_json[key], fhandle)
