import argparse
from typing import List, Tuple
import csv
import pickle
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd


class AtariHeadProcessor:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.meta = self._read_meta()

    def _read_meta(self):
        meta = pd.read_csv(self.dataset_dir / "meta_data.csv").dropna(axis=0)
        meta["TrialNumber"] = meta["TrialNumber"].astype(int)
        meta["NumberOfFrames"] = meta["NumberOfFrames"].astype(int)

        data = {
            int(path.name.split("_")[0]): path.name[: -len(".tar.bz2")]
            for path in self.dataset_dir.iterdir()
            if path.name.endswith(".tar.bz2")
        }
        meta["frames"] = meta["TrialNumber"].map(lambda tn: data[tn] + ".tar.bz2")
        meta["gazes"] = meta["TrialNumber"].map(lambda tn: data[tn] + ".txt")

        meta["Game"] = meta["Game"].str.replace("Mspacman", "MsPacman")

        return meta

    def read_gazes(self, trial_number: int):
        run_gazes = self.get_run(trial_number).gazes
        cols = [
            "frame_id",
            "episode_id",
            "score",
            "duration",
            "unclipped_reward",
            "action",
            "gaze_positions",
        ]

        records = []
        with (self.dataset_dir / run_gazes).open() as fp:
            csv_reader = csv.reader(fp)
            next(csv_reader)  # header
            for row in csv_reader:
                row = [(None if item == "null" else item) for item in row]
                values = row[: len(cols) - 1]
                gazes = row[len(cols) - 1 :]
                if gazes == [None]:
                    gazes = []
                else:
                    gazes = [float(v) for v in gazes]
                    gazes = list(zip(gazes[::2], gazes[1::2]))
                values.append(gazes)
                d = OrderedDict(zip(cols, values))
                for field in ["duration", "unclipped_reward", "action"]:
                    if d[field] is not None:
                        d[field] = int(d[field])
                records.append(d)

        df_gazes = pd.DataFrame.from_records(records)

        df_gazes["gaze_num"] = df_gazes["gaze_positions"].map(len)

        df_gazes["run_name"] = df_gazes["frame_id"].map(lambda s: s[: s.rindex("_")])
        df_gazes["frame_idx"] = df_gazes["frame_id"].map(
            lambda s: int(s[s.rindex("_") + 1 :])
        )

        return df_gazes

    def get_run(self, trial_number: int):
        matches = self.meta[self.meta["TrialNumber"] == trial_number]
        assert len(matches) == 1
        return matches.iloc[0]

    def game_trials(self, game: str):
        return self.meta[self.meta["Game"] == game]["TrialNumber"]


if __name__ == "__main__":
    import tarfile
    import tqdm

    dataset_dir = Path("/scr/aliang80/varibad_jax/atari_head/")
    ah = AtariHeadProcessor(dataset_dir)

    # Some code to extract the tar files
    print(ah.meta)
    print(ah.meta["frames"][0])
    frames = ah.meta["frames"].to_list()

    print(len(ah.meta["frames"].to_list()))

    for frame_bz2 in tqdm.tqdm(frames, total=len(frames), desc="Extracting frames"):
        img_path = dataset_dir / frame_bz2

        name = frame_bz2.replace(".tar.bz2", "")
        outdir = img_path.parent / f"extracted_{name}"

        if not outdir.exists():
            with tarfile.open(img_path, "r") as tar:
                # print(tar.getnames())
                # print(len(tar.getmembers()))
                # for member in tar.getmembers():
                #     f = tar.extractfile(member)
                #     if f is not None:
                #         content = f.read()

                #     if ".png" in member.name:
                #         tar.extract(member, outdir)
                tar.extractall(path=outdir)

    print(ah.game_trials("Asterix"))
    for trial_number in ah.game_trials("Asterix"):
        print(trial_number)
        df_gazes = ah.read_gazes(trial_number)
        # print(df_gazes["frame_idx"].to_list())
        print(len(df_gazes["frame_idx"].to_list()))
        # print(df_gazes["action"].to_list())
        print(len(df_gazes["action"].to_list()))
        print(df_gazes.head())
        break
