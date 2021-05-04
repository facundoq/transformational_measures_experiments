from __future__ import annotations
import abc
from pathlib import Path
import sys
from datetime import datetime
import texttable
import os
import argparse,argcomplete
from typing import List,Dict


class Options:
    def __init__(self, show_list: bool, force: bool):
        self.show_list = show_list
        self.force = force

class Experiment(abc.ABC):

    def __init__(self, base_folderpath:Path):
        self.base_folderpath = base_folderpath
        self.folderpath= base_folderpath / self.id()
        self.folderpath.mkdir(exist_ok=True, parents=True)
        with open(self.folderpath / "description.txt", "w") as f:
            f.write(self.description())

    def id(self):
        return self.__class__.__name__


    def __call__(self, force=False,  *args, **kwargs):
        stars = "*" * 15
        strf_format = "%Y/%m/%d %H:%M:%S"
        dt_started = datetime.now()
        dt_started_string = dt_started.strftime(strf_format)
        if not self.has_finished() or force:
            self.mark_as_unfinished()
            print(f"[{dt_started_string}] {stars} Running experiment {self.id()}  {stars}")
            self.run()

            # time elapsed and finished
            dt_finished = datetime.now()
            dt_finished_string = dt_finished.strftime(strf_format)
            elapsed = dt_finished - dt_started
            print(f"[{dt_finished_string}] {stars} Finished experiment {self.id()}  ({elapsed} elapsed) {stars}")
            self.mark_as_finished()
        else:
            print(f"[{dt_started_string}] {stars}Experiment {self.id()} already finished, skipping. {stars}")

    def print_date(self, message):
        strf_format = "%Y/%m/%d %H:%M:%S"
        dt = datetime.now()
        dt_string = dt.strftime(strf_format)
        message = f"[{dt_string}] *** {message}"
        print(message)

    def has_finished(self):
        return self.finished_filepath().exists()

    def finished_filepath(self):
        return self.folderpath / "finished"

    def mark_as_finished(self):
        self.finished_filepath().touch(exist_ok=True)

    def mark_as_unfinished(self):
        f = self.finished_filepath()
        if f.exists():
            f.unlink()

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def description(self) -> str:
        pass

    def experiment_fork(self, message, function):
        self.print_date(message)
        new_pid = os.fork()
        if new_pid == 0:
            function()
            sys.exit(0)
        else:
            pid, status = os.waitpid(0, 0)
            if status != 0:
                self.print_date(f" Error in: {message}")
                sys.exit(status)
    @classmethod
    def print_table(cls, experiments: ['Experiment']):
        table = texttable.Texttable()
        header = ["Experiment", "Finished"]
        table.header(header)
        experiments.sort(key=lambda e: e.__class__.__name__)
        for e in experiments:
            status = "Y" if e.has_finished() else "N"
            name = e.__class__.__name__
            name = name[:40]
            table.add_row((name, status))
            # print(f"{name:40}     {status}")
        print(table.draw())

    @classmethod
    def parse_args(cls, experiments: Dict[str, List[Experiment]]) -> (List[Experiment], Options):
        parser = argparse.ArgumentParser(description="Run invariance with transformation measures.")
        group_names = list(experiments.keys())

        experiments_plain = [e for g in experiments.values() for e in g]
        experiment_names = [e.id() for e in experiments_plain]
        experiment_dict = dict(zip(experiment_names, experiments_plain))

        parser.add_argument('-experiment',
                            help=f'Choose an experiment to run',
                            type=str,
                            default=None,
                            required=False, choices=experiment_names, )
        parser.add_argument('-group',
                            help=f'Choose an experiment group to run',
                            type=str,
                            default=None,
                            required=False, choices=group_names, )
        parser.add_argument('-force',
                            help=f'Force invariance to rerun even if they have already finished',
                            action="store_true")
        parser.add_argument('-list',
                            help=f'List invariance and status',
                            action="store_true")

        argcomplete.autocomplete(parser)
        args = parser.parse_args()
        if not args.experiment is None and not args.group is None:
            sys.exit("Cant specify both experiment and experiment group")
        selected_experiments = experiments_plain
        if not args.experiment is None:
            selected_experiments = [experiment_dict[args.experiment]]
        if not args.group is None:
            selected_experiments = experiments[args.group]

        return selected_experiments, Options(args.list, args.force)

