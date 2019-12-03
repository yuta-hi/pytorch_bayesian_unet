from __future__ import absolute_import

import json
import os
import six
from chainer.training import extensions


class LogReport(extensions.LogReport):

    """__init__(\
    keys=None, trigger=(1, 'epoch'), postprocess=None, filename='log')
    Trainer extension to output the accumulated results to a log file.
    This extension accumulates the observations of the trainer to
    :class:`~chainer.DictSummary` at a regular interval specified by a supplied
    trigger, and writes them into a log file in JSON format.
    There are two triggers to handle this extension. One is the trigger to
    invoke this extension, which is used to handle the timing of accumulating
    the results. It is set to ``1, 'iteration'`` by default. The other is the
    trigger to determine when to emit the result. When this trigger returns
    True, this extension appends the summary of accumulated values to the list
    of past summaries, and writes the list to the log file. Then, this
    extension makes a new fresh summary object which is used until the next
    time that the trigger fires.
    It also adds some entries to each result dictionary.
    - ``'epoch'`` and ``'iteration'`` are the epoch and iteration counts at the
      output, respectively.
    - ``'elapsed_time'`` is the elapsed time in seconds since the training
      begins. The value is taken from :attr:`Trainer.elapsed_time`.

    Args:
        keys (iterable of strs): Keys of values to accumulate. If this is None,
            all the values are accumulated and output to the log file.
        trigger: Trigger that decides when to aggregate the result and output
            the values. This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        postprocess: Callback to postprocess the result dictionaries. Each
            result dictionary is passed to this callback on the output. This
            callback can modify the result dictionaries, which are used to
            output to the log file.
        log_json_name (str): Name of the log file for json format under the output
            directory. It can be a format string: the last result dictionary is
            passed for the formatting. For example, users can use '{iteration}'
            to separate the log files for different iterations. If the log name
            is None, it does not output the log to any file.
        log_csv_name (str): Name of the log file for csv format under the output
            directory.
    """

    def __init__(self, keys=None, trigger=(1, 'iteration'), postprocess=None,
                 json_name='log', csv_name='log.csv', **kwargs):

        super(LogReport, self).__init__(
            keys, trigger, postprocess, json_name, **kwargs)

        self._log_json_name = self._log_name
        self._log_csv_name = csv_name

    def _write_json_log(self, path, _dict, indent=4):
        """ Append data in JSON format to the end of a JSON file.
        In the original implementation, if you save it for each iteration,
        you will write out more of it at once and it will be slower.
        NOTE: In the original implementation, saving per each iteration might slow down the computation time.
        NOTE: Assumes file contains a JSON object (like a Python
        dict) ending in '}'.
        """

        with open(path, 'ab') as fp:
            fp.seek(0, 2)  # Go to the end of file
            if fp.tell() == 0:  # Check if file is empty
                new_ending = json.dumps(_dict, indent=indent)
                new_ending = new_ending.split('\n')
                new_ending = [' '*indent + x for x in new_ending]
                new_ending = '\n'.join(new_ending)
                new_ending = '[\n' + new_ending + '\n]'
                fp.write(new_ending.encode())

            else:
                fp.seek(-2, 2)
                fp.truncate()  # Remove the last two character

                new_ending = json.dumps(_dict, indent=indent)
                new_ending = new_ending.split('\n')
                new_ending = [' ' * indent + x for x in new_ending]
                new_ending = '\n'.join(new_ending)
                new_ending = ',\n' + new_ending + '\n]'
                fp.write(new_ending.encode())

    def _accumulate_observations(self, trainer):

        keys = self._keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        return summary

    def initialize(self, trainer):

        keys = self._keys
        summary = self._accumulate_observations(trainer)

        # make header
        if keys is None:
            self._keys = ['epoch', 'iteration', 'elapsed_time']
            self._keys.extend(sorted(summary._summaries.keys()))
        else:
            self._keys = ['epoch', 'iteration', 'elapsed_time']
            for k in keys:
                if k not in self._keys:
                    self._keys.append(k)

        self._log_csv_name = os.path.join(trainer.out, self._log_csv_name)
        self._log_json_name = os.path.join(trainer.out, self._log_json_name)

        os.makedirs(os.path.dirname(self._log_csv_name), exist_ok=True)
        os.makedirs(os.path.dirname(self._log_json_name), exist_ok=True)

        with open(self._log_csv_name, 'w+') as fp:
            fp.write(','.join(self._keys) + '\n')

        self.__call__(trainer)

    def _update(self, data):
        entry = {key: data[key] if key in data else None for key in self._keys}

        # write CSV file
        with open(self._log_csv_name, 'a') as fp:
            temp_list = []
            for h in self._keys:
                if h in data.keys():
                    temp_list.append(str(data[h]))
                else:
                    temp_list.append(','.join(' '))
            fp.write(','.join(temp_list) + '\n')

        # write JSON file
        self._write_json_log(self._log_json_name, entry)

    def __call__(self, trainer):

        summary = self._accumulate_observations(trainer)

        # output the result
        stats = summary.compute_mean()
        stats_cpu = {}
        for name, value in six.iteritems(stats):
            stats_cpu[name] = float(value)  # copy to CPU

        updater = trainer.updater
        stats_cpu['epoch'] = updater.epoch
        stats_cpu['iteration'] = updater.iteration
        stats_cpu['elapsed_time'] = trainer.elapsed_time

        if self._postprocess is not None:
            self._postprocess(stats_cpu)

        self._log.append(stats_cpu)

        # write to the log file
        self._update(stats_cpu)

        # reset the summary for the next output
        self._init_summary()
