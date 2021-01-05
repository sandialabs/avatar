import numpy
import os
import re
import shutil
import subprocess
import tempfile

class _confusion_matrix(object):
  def __init__(self, labels, matrix):
    labels = numpy.array(labels)
    matrix = numpy.array(matrix)

    if labels.ndim != 1:
      raise ValueError("Labels must be a 1D array.")
    if matrix.ndim != 2:
      raise ValueError("Matrix must be a 2D array.")
    if matrix.shape != (labels.shape[0], labels.shape[0]):
      raise ValueError("Expected %s matrix, received %s." % ((labels.shape[0], labels.shape[0]), matrix.shape))

    self._labels = labels
    self._matrix = matrix

  def predictions(self):
    """Return the total number of predictions in the matrix."""
    return self._matrix.sum()

  def correct(self):
    """Return the number of correct predictions in the matrix."""
    return self._matrix.diagonal().sum()

  def incorrect(self):
    """Return the number of incorrect predictions in the matrix."""
    return self._matrix.sum() - self._matrix.diagonal().sum()

def build(train, test=None, truth=None, exclude=[], discrete_threshold=10, seed=None):
  import avatar.data
  import avatar.names

  workdir = tempfile.mkdtemp()

  # Setup data for training ...
  avatar.data.dump(train, os.path.join(workdir, "ensemble.data"))
  avatar.names.guess((train, test) if test is not None else train, os.path.join(workdir, "ensemble.names"), truth=truth, exclude=exclude, discrete_threshold=discrete_threshold)

  command = ["avatardt", "-o", "avatar", "-f", os.path.join(workdir, "ensemble"), "--train", "--bagging", "--use-stopping-algorithm"]
  if test is not None:
    avatar.data.dump(test, os.path.join(workdir, "ensemble.test"))
    command += ["--test", "--output-accuracies", "--output-performance-metrics", "--output-predictions", "--output-confusion-matrix"]
  if seed is not None:
    command += ["--seed", str(seed)]

  # Log the avatar command line for posterity
  with open(os.path.join(workdir, "command.txt"), "w") as file:
    file.write("%s\n" % " ".join(command))

  avatardt = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
  stdout, stderr = avatardt.communicate()

  # Log stdout and stderr for posterity
  with open(os.path.join(workdir, "stdout.txt"), "w") as file:
    file.write(stdout)

  with open(os.path.join(workdir, "stderr.txt"), "w") as file:
    file.write(stderr)

  #if avatardt.returncode != 0:
  #  raise Exception(stderr)

  class results:
    def __init__(self, workdir, command, test, stdout, stderr):
      self.workdir = workdir
      self.command = command
      self.test = test
      self.returncode = avatardt.returncode
      self.stdout = stdout
      self.stderr = stderr
    def __del__(self):
      shutil.rmtree(self.workdir)
    def oob_accuracy(self):
      return float(re.search("oob accuracy of(.*)%", self.stdout).group(1)) * 0.01
    def average_accuracy(self):
      if not test:
        raise Exception("Cannot compute average accuracy without test data.")
      return float(re.search("Average Accuracy\s+=(.*)%", self.stdout).group(1)) * 0.01
    def voted_accuracy(self):
      if not test:
        raise Exception("Cannot compute voted accuracy without test data.")
      return float(re.search("Voted Accuracy\s+=(.*)%", self.stdout).group(1)) * 0.01
    def performance_metrics(self):
      if not test:
        raise Exception("Cannot compute performance metrics without test data.")
      lines = [line.split() for line in self.stdout.split("Performance Metrics:")[1].split("TRUTH")[0].strip().split("\n")]
      labels = lines[0][:-3]
      metrics = {}
      for line in lines[2:]:
        metric = line[0]
        if metric == "Brier":
          metric = "brier"
          values = line[2:2+len(labels)]
        else:
          metric = metric[:-1].lower()
          values = line[1:1+len(labels)]
        metrics[metric] = numpy.array(values, dtype="float64")
      return {"labels":labels, "metrics":metrics}
    def confusion_matrix(self):
      def is_integer(value):
        try:
          int(value)
          return True
        except:
          return False

      if not test:
        raise Exception("Cannot compute confusion matrix without test data.")
      rows = self.stdout.split("TRUTH")[1].split("\n")
      labels = [label.strip() for label in rows[1].split()]
      matrix = [[int(field) for field in row.split() if is_integer(field)][:len(labels)] for row in rows[3:] if len(row.split())]
      return _confusion_matrix(labels, matrix)

    def predictions(self):
      if not test:
        raise Exception("Cannot compute predictions without test data.")
      return numpy.array([line.split(",")[-1].strip() for line in open(os.path.join(self.workdir, "ensemble.pred"), "r")][1:])
    def proximity(self):
      if not test:
        raise Exception("Cannot compute proximity without test data.")
      import numpy
      import scipy.sparse
      proximity = subprocess.Popen(["proximity", "-o", "avatar", "-f", os.path.join(self.workdir, "ensemble")], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = proximity.communicate()

      i, j, data = zip(*[line.split(",") for line in open(os.path.join(self.workdir, "ensemble.proximity_matrix"))])
      i = numpy.array(i, dtype="int64")
      j = numpy.array(j, dtype="int64")
      data = numpy.array(data, dtype="double")
      return scipy.sparse.coo_matrix((data, (i, j)))

  return results(workdir, command, test is not None, stdout, stderr)
