import numbers
import numpy

"""Helpers for working with data in Avatar .data format.
"""

class column(numpy.ndarray):
  """Encapsulates a labelled column from an Avatar .data file, using a Numpy ndarray for storage.
  """
  def __new__(cls, sequence, label):
    obj = numpy.asarray(sequence).view(cls)
    obj.label = label
    return obj

  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.label = getattr(obj, 'label', None)

  def __reduce__(self):
    state = list(numpy.ndarray.__reduce__(self))
    state[2] = (state[2], (self.label, ))
    return tuple(state)

  def __setstate__(self, state):
    numpy.ndarray.__setstate__(self, state[0])
    self.label = state[1][0]

class columns(list):
  """Encapsulates an ordered collection of labelled columns from an Avatar .data file.

  You may retrieve columns by integer index as with a normal list, or by column
  label, both using [] syntax.  Note that a .data file could contain duplicate
  column labels, in which case label lookup will throw an exception and index
  lookup must be used.
  """
  def index(self, key):
    """Return the zero-based index of a column, identified by zero-based index or label.

    Raises an exception if the label can't be found, or there is more than one
    column with the given label.
    """
    if isinstance(key, numbers.Integral):
      return key
    else:
      candidates = [(index, column) for index, column in enumerate(self) if column.label == key]
      if len(candidates) == 0:
        raise LookupError("No column with label %s." % key)
      elif len(candidates) == 1:
        return candidates[0][0]
      else:
        raise LookupError("More than one column with label %s." % key)

  def __getitem__(self, key):
    return list.__getitem__(self, self.index(key))

  def __setitem__(self, key, value):
    list.__setitem__(self, self.index(key), value)

  def __getslice__(self,i,j):
    return columns(list.__getslice__(self, i, j))

  def __add__(self,other):
    return columns(list.__add__(self,other))

  def __mul__(self,other):
    return columns(list.__mul__(self,other))

  def __delitem__(self, key):
    return list.__delitem__(self, self.index(key))

  def __repr__(self):
    return "<avatar.data.columns %s>" % ", ".join([column.label for column in self])

  def _repr_html_(self):
    import StringIO
    buffer = StringIO.StringIO()

    buffer.write("<table>")
    buffer.write("<tr>")
    buffer.write("".join(["<th>" + column.label + "</th>" for column in self]))
    buffer.write("</tr>")
    try:
      iterators = [iter(column) for column in self]
      while True:
        for index, iterator in enumerate(iterators):
          value = iterator.next()
          if index == 0:
            buffer.write("<tr>")
          buffer.write("<td>")
          buffer.write(str(value).strip())
          buffer.write("</td>")
        buffer.write("</tr>")
    except StopIteration:
      pass
    buffer.write("</table>")
    return buffer.getvalue()

  def insert(self, key, column):
    list.insert(self, self.index(key), column)

def load(file, skiplines=0):
  """Load an Avatar .data file from a file object, returning a collection of labelled columns.

  Returns a 2-tuple containing the number of rows in the returned data, and an
  ordered collection of columns.  Each column will be an instance of
  avatar.data.column, which is a subclass of numpy.ndarray with an additional
  "label" attribute containing the column label contained in the file, if any.
  If they file does not contain a #labels line, a column label will be
  synthesized.  The collection of columns is an instance of avatar.data.columns
  that allows column lookup by string label and integer index, using [] syntax.

  Use the optional skiplines argument to specify an arbitrary number of lines
  to skip at the beginning of the file.
  """
  if isinstance(file, basestring):
    file = open(file, "r")

  lines = []
  labels = []
  for index, line in enumerate(file):
    if index < skiplines:
      continue
    line = line.strip()
    if line.startswith("#labels") and not labels:
      labels = line[7:].strip().split(",")
    elif line.startswith("#"):
      pass
    else:
      line = line.split(",")
      if not labels:
        labels = ["c{}".format(index) for index, field in enumerate(line)]
      lines.append(line)

  row_count = len(lines)
  data_columns = zip(*lines)
  data_columns = [column(numpy.array(data_column, dtype="object"), label) for label, data_column in zip(labels, data_columns)]

  return row_count, columns(data_columns)

def dump(columns, file):
  """Write a collection of columns to a file in Avatar .data format.

  The columns must be a collection of avatar.data.column instances
  that include column labels.
  """

  for column in columns:
    if " " in column.label:
      raise Exception("Avatar .data column labels cannot contain spaces.")

  def formatter(value):
    if isinstance(value, basestring):
      if "," in value:
        raise Exception("Avatar .data file values cannot contain commas.")
      return value.strip()
    if numpy.isnan(value):
      raise Exception("Avatar .data file values cannot be NaN.")
    return repr(value)

  if isinstance(file, basestring):
    file = open(file, "w")

  iterators = [iter(column) for column in columns]
  file.write("#labels {}\n".format(",".join([column.label for column in columns])))
  try:
    while True:
      for index, iterator in enumerate(iterators):
        if index:
          file.write(",")
        file.write(formatter(iterator.next()))
      file.write("\n")
  except StopIteration:
    pass
