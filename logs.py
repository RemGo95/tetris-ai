from keras.callbacks import TensorBoard
import tensorflow as tf
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import os.path
import time
import warnings

import tensorflow._api
import tensorflow.include

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter
from tensorflow.python.summary.writer.event_file_writer_v2 import EventFileWriterV2
from tensorflow.python.util.tf_export import tf_export

from tensorflow.core.framework import summary_pb2
from tensorflow.python.summary import plugin_asset

from tensorflow.python.framework import meta_graph

#from tensorflow import FileWriter
#fromTensorBoard import SelfWriter
#import tensorflow as tf
#from tensorflow.xla_aot_runtime_src import 
#import keras.models
_PLUGINS_DIR = "plugins"

class SummaryToEventTransformer(object):

  def __init__(self, event_writer, graph=None, graph_def=None):
  
    self.event_writer = event_writer
    # For storing used tags for session.run() outputs.
    self._session_run_tags = {}
    if graph is not None or graph_def is not None:
      # Calling it with both graph and graph_def for backward compatibility.
      self.add_graph(graph=graph, graph_def=graph_def)
      # Also export the meta_graph_def in this case.
      # graph may itself be a graph_def due to positional arguments
      maybe_graph_as_def = (graph.as_graph_def(add_shapes=True)
                            if isinstance(graph, ops.Graph) else graph)
      self.add_meta_graph(
          meta_graph.create_meta_graph_def(graph_def=graph_def or
                                           maybe_graph_as_def))

    
    self._seen_summary_tags = set()

  def add_summary(self, summary, global_step=None):
    
    if isinstance(summary, bytes):
      summ = summary_pb2.Summary()
      summ.ParseFromString(summary)
      summary = summ

    
    for value in summary.value:
      if not value.metadata:
        continue

      if value.tag in self._seen_summary_tags:
        # This tag has been encountered before. Strip the metadata.
        value.ClearField("metadata")
        continue

      
      self._seen_summary_tags.add(value.tag)

    event = event_pb2.Event(summary=summary)
    self._add_event(event, global_step)

  def add_session_log(self, session_log, global_step=None):
    event = event_pb2.Event(session_log=session_log)
    self._add_event(event, global_step)

  def _add_graph_def(self, graph_def, global_step=None):
    graph_bytes = graph_def.SerializeToString()
    event = event_pb2.Event(graph_def=graph_bytes)
    self._add_event(event, global_step)

  def add_graph(self, graph, global_step=None, graph_def=None):
  

    if graph is not None and graph_def is not None:
      raise ValueError("Please pass only graph, or graph_def (deprecated), "
                       "but not both.")

    if isinstance(graph, ops.Graph) or isinstance(graph_def, ops.Graph):
      # The user passed a `Graph`.

      # Check if the user passed it via the graph or the graph_def argument and
      # correct for that.
      if not isinstance(graph, ops.Graph):
        logging.warning("When passing a `Graph` object, please use the `graph`"
                        " named argument instead of `graph_def`.")
        graph = graph_def

      # Serialize the graph with additional info.
      true_graph_def = graph.as_graph_def(add_shapes=True)
      self._write_plugin_assets(graph)
    elif (isinstance(graph, graph_pb2.GraphDef) or
          isinstance(graph_def, graph_pb2.GraphDef)):
      # The user passed a `GraphDef`.
      logging.warning("Passing a `GraphDef` to the SummaryWriter is deprecated."
                      " Pass a `Graph` object instead, such as `sess.graph`.")

      # Check if the user passed it via the graph or the graph_def argument and
      # correct for that.
      if isinstance(graph, graph_pb2.GraphDef):
        true_graph_def = graph
      else:
        true_graph_def = graph_def

    else:
      # The user passed neither `Graph`, nor `GraphDef`.
      raise TypeError("The passed graph must be an instance of `Graph` "
                      "or the deprecated `GraphDef`")
    # Finally, add the graph_def to the summary writer.
    self._add_graph_def(true_graph_def, global_step)

  def _write_plugin_assets(self, graph):
    plugin_assets = plugin_asset.get_all_plugin_assets(graph)
    logdir = self.event_writer.get_logdir()
    for asset_container in plugin_assets:
      plugin_name = asset_container.plugin_name
      plugin_dir = os.path.join(logdir, _PLUGINS_DIR, plugin_name)
      gfile.MakeDirs(plugin_dir)
      assets = asset_container.assets()
      for (asset_name, content) in assets.items():
        asset_path = os.path.join(plugin_dir, asset_name)
        with gfile.Open(asset_path, "w") as f:
          f.write(content)

  def add_meta_graph(self, meta_graph_def, global_step=None):
    
    if not isinstance(meta_graph_def, meta_graph_pb2.MetaGraphDef):
      raise TypeError("meta_graph_def must be type MetaGraphDef, saw type: %s" %
                      type(meta_graph_def))
    meta_graph_bytes = meta_graph_def.SerializeToString()
    event = event_pb2.Event(meta_graph_def=meta_graph_bytes)
    self._add_event(event, global_step)

  def add_run_metadata(self, run_metadata, tag, global_step=None):
    
    if tag in self._session_run_tags:
      raise ValueError("The provided tag was already used for this event type")
    self._session_run_tags[tag] = True

    tagged_metadata = event_pb2.TaggedRunMetadata()
    tagged_metadata.tag = tag
    # Store the `RunMetadata` object as bytes in order to have postponed
    # (lazy) deserialization when used later.
    tagged_metadata.run_metadata = run_metadata.SerializeToString()
    event = event_pb2.Event(tagged_run_metadata=tagged_metadata)
    self._add_event(event, global_step)

  def _add_event(self, event, step):
    event.wall_time = time.time()
    if step is not None:
      event.step = int(step)
    self.event_writer.add_event(event)



class FileWriter(SummaryToEventTransformer):


  def __init__(self,
               logdir,
               graph=None,
               max_queue=10,
               flush_secs=120,
               graph_def=None,
               filename_suffix=None,
               session=None):

    #if context.executing_eagerly():
    #  raise RuntimeError(tf.summary.create_file_writer)

    if session is not None:
      event_writer = EventFileWriterV2(
          session, logdir, max_queue, flush_secs, filename_suffix)
    else:
      event_writer = EventFileWriter(logdir, max_queue, flush_secs,
                                     filename_suffix)

    self._closed = False
    super(FileWriter, self).__init__(event_writer, graph, graph_def)

  def __enter__(self):
    
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    
    self.close()

  def get_logdir(self):
    
    return self.event_writer.get_logdir()

  def _warn_if_event_writer_is_closed(self):
    if self._closed:
      warnings.warn("Attempting to use a closed FileWriter. "
                    "The operation will be a noop unless the FileWriter "
                    "is explicitly reopened.")

  def _add_event(self, event, step):
    self._warn_if_event_writer_is_closed()
    super(FileWriter, self)._add_event(event, step)

  def add_event(self, event):
    
    self._warn_if_event_writer_is_closed()
    self.event_writer.add_event(event)

  def flush(self):
    
    self._warn_if_event_writer_is_closed()
    self.event_writer.flush()

  def close(self):
    
    self.event_writer.close()
    self._closed = True

  def reopen(self):
   
    self.event_writer.reopen()
    self._closed = False

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):
        pass

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

    def log(self, step, **stats):
        self._write_logs(stats, step)
       # print("\n\n ******step***** :", step)
        #print("\n\n ******stats***** :", stats)
        
    #save_model
    #filepath = "data.txt"
    #f = open(filepath, "a")
    #f.write(step)
    #f.close()
    #self._log_weights(stats, step)