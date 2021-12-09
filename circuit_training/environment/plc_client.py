# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PlacementCost client class."""

import json
import socket
import subprocess
import tempfile
from typing import Any, Text

from absl import flags
from absl import logging

flags.DEFINE_string('plc_wrapper_main', 'plc_wrapper_main',
                    'Path to plc_wrapper_main binary.')

FLAGS = flags.FLAGS


class PlacementCost(object):
  """PlacementCost object wrapper."""

  BUFFER_LEN = 1024 * 1024
  MAX_RETRY = 10

  def __init__(self,
               netlist_file: Text,
               macro_macro_x_spacing: float = 0.0,
               macro_macro_y_spacing: float = 0.0) -> None:
    """Creates a PlacementCost client object.

    It creates a subprocess by calling plc_wrapper_main and communicate with
    it over an `AF_UNIX` channel.

    Args:
      netlist_file: Path to the netlist proto text file.
      macro_macro_x_spacing: Macro-to-macro x spacing in microns.
      macro_macro_y_spacing: Macro-to-macro y spacing in microns.
    """
    if not FLAGS.plc_wrapper_main:
      raise ValueError('FLAGS.plc_wrapper_main should be specified.')

    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    address = tempfile.NamedTemporaryFile().name
    self.sock.bind(address)
    self.sock.listen(1)
    args = [
        FLAGS.plc_wrapper_main,  #
        '--uid=',
        '--gid=',
        f'--pipe_address={address}',
        f'--netlist_file={netlist_file}',
        f'--macro_macro_x_spacing={macro_macro_x_spacing}',
        f'--macro_macro_y_spacing={macro_macro_y_spacing}',
    ]
    self.process = subprocess.Popen([str(a) for a in args])
    self.conn, _ = self.sock.accept()

  # See circuit_training/environment/plc_client_test.py for the supported APIs.
  def __getattr__(self, name) -> Any:

    # snake_case to PascalCase.
    name = name.replace('_', ' ').title().replace(' ', '')

    def f(*args) -> Any:
      json_args = json.dumps({'name': name, 'args': args})
      self.conn.send(json_args.encode('utf-8'))
      json_ret = b''
      retry = 0
      # The stream from the unix socket can be incomplete after a single call
      # to `recv` for large (200kb+) return values, e.g. GetMacroAdjacency. The
      # loop retries until the returned value is valid json. When the host is
      # under load ~10 retries have been needed. Adding a sleep did not seem to
      # make a difference only added latency. b/210838186
      while True:
        part = self.conn.recv(PlacementCost.BUFFER_LEN)
        json_ret += part
        if len(part) < PlacementCost.BUFFER_LEN:
          json_str = json_ret.decode('utf-8')
          try:
            output = json.loads(json_str)
            break
          except json.decoder.JSONDecodeError as e:
            logging.warn('JSONDecode Error for %s \n %s', name, e)
            if retry < PlacementCost.MAX_RETRY:
              logging.info('Looking for more data for %s on connection:%s/%s',
                           name, retry, PlacementCost.MAX_RETRY)
              retry += 1
            else:
              raise e
      if isinstance(output, dict):
        if 'ok' in output and not output['ok']:  # Status::NotOk
          raise ValueError(
              f"Error in calling {name} with {args}: {output['message']}.")
        elif '__tuple__' in output:  # Tuple
          output = tuple(output['items'])
      return output

    return f

  def __del__(self) -> None:
    self.conn.close()
    self.process.kill()
    self.process.wait()
    self.sock.close()
