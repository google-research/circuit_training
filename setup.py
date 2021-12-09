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
"""Build, test, and install circuit training."""
import argparse
import codecs
import datetime
import os
import sys

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

# Enables importing version.py directly by adding its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'circuit_training')
sys.path.append(version_path)
import version as circuit_training_version  # pylint: disable=g-import-not-at-top

# Default versions the tf-agents dependency.
TF_AGENTS = 'tf-agents[reverb]'
TF_AGENTS_NIGHTLY = 'tf-agents-nightly[reverb]'


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class SetupToolsHelper(object):
  """Helper to execute `setuptools.setup()`."""

  def __init__(self, release=True, tf_agents_override=None):
    """Initialize SetupToolsHelper class.

    Args:
      release: True to do a release build. False for a nightly build.
      tf_agents_override: Set to override the tf_agents dependency.
    """
    self.release = release
    self.tf_agents_override = tf_agents_override

  def _get_version(self):
    """Returns the version and project name to associate with the build."""
    if self.release:
      project_name = 'circuit_training'
      version = circuit_training_version.__rel_version__
    else:
      project_name = 'circuit_training-nightly'
      version = circuit_training_version.__dev_version__
      version += datetime.datetime.now().strftime('%Y%m%d')

    return version, project_name

  def _get_tf_agents_packages(self):
    """Returns required tf_agents package."""
    if self.release:
      tf_agents_version = TF_AGENTS
    else:
      tf_agents_version = TF_AGENTS_NIGHTLY

    # Overrides required versions if tf_version_override is set.
    if self.tf_agents_override:
      tf_agents_version = self.tf_agents_override

    return [tf_agents_version]

  def run_setup(self):
    # Builds the long description from the README.
    root_path = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(root_path, 'README.md'),
                     encoding='utf-8') as f:
      long_description = f.read()

    version, project_name = self._get_version()
    setup(
        name=project_name,
        version=version,
        description=('Circuit Training'),
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Google LLC',
        author_email='no-reply@google.com',
        url='https://github.com/google-research/circuit_training',
        license='Apache 2.0',
        packages=find_packages(),
        include_package_data=True,
        install_requires=self._get_tf_agents_packages(),
        extras_require={
            'testing': self._get_tf_agents_packages(),
        },
        distclass=BinaryDistribution,
        python_requires='>=3',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        keywords='google-research reinforcement learning circuit training',
    )


if __name__ == '__main__':
  # Hide argparse help so `setuptools.setup` help prints. This pattern is an
  # improvement over using `sys.argv` and then `sys.argv.remove`, which also
  # did not provide help about custom arguments.
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument(
      '--release',
      action='store_true',
      help='Pass as true to do a release build')
  parser.add_argument(
      '--tf-agents-version',
      type=str,
      default=None,
      help='Overrides version of TF-Agents required')

  FLAGS, unparsed = parser.parse_known_args()
  # Go forward with only non-custom flags.
  sys.argv.clear()
  # Downstream `setuptools.setup` expects args to start at the second element.
  unparsed.insert(0, 'foo')
  sys.argv.extend(unparsed)
  setup_tools_helper = SetupToolsHelper(
      release=FLAGS.release,
      tf_agents_override=FLAGS.tf_agents_version)
  setup_tools_helper.run_setup()
