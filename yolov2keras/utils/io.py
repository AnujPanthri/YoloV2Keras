# Copyright 2023 The yolov2keras Authors. All Rights Reserved.
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

import os
import wget
import pathlib
import yolov2keras as y2k

from urllib.parse import urlparse
from appdirs import user_cache_dir

_CACHE_DIR = user_cache_dir(y2k.__name__)


def GetFileFromUrl(
  url: str, /, *, force=False, out=_CACHE_DIR
) -> pathlib.Path:
  """Download a file from a given URL and save it to a cache directory.

  This function downloads a file from the specified URL and saves it to a cache
  directory. If the file already exists in the cache directory and the `force`
  argument is set to False, it is returned without downloading it again. If
  `force` is set to True, the existing file will be replaced.

  Args:
    url (str): The URL of the file to download.
    force (bool, optional): If True, force the download even if the file exists.
    												Default is False.
    out (str, optional): The output directory to save the downloaded file.
    											Default is `_CACHE_DIR`.

  Returns:
    pathlib.Path: A Path object representing the downloaded file's location in
                  the cache directory.

  Example:
    >>> url = "https://example.com/sample.txt"
    >>> downloaded_file = GetFileFromUrl(url)
    >>> print(downloaded_file)
    PosixPath('download_cache/sample.txt')

  Notes:
    - The cache directory (_CACHE_DIR) is used to store downloaded files.
    - The function checks if the file already exists in the cache before
      downloading it, based on the `force` argument.
    - The `wget` library is used to perform the file download.
    - You can change the cache directory by modifying the `_CACHE_DIR` variable.
  """
  if not os.path.exists(out):
    os.makedirs(out)

  basename = os.path.basename(urlparse(url).path)

  f_exists = os.access(os.path.join(out, basename), os.F_OK | os.R_OK)
  if f_exists and force is False:
    return pathlib.Path(os.path.join(out, basename))
  else f_exists and force is True:
  	os.remove(os.path.join(out, basename))

  wget.download(url, out=out)

  return pathlib.Path(os.path.join(out, basename))
