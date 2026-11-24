"""Read a known-size Parquet file using validated HTTP byte ranges."""

import io
import re
import time

import requests


_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
_TEMPORARY_STATUS = {408, 429, 500, 502, 503, 504}


class RangeReader(io.RawIOBase):
    """Expose a remote file to PyArrow without downloading unrelated columns."""

    def __init__(self, url, size, *, session=None):
        super().__init__()
        if size < 0:
            raise ValueError("Remote file size cannot be negative.")
        self._url = url
        self._size = size
        self._position = 0
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        self.bytes_read = 0
        self.request_count = 0

    def readable(self):
        return not self.closed

    def seekable(self):
        return not self.closed

    def tell(self):
        self._checkClosed()
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        self._checkClosed()
        if whence == io.SEEK_SET:
            destination = offset
        elif whence == io.SEEK_CUR:
            destination = self._position + offset
        elif whence == io.SEEK_END:
            destination = self._size + offset
        else:
            raise ValueError("Invalid seek origin.")
        if destination < 0:
            raise ValueError("Cannot seek before the start of a remote file.")
        self._position = destination
        return destination

    def read(self, size=-1):
        self._checkClosed()
        size = min(size, self._size - self._position) if size >= 0 else self._size - self._position
        if size <= 0:
            return b""
        start = self._position
        stop = start + size - 1
        headers = {"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity"}
        for attempt in range(4):
            self.request_count += 1
            try:
                with self._session.get(self._url, headers=headers, timeout=(15, 120), stream=True) as response:
                    response.raise_for_status()
                    actual = _CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                    if response.status_code != 206 or actual is None:
                        raise OSError(f"The server did not honor the byte range for {self._url}.")
                    if tuple(map(int, actual.groups())) != (start, stop, self._size):
                        raise OSError(f"The server returned a different byte range for {self._url}.")
                    encoding = response.headers.get("Content-Encoding", "identity")
                    if encoding.lower() != "identity":
                        raise OSError(f"The server encoded the byte range ({encoding}) for {self._url}.")
                    content = response.content
                    if len(content) != size:
                        raise OSError(f"The server returned a truncated byte range for {self._url}.")
            except requests.HTTPError as exc:
                if attempt == 3 or exc.response is None or exc.response.status_code not in _TEMPORARY_STATUS:
                    raise
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
                if attempt == 3:
                    raise
            else:
                self._position += size
                self.bytes_read += size
                return content
            time.sleep(2**attempt)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        if not self.closed and self._owns_session:
            self._session.close()
        super().close()
