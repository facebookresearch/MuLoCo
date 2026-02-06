# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiprocessing Dummy Context
=========================

This module provides a context-like interface for multiprocessing.dummy,
which is a wrapper around the threading module that provides a multiprocessing-like
interface but uses threads instead of processes.

This allows code that uses multiprocessing.get_context() to work with
multiprocessing.dummy by providing a compatible interface.
"""

import multiprocessing.dummy as mp
import threading
from typing import Callable, Iterable, Mapping


class DummyContext:
    """
    A context-like class for multiprocessing.dummy that mimics the interface
    of a context returned by multiprocessing.get_context().
    """

    def __init__(self, method: object = None) -> None:
        """
        Initialize the dummy context.

        Args:
            method: Ignored, only for compatibility with multiprocessing.get_context()
        """
        pass

    def Process(
        self,
        group: object = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args: Iterable[object] = (),
        kwargs: Mapping[str, object] = {},
        daemon: bool | None = None,
    ) -> mp.DummyProcess:
        """
        Create a Process using multiprocessing.dummy.Process.
        """
        return mp.Process(
            group=group, target=target, name=name, args=args, kwargs=kwargs
        )

    def Pipe(
        self, duplex: bool = True
    ) -> tuple[mp.connection.Connection, mp.connection.Connection]:
        """
        Create a Pipe using multiprocessing.dummy.Pipe.
        """
        return mp.Pipe(duplex)

    def Queue(self, maxsize: int = 0) -> mp.Queue:
        """
        Create a Queue using multiprocessing.dummy.Queue.
        """
        return mp.Queue(maxsize)

    def Event(self) -> threading.Event:
        """
        Create an Event using multiprocessing.dummy.Event.
        """
        return mp.Event()

    def Lock(self) -> threading.Lock:
        """
        Create a Lock using multiprocessing.dummy.Lock.
        """
        return mp.Lock()

    def RLock(self) -> threading.RLock:
        """
        Create an RLock using multiprocessing.dummy.RLock.
        """
        return mp.RLock()

    def Semaphore(self, value: int = 1) -> threading.Semaphore:
        """
        Create a Semaphore using multiprocessing.dummy.Semaphore.
        """
        return mp.Semaphore(value)

    def BoundedSemaphore(self, value: int = 1) -> threading.BoundedSemaphore:
        """
        Create a BoundedSemaphore using multiprocessing.dummy.BoundedSemaphore.
        """
        return mp.BoundedSemaphore(value)

    def Condition(
        self, lock: threading.Lock | threading.RLock | None = None
    ) -> threading.Condition:
        """
        Create a Condition using multiprocessing.dummy.Condition.
        """
        return mp.Condition(lock)

    def Manager(self) -> object:
        """
        Create a Manager using multiprocessing.dummy.Manager.
        """
        return mp.Manager()


def get_context(method: object = None) -> DummyContext:
    """
    Return a context object for multiprocessing.dummy.

    This function mimics multiprocessing.get_context() but returns a DummyContext
    that works with multiprocessing.dummy. This can be used to patch
    multiprocessing.dummy like so


    ```
    import multiprocessing.dummy as mp
    from torchft.multiprocessing_dummy_context import get_context
    mp.get_context = get_context
    ```

    Args:
        method: Ignored, only for compatibility with multiprocessing.get_context()

    Returns:
        A DummyContext instance
    """
    return DummyContext(method)
