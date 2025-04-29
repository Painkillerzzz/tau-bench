import abc
from typing import Optional
from tau_bench.envs.base import Env
from tau_bench.types import ProcessResult


class Module(abc.ABC):
    @abc.abstractmethod
    def process(
        self, env: Env, task_index: Optional[int] = None
    ) -> ProcessResult:
        raise NotImplementedError