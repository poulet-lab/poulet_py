try:
    from collections.abc import Generator, Sequence
    from pathlib import Path
    from typing import Any

    from pandas import DataFrame
    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import DATA_SIGNATURES, BaseData, DataSignature, DataStructure, DiscoveryStrategy
    from poulet_py.io.data_structures.filters import Index, _ILocIndexer, _LocIndexer
except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[analysis]
- Module group: pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]

"""
    raise ImportError(msg) from e


class Trial(BaseModel):
    path: Path = Field(..., description="Path to trial folder or file")

    _signature: DataSignature = PrivateAttr()
    _data: BaseData = PrivateAttr()
    _n_trials: int | None = PrivateAttr(default=None)
    _metadata: DataFrame = PrivateAttr(default_factory=DataFrame)

    @property
    def n_trials(self) -> int | None:
        return self._n_trials

    @property
    def signature(self) -> DataSignature | None:
        return self._signature

    @property
    def metadata(self) -> DataFrame | None:
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any] | DataFrame):
        if isinstance(value, dict):
            self._metadata = DataFrame([value])
        elif isinstance(value, DataFrame):
            self._metadata = value
        else:
            msg = "Metadata must be a dict or DataFrame"
            raise ValueError(msg)

    @property
    def data(self) -> BaseData | None:
        return self._data

    def model_post_init(self, __context):
        for signature in DATA_SIGNATURES.values():
            if signature.matches(self.path):
                self._signature = signature
                self._data = signature.data_type(path=self.path)

                if self._signature.data_structure == DataStructure.FOLDER_PER_TRIAL:
                    self._n_trials = 1
                elif self._signature.data_structure == DataStructure.SINGLE_FILE:
                    # TODO implement logic to determine number of trials in a single file
                    self._n_trials = None
                else:
                    self._n_trials = None

                return

        msg = f"No matching data signature found for path: {self.path}"
        raise ValueError(msg)


class Session(BaseModel):
    path: Path = Field(..., description="Root path to search for sessions")
    discovery_strategy: DiscoveryStrategy | None = Field(
        default=None, description="How to discover trials"
    )
    signature: DataSignature | None = Field(
        default=None, description="Expected data signature for trials"
    )
    _paths: DataFrame = PrivateAttr()
    _trials: Index[Trial] = PrivateAttr()

    def model_post_init(self, __context):
        self._paths = (
            self.discovery_strategy.to_df(self.path)
            if self.discovery_strategy
            else DataFrame([DiscoveryStrategy.path_stats(p) for p in self.path.rglob("*")])
        )

        trials = []
        for _, row in self._paths.iterrows():
            if row["is_dir"]:
                trial = Trial(path=row["parent"] / row["name"])
                trial.metadata = row.to_frame().T
                if not self.signature or (trial.signature and self.signature == trial.signature):
                    trials.append(trial)

        self._trials = Index(trials)

    @property
    def trials(self) -> Index[Trial]:
        return self._trials

    @property
    def iloc(self) -> _ILocIndexer[Trial]:
        return self._trials.iloc

    @property
    def loc(self) -> _LocIndexer[Trial]:
        return self._trials.loc

    # def query(self) -> Query:
    #     return Query(self)

    def __len__(self) -> int:
        return sum(trial._n_trials or 1 for trial in self._trials)

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for trial in self._trials:
            if trial.data is not None:
                yield str(trial.path), trial.data

    def __getitem__(
        self, key: int | str | slice | Sequence[int] | Sequence[str]
    ) -> Trial | Sequence[Trial]:
        return self._trials[key]
