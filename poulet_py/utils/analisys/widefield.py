try:
    from pathlib import Path
    from pydantic import BaseModel, Field, PrivateAttr

    from poulet_py import LOGGER, Session, WidefieldData

except ImportError as e:
    msg = """
Missing required modules. Install options:
- Dedicated:    pip install poulet_py[analysis]
- Module group: pip install poulet_py[hardware]
- Full:         pip install poulet_py[all]

Also ensure: h5py, numpy, pandas, scikit-image, imageio, matplotlib are installed
"""
    raise ImportError(msg) from e


class WidefieldAnalysis(BaseModel):
    path: Path = Field(..., description="Path to a widefield trial or session folder")
    _session: Session = PrivateAttr()
    _active_trial_idx: int = PrivateAttr(default=0)

    @property
    def session(self) -> Session:
        if not hasattr(self, "_session"):
            msg = "Session has not been loaded. Call load() first."
            raise RuntimeError(msg)
        return self._session

    @property
    def active_trial(self) -> WidefieldData:
        if not self.session.trials:
            msg = "Session has no trials configured"
            raise ValueError(msg)
        return self.session.trials[self._active_trial_idx]

    def load(self, path: Path | str | None = None) -> None:
        """
        Load widefield data.

        Behavior:
        - path is a trial folder: open a session containing that single trial.
        - path is a session folder: discover and open child trial folders.
        - path is None: open the configured session.
        """
        if path is not None:
            self.path = Path(path)

        if not self.path.exists() or not self.path.is_dir():
            msg = f"Path does not exist or is not a directory: {self.path}"
            raise ValueError(msg)

        if hasattr(self, "_session"):
            self._session.close()

        self._session = Session(path=self.path, data_type=WidefieldData)
        self.session.open()
        self._active_trial_idx = 0
        LOGGER.info(str(self.active_trial))
