class ProcessFailed(Exception):
    """Raised when a mandatory process fails."""


class ProcessHallucinationDetected(Exception):
    """Raised when the agent keeps repeating actions without progress."""

    def __init__(self, message: str, state_hash: str | None = None):
        super().__init__(message)
        self.state_hash = state_hash
