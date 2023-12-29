import os
from datetime import datetime

from config import TRIALS_DIR


class Project:
    def __init__(self, trial_id=None):
        if trial_id:
            self.trial_id = trial_id
            if trial_id == "latest":
                self.trial_id = sorted(os.listdir(TRIALS_DIR), key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
        else:
            self.trial_id = str(datetime.now().strftime("%Y%m%d%H%M%S"))
        self.trial_dir = os.path.join(TRIALS_DIR, self.trial_id)
        self.model_dir = os.path.join(str(self.trial_dir), "model")
        self.model_path = os.path.join(self.model_dir, "model.keras")
        self.metrics_dir = os.path.join(str(self.trial_dir), "metrics")
        self.summary_path = os.path.join(self.metrics_dir, "summary.txt")
        self.backup_dir = os.path.join(str(self.trial_dir), "backup")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
