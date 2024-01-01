import os
from datetime import datetime

from config import TRIALS_DIR, CLASS_NAMES, INPUT_SHAPE, BATCH_SIZE, LEARNING_RATE, EPOCHS, METHOD, TRIAL_ID


class Config:
    def __init__(self, method=METHOD, trial_id=TRIAL_ID, class_names=CLASS_NAMES,
                 input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS):
        self.method = method.lower()
        if trial_id is None:
            self.trial_id = datetime.now().strftime("%Y%m%d%H%M%S")
        elif trial_id == "latest":
            self.trial_id = sorted(os.listdir(os.path.join(str(TRIALS_DIR), self.method)),
                                   key=lambda x: datetime.strptime(x, "%Y%m%d%H%M%S"))[-1]
        else:
            self.trial_id = str(trial_id)

        self.trial_dir = os.path.join(TRIALS_DIR, self.method, str(self.trial_id))
        self.model_dir = os.path.join(str(self.trial_dir), "model")
        self.metrics_dir = os.path.join(str(self.trial_dir), "metrics")
        [os.makedirs(_dir, exist_ok=True) for _dir in [self.model_dir, self.metrics_dir]]

        self.class_names = class_names
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
