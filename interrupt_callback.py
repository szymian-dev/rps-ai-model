import tensorflow as tf
import os
import signal
import pickle

class InterruptibleTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_path, history_path):
        super().__init__()
        self.model_path = model_path
        self.history_path = history_path
        self.history_data = []
        self._interrupted = False
        
        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        print("Interrupt received! Saving model and history...")
        self._interrupted = True
        self._save_checkpoint()

    def _save_checkpoint(self):
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        with open(self.history_path, 'wb') as f:
            pickle.dump(self.history_data, f)
        print(f"History saved to {self.history_path}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['epoch'] = epoch
        self.history_data.append(logs)

        if self._interrupted:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self._save_checkpoint()

