import os
import json
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from src.app.constants import DATA_DIR

@dataclass
class AccuracyHistory:
    x_label: str = field(default='Date')
    y_label: str = field(default='Accuracy')
    title: str = field(default='Accuracy Training History')
    fig_name: str = field(default='accuracy_training_history')
    _dates: list = field(default_factory=list, init=False)
    _acc_history: list = field(default_factory=list, init=False)
    _date: int = field(default=0, init=False)
    
    def add(self, accuracy: float) -> None:
        """Add training history"""
        self._dates.append(self._date)
        self._acc_history.append(accuracy)
        self._date += 1
        
    def save(self) -> None:
        """Convert to dictionary"""
        data = {
            'acc_history': self._acc_history
        }
        with open(os.path.join(DATA_DIR, "accuracy_history.json"), 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def from_checkpoint(cls) -> "AccuracyHistory":
        try:
            with open(os.path.join(DATA_DIR, "accuracy_history.json"), 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {'acc_history': []}
        
        his = cls()
        for acc in data['acc_history']:
            his.add(acc)
            
        return his
    
    @classmethod
    def clear_checkpoint(cls) -> None:
        with open(os.path.join(DATA_DIR, "accuracy_history.json"), 'w') as f:
            pass
        
        
    def render(self, show: bool=False) -> None:
        """Render accuracy graph of training history"""
        plt.figure()
        plt.plot(self._dates, self._acc_history)
        plt.xlabel(self.x_label)
        plt.ylabel("Accuracy")
        plt.title(self.title)
        plt.savefig(os.path.join(DATA_DIR, f"{self.fig_name}.png"))
        if show:
            plt.show()
            
    def __getitem__(self, index: int) -> float:
        return self._acc_history[index]

@dataclass
class LossHistory:
    x_label: str = field(default='Date')
    y_label: str = field(default='Loss')
    title: str = field(default='Loss Training History')
    fig_name: str = field(default='loss_training_history')
    _dates: list = field(default_factory=list, init=False)
    _loss_history: list = field(default_factory=list, init=False)
    _date: int = field(default=0, init=False)
        
    def add(self, gen_loss: float, est_loss: float) -> None:
        """Add training history"""
        self._dates.append(self._date)
        self._loss_history.append({
            'generator_loss': gen_loss,
            'estimator_loss': est_loss
        })
        self._date += 1
        
    def save(self) -> None:
        """Convert to dictionary"""
        data = {
            'loss_history': self._loss_history
        }
        with open(os.path.join(DATA_DIR, "loss_history.json"), 'w') as f:
            json.dump(data, f)
      
    @classmethod
    def from_checkpoint(cls) -> "LossHistory":
        try:
            with open(os.path.join(DATA_DIR, "loss_history.json"), 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {'loss_history': []}
            
        his = cls()
        for h in data['loss_history']:
            his.add(h['generator_loss'], h['estimator_loss'])
        
        return his
    
    @classmethod
    def clear_checkpoint(cls) -> None:
        with open(os.path.join(DATA_DIR, "loss_history.json"), 'w') as f:
            pass
        
    def render(self, show: bool=False) -> None:
        """Render loss graph of training history"""
        # Generator Loss
        plt.figure()
        plt.plot(
            self._dates, [h['generator_loss'] for h in self._loss_history], label='Generator Loss'
        )
        plt.xlabel(self.x_label)
        plt.ylabel("Loss")
        plt.title(self.title + " - Generator Loss")
        plt.legend()
        plt.savefig(os.path.join(DATA_DIR, f"{self.fig_name}_generator_loss.png"))
        if show:
            plt.show()

        # Estimator Loss
        plt.figure()
        plt.plot(
            self._dates, [h['estimator_loss'] for h in self._loss_history], label='Estimator Loss'
        )
        plt.xlabel(self.x_label)
        plt.ylabel("Loss")
        plt.title(self.title + " - Estimator Loss")
        plt.legend()
        plt.savefig(os.path.join(DATA_DIR, f"{self.fig_name}_estimator_loss.png"))
        if show:
            plt.show()

        
    def __getitem__(self, index: int) -> dict:
        return self._loss_history[index]
        
    def __len__(self) -> int:
        return len(self._loss_history)
