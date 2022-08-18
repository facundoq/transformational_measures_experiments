from poutyne import  Callback
from tqdm.auto import tqdm

class TotalProgressCallback(Callback):

    #TODO make nested
    
    '''
    Provides a callback that shows the total training progression in a single progress bar for the entire training, compared with a progress bar for each epoch as in poutyne.ProgressCallback.
    
    '''
    def __init__(self):
        super().__init__()

    def set_params(self, params: dict):
        super().set_params(params)
        self.epochs = self.params['epochs']
        self.train_steps = self.params['steps']
        self.valid_steps = self.params['valid_steps']
        self.steps = self.train_steps+self.valid_steps

    def on_train_begin(self, logs: dict):
        self.val_logs={}
        self.test_logs={}

    def on_epoch_begin(self, epoch_number: int, logs:dict):
        if epoch_number==1:
            self.bar = tqdm(total=float(self.epochs),desc="Training",
             bar_format = "{desc}: {percentage:.1f}%|{bar}| {n:.2f}/{total_fmt} epochs [{elapsed}<{remaining}{postfix}]")

    def on_epoch_end(self, epoch_number: int, logs: dict):
        val_logs = {k:v for k,v in logs.items() if k.startswith("val")}
        self.val_logs = self.format_logs(val_logs)
        
        test_logs = {k:v for k,v in logs.items() if k.startswith("test")}
        self.test_logs = self.format_logs(test_logs)

    def format_logs(self,logs):
        ignored_keys =  ["batch","size","time","epoch"]
        for k in ignored_keys:
            if k in logs:
                del logs[k]
        logs = {k:f"{v:.3f}" for k,v in logs.items()}
        return logs

    def on_train_batch_end(self, batch_number: int, logs: dict):
        logs = self.format_logs(logs)
        logs.update(self.val_logs)
        logs.update(self.test_logs)
        self.bar.set_postfix(logs)
        self.bar.update(1/self.steps)
    def on_valid_batch_end(self, batch_number: int, logs: dict):
        self.bar.update(1/self.steps)

    def on_train_end(self, logs: dict):
        self.bar.close()