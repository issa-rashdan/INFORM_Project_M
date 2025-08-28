class EarlyStopping():
    def __init__(self, patience = 10, verbose = False):
        self.patience = patience 
        self.verbose = verbose
        self.early_stop = False
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping  counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
            

                
