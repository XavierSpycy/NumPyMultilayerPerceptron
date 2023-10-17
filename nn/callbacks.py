class EarlyStopping(object):
    def __init__(self, validation_set, criterion, min_delta, patience, step_size=5, mode='min', restore_best_weights=False, start_from_epoch=0):
        assert mode in ['min', 'max'], "Mode must be either 'min' or 'max'"
        self.validation_set = validation_set
        self.criterion = criterion
        self.min_delta = min_delta
        self.patience = patience
        self.step_size = step_size
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch

        self.current_iters = 0
        self.counter = 0
        self.best_weights = None
        if self.mode == 'min':
            self.best_score = float('inf')
        elif self.mode == 'max':
            self.best_score = -float('inf')
    
    def check_start(self):
        self.current_iters += 1
        if (self.current_iters - 1) < self.start_from_epoch:
            return False
        elif (self.current_iters - self.start_from_epoch - 1) % self.step_size != 0:
            return False
        else:
            return True
        
    def callback(self, model):
        if not self.check_start():
            return False
        
        X_val = self.validation_set[0]
        y_val = self.validation_set[1]
        y_pred = model.predict(X_val)
        current_score = self.criterion(y_val, y_pred)
        if self.mode == 'min':
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = model.get_weights()
            else:
                self.counter += 1
        elif self.mode == 'max':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = model.get_weights()
            else:
                self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.set_weights(self.best_weights)
            print(f"EarlyStopping triggered after {self.current_iters} iterations with best score {self.best_score:.4f}")
            return True
        return False