class EarlyStopping(object):
    def __init__(self, criterion, min_delta=0.0, patience=5, mode='min', restore_best_weights=False, start_from_epoch=0) -> None:
        self.criterion = criterion
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        if self.mode == 'min':
            self.prev_value = float('inf')
        elif self.mode == 'max':
            self.prev_value = float('-inf')
        self.counter = 0
        self.step = 0
        self.save_best = False

    def check_improvement(self, curr_value):
        if self.mode == 'min':
            return curr_value < self.prev_value - self.min_delta
        elif self.mode == 'max':
            return curr_value > self.prev_value + self.min_delta
    
    def __call__(self, y_val, y_pred):
        self.step += 1
        if self.step <= (self.start_from_epoch + 1):
            return False, False
        
        curr_value = self.criterion(y_val, y_pred)

        if self.check_improvement(curr_value):
            self.counter = 0
            self.prev_value = curr_value
            if self.restore_best_weights:
                self.save_best = True
        else:
            self.counter += 1
        return self.counter >= self.patience, self.save_best