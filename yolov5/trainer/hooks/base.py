class BaseHook:

    def before_train(self, trainer):
        pass

    def after_train(self, trainer):
        pass

    def before_epoch(self, trainer):
        pass

    def after_epoch(self, trainer):
        pass

    def before_batch(self, trainer):
        pass

    def after_batch(self, trainer):
        pass
