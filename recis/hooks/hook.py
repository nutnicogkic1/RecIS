class Hook:
    """Base class for all hooks in the RecIS training system.

    Hooks provide a way to extend the training process by defining callback methods
    that are called at specific points during training, evaluation, and execution.
    All custom hooks should inherit from this base class and override the relevant
    callback methods.

    The hook system supports the following callback points:
    - Training lifecycle: before_train, after_train
    - Evaluation lifecycle: before_evaluate, after_evaluate
    - Epoch lifecycle: before_epoch, after_epoch
    - Step lifecycle: before_step, after_step
    - Cleanup: end

    Example:
        Creating a custom hook:

        >>> class CustomHook(Hook):
        ...     def __init__(self, custom_param):
        ...         self.custom_param = custom_param
        ...
        ...     def before_train(self):
        ...         print(f"Training started with {self.custom_param}")
        ...
        ...     def after_step(self):
        ...         # Custom logic after each training step
        ...         pass
        >>> # Use the custom hook
        >>> custom_hook = CustomHook("my_parameter")
        >>> trainer.add_hook(custom_hook)
    """

    def before_train(self):
        """Called before training starts.

        This method is invoked once at the beginning of the training process,
        before any training steps are executed. Use this for initialization
        logic that needs to happen before training begins.
        """

    def after_train(self):
        """Called after training completes.

        This method is invoked once at the end of the training process,
        after all training steps have been completed. Use this for cleanup
        or final processing logic.
        """

    def before_evaluate(self):
        """Called before evaluation starts.

        This method is invoked before each evaluation phase begins.
        Use this for setup logic specific to evaluation.
        """

    def after_evaluate(self):
        """Called after evaluation completes.

        This method is invoked after each evaluation phase ends.
        Use this for processing evaluation results or cleanup.
        """

    def before_epoch(self):
        """Called before each training epoch starts.

        This method is invoked at the beginning of each training epoch,
        before any steps in that epoch are executed.
        """

    def after_epoch(self):
        """Called after each training epoch completes.

        This method is invoked at the end of each training epoch,
        after all steps in that epoch have been executed.
        """

    def before_step(self):
        """Called before each training step.

        This method is invoked before each individual training step
        is executed. Use this for per-step setup logic.
        """

    def after_step(self):
        """Called after each training step completes.

        This method is invoked after each individual training step
        has been executed. Use this for per-step processing logic,
        such as logging metrics or updating statistics.
        """

    def end(self):
        """Called at the very end of the training process.

        This method is invoked for final cleanup operations,
        such as closing files, finalizing logs, or releasing resources.
        It is called after all other hook methods have completed.
        """
