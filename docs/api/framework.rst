Training Framework Module
=========================

RecIS's training framework module provides comprehensive model training, evaluation, and management capabilities, simplifying the development workflow for deep learning models.

Core Components
---------------

.. currentmodule:: recis.framework.trainer

TrainingArguments
~~~~~~~~~~~~~~~~~~

.. autoclass:: TrainingArguments

Trainer
~~~~~~~

.. autoclass:: Trainer
   :members: __init__, add_hook, add_hooks, train, evaluate, restore

Saver
~~~~~~~~~~~~~~~~~

.. currentmodule:: recis.framework.checkpoint_manager

.. autoclass:: Saver
   :members: __init__, register_for_checkpointing, save, load, load_by_config

CheckpointManager
~~~~~~~~~~~~~~~~~

.. currentmodule:: recis.framework.checkpoint_manager

.. autoclass:: CheckpointManager
   :members: __init__, save, restore

**Example Usage:**

.. code-block:: python

   from recis.framework.checkpoint_manager import CheckpointManager, Saver
   
   # Create Saver
   saver = Saver(
         model,
         sparse_optimizer,
         output_dir=output_dir,
         max_keep=2,
         concurrency=2,
      )
   # Create checkpoint manager
   checkpoint_manager = CheckpointManager(
         saver,
         save_interval=1000
   )
   
   # Save checkpoint
   checkpoint_manager.save()
   
   # Load latest checkpoint
   checkpoint = checkpoint_manager.restore()

Advanced Usage
--------------

**Custom Training Pipeline**

.. code-block:: python

    from framework.trainer import Trainer
    class MyTrainer(Trainer):
         def _train_step(self, data, epoch, metrics):
            self.dense_optimizer.zero_grad()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.zero_grad()
            loss = self.model(data)
            metrics.update(epoch=epoch)
            metrics.update(loss=loss)
            metrics.update(get_global_metrics())
            loss.backward()
            self.dense_optimizer.step()
            if self.sparse_optimizer is not None:
                self.sparse_optimizer.step()
            if self.dense_lr_scheduler is not None:
                self.dense_lr_scheduler.step()

**Gradient Accumulation Training**

.. code-block:: python

   # Configure gradient accumulation
   training_args = TrainingArguments(
       output_dir="./output",
       train_steps=10000,
       gradient_accumulation_steps=8,  # Accumulate 8 steps before update
       log_steps=100
   )
   
   # Trainer will automatically handle gradient accumulation
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset
   )
