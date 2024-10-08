import tensorflow as tf
from transformers import create_optimizer
from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, train_data, train_labels, val_data, val_labels, callbacks):
    try:
        built_model = model.build_model()
        
        # Build the model by calling it on a batch of data
        dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        _ = built_model(dummy_input, training=False)
        
        # Print model summary
        built_model.summary()
        
        # Create optimizer
        num_train_steps = len(train_data) // 32 * 50  # assuming batch_size=32 and epochs=50
        optimizer, lr_schedule = create_optimizer(
            init_lr=2e-5,
            num_train_steps=num_train_steps,
            num_warmup_steps=0,
            weight_decay_rate=0.01,
        )

        if callbacks is None:
            callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]

        # Preprocess data using the feature extractor
        train_data = model.preprocess_input(train_data)['pixel_values']
        val_data = model.preprocess_input(val_data)['pixel_values']

        # Convert labels to TensorFlow tensors
        train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
        val_labels = tf.convert_to_tensor(val_labels, dtype=tf.int32)

        # Print shapes after conversion
        print(f"Train data shape: {train_data.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Validation labels shape: {val_labels.shape}")

        # Custom training loop
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(32)

        for epoch in range(50):  # 50 epochs
            print(f"Epoch {epoch + 1}/{50}")
            for step, (batch_images, batch_labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = built_model(batch_images, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        batch_labels, outputs.logits, from_logits=True
                    )
                    loss = tf.reduce_mean(loss)
                
                grads = tape.gradient(loss, built_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, built_model.trainable_variables))

                if step % 50 == 0:
                    print(f"Step {step}, Loss: {loss:.4f}")

            # Validation
            val_loss = 0
            val_accuracy = 0
            for val_images, val_labels in val_dataset:
                val_outputs = built_model(val_images, training=False)
                val_loss += tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        val_labels, val_outputs.logits, from_logits=True
                    )
                )
                val_accuracy += tf.reduce_mean(
                    tf.keras.metrics.sparse_categorical_accuracy(val_labels, val_outputs.logits)
                )
            
            val_loss /= len(val_dataset)
            val_accuracy /= len(val_dataset)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        return built_model

    except Exception as e:
        print(f"An error occurred during model training: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        raise