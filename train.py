from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def train_model(model, train_data, train_labels, val_data, val_labels, datagen):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Initial training phase
    history = model.fit(
        train_data, train_labels,
        epochs=50,
        batch_size=32,
        validation_data=(val_data, val_labels),
        callbacks=[early_stopping, reduce_lr]
    )

    # Check if it's a ResNet model
    if 'resnet' in model.name.lower():
        print("Fine-tuning ResNet model...")
        # Fine-tuning phase for ResNet model
        base_model = model.layers[0]
        base_model.trainable = True

        # Freeze first 100 layers
        for layer in base_model.layers[:100]:
            layer.trainable = False

        model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

        history_fine = model.fit(
            train_data, train_labels,
            epochs=50,
            batch_size=32,
            validation_data=(val_data, val_labels),
            callbacks=[early_stopping, reduce_lr]
        )

    return model