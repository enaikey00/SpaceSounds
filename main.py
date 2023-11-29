#!/usr/bin/env python
# coding: utf-8

# --- Libs & Functions
import librosa
import librosa.display
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf # writing audio files
import pandas as pd
import tensorflow as tf
from tqdm import tqdm # progress bar
import sox # trimming audio files
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import CategoricalCrossentropy

def inspect_audio(audio_file: str):
    # load audio file
    audio_file = audio_file
    y, sr = librosa.load(audio_file)
    
    # display waveform
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    
    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.show()


# Function to list all files in a directory
def list_files(directory):
    file_list = []
    if os.path.exists(directory) and os.path.isdir(directory):
        files = os.listdir(directory)
        if files:
            print(f"Files in directory '{directory}':")
            for file in files:
                print(file)
                file_list.append(file)
            return file_list

        else:
            print(f"No files found in directory '{directory}'")
    else:
        print(f"Directory '{directory}' does not exist or is not a valid directory.")


def amplitudes_ranges(directory_path: str, file_list: list()):
    for file in file_list:
        y, sr = librosa.load(directory_path + file)
        print("Amplitudes")
        print("Min: ", np.min(y), "Max:", np.max(y))


def normalize_audio(directory_path: str, file_list: list(), output_dir: str):
    for file in file_list:
        print(file)
        y, sr = librosa.load(directory_path + file)
        normalized_audio = librosa.util.normalize(y, norm=2) # options: norm = None or norm = 2
        # also see: subtype='PCM_24
        sf.write(file = output_dir + file, data = normalized_audio, samplerate = sr)

def split_audio(file_list):
    directory_path = './data/raw/Solar System & Beyond Sounds/'
    output_path = './data/trimmed/'
    for file in file_list:
        print("Splitting: ", file)
        file_path = directory_path + file
        sample_rate = sox.file_info.sample_rate(file_path)
        n_samples = sox.file_info.num_samples(file_path)
        # how many splits?
        n_split = int(n_samples / sample_rate)
        print("Number of splits: ", n_split)
        #for i in tqdm(range(0, n_samples, n_split)): # more than 48k files for first audio! (too much time)
        for i in range(0, n_split):
            tfm = sox.Transformer()
            # trim the audio between every n_split seconds.
            tfm.trim(i, i+sample_rate)
            # attach label to file
            label = label_mapping.get(file)
            if label:
                output_file = output_path + str(i) + "_" + label + "_" + file
                tfm.build_file(file_path, output_file)
                print("Wrote: ", output_file)
            else:
                print("Label not found.")

# load audio, compute Mel spectrogram, and generate features and labels
def load_and_process_audio(audio_file, label, max_length):
    directory_path = './data/trimmed/'
    y, sr = librosa.load(directory_path + audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)  # Convert to dB scale

    # pad or trim spectrogram to a fixed length (features must have equal size)
    if mel_spectrogram_db.shape[1] < max_length:
        pad_width = max_length - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spectrogram_db = mel_spectrogram_db[:, :max_length]

    return mel_spectrogram_db, label


# label mapping for audio files (original list)
label_mapping = {
    'Solar System & Beyond Sounds_Audio of Juno’s Ganymede Flyby.wav': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Cassini Enceladus Sound.mp3': 'Planetary Sounds',
    'Solar System & Beyond Sounds_Cassini Saturn Radio Emissions #1.mp3': 'Planetary Sounds',
    'Solar System & Beyond Sounds_Cassini Saturn Radio Emissions #2.mp3': 'Planetary Sounds',
    'Solar System & Beyond Sounds_Chorus Radio Waves within Earth\'s Atmosphere.mp3': 'Earth\'s Magnetosphere',
    'Solar System & Beyond Sounds_First Likely Marsquake Heard by NASA\'s InSight.mp3': 'Planetary Sounds',
    'Solar System & Beyond Sounds_Juno Morse code HI received from Earth.mp3': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Kepler Star KIC12268220C Light Curve Waves to Sound.mp3': 'Cosmic Phenomena',
    'Solar System & Beyond Sounds_Kepler Star KIC7671081B Light Curve Waves to Sound.mp3': 'Cosmic Phenomena',
    'Solar System & Beyond Sounds_LCROSS Water on the Moon Song.mp3': 'Planetary Sounds',
    'Solar System & Beyond Sounds_Parker Solar Probe - Langmuir Waves.mp3': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Parker Solar Probe - Whistler Mode Waves 1.mp3': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Parker Solar Probe - Whistler Mode Waves 2.mp3': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Plasmaspheric Hiss.wav': 'Cosmic Phenomena',
    'Solar System & Beyond Sounds_Plasmawaves - Chorus.mp3': 'Earth\'s Magnetosphere',
    'Solar System & Beyond Sounds_Quindar Sound #1.mp3': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Quindar Sound #2.mp3': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Sonification of a Hubble Deep Space Image.mp3': 'Cosmic Phenomena',
    'Solar System & Beyond Sounds_Sounds of Earth\'s Magnetic Drum in Space.mp3': 'Earth\'s Magnetosphere',
    'Solar System & Beyond Sounds_Sounds of Saturn Hear Radio Emissions of the Planet and Its Moon Enceladus.mp3': 'Planetary Sounds',
    'Solar System & Beyond Sounds_Sputnik Beep.mp3': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Stardust Passing Comet Tempel 1.mp3': 'Planetary Sounds',
    'Solar System & Beyond Sounds_Sun Sonification.wav': 'Cosmic Phenomena',
    'Solar System & Beyond Sounds_Voyager 1 Three Tsunami Waves in Interstellar Space.mp3': 'Cosmic Phenomena',
    'Solar System & Beyond Sounds_Voyager Interstellar Plasma Sounds.mp3': 'Spacecraft Signals',
    'Solar System & Beyond Sounds_Voyager Lightning on Jupiter.mp3': 'Planetary Sounds',
    'Solar System & Beyond Sounds_Whistler Waves.mp3': 'Cosmic Phenomena'
}


if __name__ == '__main__':

    # list files
    directory_path = './data/raw/Solar System & Beyond Sounds/'
    file_list = list_files(directory_path)

    inspect_audio(directory_path + file_list[0])

    # Controllo intervalli ampiezze
    amplitudes_ranges(directory_path, file_list)
    # Nota
    # Gli intervalli sono quasi tutti in (-1, 1), tuttavia almeno un file va oltre l'intervallo -> normalizzazione

    # Normalizzazione (nota: da approfrondire)
    directory_path = './data/raw/Solar System & Beyond Sounds/'
    normalize_audio(directory_path, file_list, "./data/normalized/")
    # Verifica
    amplitudes_ranges('./data/normalized/', file_list)

    split_audio(file_list) # generates 578 files

    # Distribuzione classi (file di origine)
    df = pd.DataFrame(data = label_mapping.items(), columns = ["file", "label"])
    print(df["label"].value_counts())


    # create TensorFlow dataset
    spectrograms = []
    labels = []

    for audio_file in tqdm(audio_files):
        label = audio_file.split("_")[1]
        if label:
            max_length = 256 # first spectrogram.shape[1]
            spectrogram, label = load_and_process_audio(audio_file, label, max_length)
            print(spectrogram.shape)
            spectrograms.append(spectrogram)
            labels.append(label)


    print("Spectrogram shape", spectrograms[0].shape)

    # lists to TensorFlow tensors
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)

    # convert labels to numerical IDs
    label_to_id = {label: i for i, label in enumerate(set(labels))}
    numeric_labels = [label_to_id[label] for label in labels]
    # one-hot numeric labels
    one_hot_labels = tf.one_hot(numeric_labels, depth=len(set(labels)))

    # TensorFlow dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, one_hot_labels))


    # --- Dataset Split

    # training, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    # shuffle and batch the datasets
    batch_size = 16
    train_dataset = train_dataset.shuffle(buffer_size=train_size).batch(batch_size, drop_remainder=True)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)



    # Inspect dataset
    for spectrogram, label in train_dataset.take(3):
        print(f"Spectrogram shape: {spectrogram.shape}, Label: {label}")


    # information about dataset elements
    print(dataset.element_spec)


    # --- Model
    num_classes = 4

    # CNN
    model = models.Sequential([
        layers.Input(shape=(spectrogram.shape[1], spectrogram.shape[2], 1) ),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'), 
        layers.MaxPooling2D((2, 2), padding ='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding ='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    print(model.summary())



    # Compile
    model.compile(optimizer='adam',
                  loss=CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train
    history = model.fit(train_dataset, epochs=5, initial_epoch=0, validation_data=val_dataset)


    # --- Charts

    # Plot Loss
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history.history["accuracy"], 'b', label='Training')
    plt.plot(epochs, history.history["val_accuracy"], 'r', label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # confusion matrix
    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.concatenate([y for x, y in test_dataset], axis=0)

    # convert true_labels and predicted_labels to class indices
    true_labels = np.argmax(true_labels, axis=1)
    predicted_labels = np.argmax(predictions, axis=1)

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    class_names = ["Spacecraft Signals","Planetary Sounds", "Cosmic Phenomena", "Earth's Magnetosphere"]

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # evaluation on test dataset
    print(model.evaluate(test_dataset))