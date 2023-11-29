# SpaceSounds 🎵👾
Space sounds classification with a Deep Learning approach. This is my final project for class "Machine & Deep Learning for Vision & Multimedia".

## Dataset

Data was downloaded from NASA Soundcloud profile. 
Playlist [link](https://soundcloud.com/nasa/sets/solar-system-beyond-sounds)

`scdl -l https://soundcloud.com/nasa/sets/solar-system-beyond-sounds`

This downloads 27 audio files (.mp3, .wav).

I then split each audio file with this rule: 

`number of split = n_samples / sample_rate`

resulting in a total of 578 audio files.

### Dataset Splits
* Traning Set: 70%
* Validation Set: 15%
* Test Set: 15%

## Labeling

I labeled the original 27 files into 4 categories:
1. Spacecraft Signals
2. Planetary Sounds
3. Cosmic Phenomena
4. Earth's Magnetosphere

At split time the label was added to the file in following this fashion:

`<path_to_data_folder> / <id_from_split> _ <label> _ <file_name>.<extension>`

Example: ./data/trimmed/19_Cosmic Phenomena_Solar System & Beyond Sounds_Kepler Star KIC7671081B Light Curve Waves to Sound.mp3

## Architecture
![image](https://github.com/enaikey00/SpaceSounds/assets/64537810/81c58ad4-c56a-434d-9560-2b156b7c663e)

## Results
Training and Validation Loss

![Loss](https://github.com/enaikey00/SpaceSounds/assets/64537810/45872aec-4f82-4290-b828-8475e96434bb)

Training and Validation Accuracy

![Accuracy](https://github.com/enaikey00/SpaceSounds/assets/64537810/21d7b5be-4b40-418c-8c8d-a352626d016d)

Confusion Matrix on Test Set

![ConfusionMatrix](https://github.com/enaikey00/SpaceSounds/assets/64537810/73b89fcc-86c5-44e4-ae40-f51ddabb5ca0)



 ## Improvements
 * Audio augmentation
 * More data
 * Mel Spectrograms Normalization
 * More epochs
 * Square Spectrogram for Convolution (current shape: 128x256)
 * ...
