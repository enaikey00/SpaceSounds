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

 ## Improvements
 * Audio augmentation
 * More data
 * Mel Spectrograms Normalization
 * More epochs
 * ...
