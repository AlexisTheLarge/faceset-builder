# Faceset builder

Faceset Builder is a command line tool for extracting faces from videos and photos.


[![Pyhton version][python-version]](https://www.python.org/) [![License][license]](LICENSE)


## Requirements

### Environment
 - Python 3.5+
 - CUDA 8+
 - cuDNN 5+

### Modules
 - face_recognition
 - opencv-python
 - numpy
 - scipy
 - Distance
 - tqdm
 - click


## Installation

Clone this repository

```sh
git clone https://github.com/AlexisTheLarge/faceset-builder.git faceset-builder
cd faceset_builder
```

Install python module and dependencies

```sh
python setup.py install
```

## Usage

Collect faces from src data. The output directory will have faces sorted into directories for easier manual cleaning.

    faceset-builder collect <input_dir> <reference_faces_dir> <output_dir>

Compile cleaned faceset to flat directory with sequential naming.

    faceset-builder compile <input_dir> <output_dir>

Supported options for the ```collect``` command:

    Usage: faceset-builder collect [OPTIONS] SOURCE_DIR REFERENCE_DIR OUTPUT_DIR

      Go through video files and photographs in OUTPUT_DIR and extract faces
      matching those found in REFERENCE_DIR. Extracted faces will be placed in
      OUTPUT_DIR sorted into folders corresponding to their source.
    
    Options:
      -t, --tolerance FLOAT          Threshold for face comparison. Default is 0.5.
 
      --min-face-size INTEGER        Minimum size in pixels for faces to extract.
                                     Default is 256.

      --crop-size INTEGER            Cropped images larger than this will be
                                     scaled down. Must be at least 1.5 times the
                                     size of --min-face-size. Default is 512.

      --one-face                     Discard any cropped images containing more
                                     than one face.

      --mask-faces                   Attempt to black out unwanted faces.

      --luminosity-range INTEGER...  Range from 0-255 for acceptable face
                                     brightness. Default is 10-245.

      --scan-rate FLOAT              Number of frames per second to scan for
                                     reference faces. Default is 0.2 fps (1 frame
                                     every 5 seconds).

      --capture-rate FLOAT           Number of frames per second to extract when
                                     reference face has been found. Default is 5.
 
      --sample-height INTEGER        Height in pixel to downsample the image/frame
                                     to before running facial recognition. Default
                                     is 500. (AFFECTS VRAM)
 
      --batch-size INTEGER           Number of video frames to run facial
                                     recognition on per batch. Default is 32.
                                     (AFFECTS VRAM)
 
      --buffer-size INTEGER          Number of frames to buffer between each scan.
                                     Default is the number of frames in one scan
                                     interval. (AFFECTS RAM)

      -h, --help                     Show this message and exit.


# License

Faceset Builder is licensed under the Mozilla Public License Version 2.0 (see [LICENSE](LICENSE))

[python-version]: <https://img.shields.io/badge/python-3.5%2B-2b5b84.svg>
[license]: <https://img.shields.io/github/license/AlexisTheLarge/faceset-builder.svg>


