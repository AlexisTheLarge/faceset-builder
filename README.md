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

      --luminosity-range INTEGER...  Range from 0-255 for acceptable face
                                     brightness. Default is 10-245.

      --laplacian-threshold FLOAT    Threshold for blur detection, values below
                                     this will be considered blurry. 0 means all
                                     images pass, 10 might be a good place to
                                     start. This option is very inconsistent and
                                     is not recommended, use of the --save-invalid
                                     option is strongly recommended in
                                     conjunction. Default is 0.

      --one-face                     Discard any cropped images containing more
                                     than one face.

      --mask-faces                   Attempt to black out unwanted faces.

      --save-invalid                 Duplicates, corrupted files, and faces that
                                     fail validation will be saved to
                                     '<output_dir>/invalid'. Otherwise duplicates
                                     and corrupted files will be deleted, and
                                     invalid faces ignored.

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

      --greedy                       While scanning, consider face detected even
                                     if it's smaller than --min-face-size. Might
                                     capture a few more faces at a potential
                                     performance loss.

      -h, --help                     Show this message and exit.

### Blur Detection

This script provies an option for blur detection using Laplacian Variance (```--laplacian-tolerance```). This has been included because some people have expressed interest in such a feature. I can not personally recommend using said option, since I have had **very** unpredictable results when testing it. It should be considered an experimental feature.

## Troubleshooting

If the extraction process is slow (taking multiple seconds per photo), the problem is most likely that dlib isn't compiled with CUDA support. Install dlib from git (https://github.com/davisking/dlib) in order see more info during the installation about why CUDA support might not be working.


# License

Faceset Builder is licensed under the Mozilla Public License Version 2.0 (see [LICENSE](LICENSE))

[python-version]: <https://img.shields.io/badge/python-3.5%2B-2b5b84.svg>
[license]: <https://img.shields.io/github/license/AlexisTheLarge/faceset-builder.svg>


