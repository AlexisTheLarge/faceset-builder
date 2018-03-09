from distutils.core import setup

setup(
    # Application name:
    name="Faceset builder",

    # Version number (initial):
    version="1.0.0",

    # Application author details:
    author="Alexis_TheLarge",

    # Packages
    packages=["faceset_builder", "faceset_builder.face_collector"],

    # Include additional files into the package
    include_package_data=True,


    scripts=['bin/faceset-builder'],


    #
    # license="LICENSE.txt",
    description="Extract faces from videos and photos",

    # long_description=open("README.txt").read(),

    # Dependent packages (distributions)
    install_requires=[
        "click",
        "opencv_python",
        "tqdm",
        "face_recognition",
        "numpy",
        "scipy"
    ],
)
