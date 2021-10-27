# Tips of common data tools on M1 Macbook Air

Long story short, the M1 Macbook Air is so sweet and it is friendly to data modeling work with `python`.

## First, homebrew and miniforge

The easiest start comes from `Homebrew` by following the official installation at <https://brew.sh/>

The next step is `conda` via <https://github.com/conda-forge/miniforge>. Please make sure to choose `Miniforge3-MacOSX-arm64`. It runs like a charm.

## xgboost

`xgboost` comes with native M1 support and the best practice as tody is compiling via `pip` with the `xcode`'s `clang`. Some missing libs can be either install via `brew` or use `conda`'s `numpy` and `scipy`:

```
brew install cmake
conda create -n xgboost_env python=3.9
conda install numpy scipy 
pip install xgboost
```

The `conda` installed `numpy` and `scipy` can bring in `llvm-openmp-12.0.1` which is the trick.

## `neo4j`

`neo4j` runs fine on M1 and one can download it from <https://neo4j.com/download-center/>. I am not sure if it is M1 native, but seems no problem. The only pitfall comes from the `java` version and I have this best practice to install Java 11 <https://www.azul.com/downloads/?os=macos&architecture=arm-64-bit&package=jdk>

By any chance one needs to uninstall another java runtime to resolve the conflict, please follow <https://docs.oracle.com/en/java/javase/16/install/installation-jdk-macos.html>

## `pytorch` and `tensorflow`

`pytorch` has no surprise of no native M1 support as today, so the official installation works fine.

`tensorflow` has some luck from Apple. By following this <https://developer.apple.com/metal/tensorflow-plugin/>, one can use the M1 chip's GPU via `Metal`. The benchmark speed is not bad.
