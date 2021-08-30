# lgvRouting

lgv routing project for algoritmi di ottimizzazione.

## Build

```
mkdir build
cd build
cmake ..
make -j8
```
## test
Tests are made with catch2 library, the lgvRouting_test check if all works fine and all library are installed in a correct way

## Samples
- lgvRouting_import: sample for reading dataset
- lgvRouting_launch: sample for launch a method on a given dataset
- lgvRouting_stats: application that create a performance report about all methods


## Docs

dataset source [link](https://neo.lcc.uma.es/vrp/)

```
doxygen doc/Doxyfile
```
Documentation will be created in doc folder, search index.html in html folder

## Third party software

| Repository | Author | LICENSE |
|------------|--------|---------|
|[imgui](https://github.com/ocornut/imgui)| Omar Cornut | MIT|
|[implot](https://github.com/epezent/implot)| Evan Pezent | MIT|
|[Argh!](https://github.com/adishavit/argh)| Adi Shavit| BSD-3 |


## Authors

* **Luca Bartoli** - *Main developer* - [lbartoli](https://github.com/lucabart97)
* **Massimiliano Bosi** - *Main developer* - [mbosi](https://github.com/FisherTiger95)