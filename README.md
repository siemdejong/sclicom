<!-- This file incorporates work covered by the following copyright and permission notice:

    Copyright (c) 2021 Othneil Drew

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
""" -->

<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!-- [![Contributors][contributors-shield]][contributors-url] -->
<!-- [![Forks][forks-shield]][forks-url] -->
<!-- [![Stargazers][stars-shield]][stars-url] -->
<!-- [![Issues][issues-shield]][issues-url] -->
<!-- [![GNU License][license-shield]][license-url] -->
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/siemdejong/shg-strain-stress">
    <img src="images/skinstression-logo.png" alt="Logo" width="80" height="80">
  </a> -->


<h2 align="center">Deep Learning for Higher Harmonic Generation Microscopy</h2>
<h3 align="center">Deep learning utilities for higher harmonic generation microscopy images.</h3>

  <p align="center">
    This project is a deep learning application to classify various pediatric brain tumours from higher harmonic generation microscopy images.
    <br />
    <a href="https://siemdejong.github.io/dpat"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/siemdejong/dpat">View Demo</a>
    · -->
    <a href="https://github.com/siemdejong/dpat/issues">Report Bug</a>
    ·
    <a href="https://github.com/siemdejong/dpat/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <!-- <li><a href="#roadmap">Roadmap</a></li> -->
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
# About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

The project aims to do deep learning classification on higher harmonic generation (HHG) microscopy images of pediatric brain tumours.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Built With

[![Python][Python]][Python-url]
<!-- [![PyTorch][PyTorch]][Pytorch-url]
[![TensorFlow][TensorFlow]][TensorFlow-url]
[![Optuna][Optuna]][Optuna-url]
[![scikit-learn][scikit-learn]][scikit-learn-url]
[![Numpy][Numpy]][Numpy-url]
[![scipy][scipy]][scipy-url]
[![Hydra][Hydra]][Hydra-url]
[![Jupyter][Jupyter]][Jupyter-url]
[![tqdm][tqdm]][tqdm-url]
[![matplotlib][matplotlib]][matplotlib-url]
[![plotly][plotly]][plotly-url]
[![pyimq][pyimq]][pyimq-url] -->


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
# Getting Started

This section includes instructions on setting up the project locally.

## Prerequisites

### Conda
For package management, it is advised to use a conda package manager.
The author recommends [Miniforge](https://github.com/conda-forge/miniforge) or [Mambaforge](https://github.com/conda-forge/miniforge).

### vips
This project depends on [dlup](https://github.com/NKI-AI/dlup) (automatically installed), which depends on vips.
On Windows, vips needs to be installed locally.
Download the latest [libvips](https://github.com/libvips/libvips/releases) Windows binary and unzip somewhere.

### OpenSlide
Vips comes with OpenSlide.
It is not needed to install OpenSlide separately.

<!-- #### CUDA
This project expects CUDA enabled GPUs.
Run `nvidia-smi` to see if there are CUDA enabled GPUs available. -->


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation
Run the following commands from a conda enabled shell (such as Miniforge Prompt, if Miniforge/Mambaforge is installed).

1.  Clone this repository and change directories
    ```
    git clone https://github.com/siemdejong/dpat.git dpat && cd dpat
    ```
1.  Create a new conda environment and activate it.
    ```
    conda create -n <env_name>
    conda activate <env_name>
    ```
1.  Install dependencies from `environment.yml`.
    ```
    conda env update -f environment.yml
    ```
1.  Make sure libvips is available, see <a href="#prerequisites">Prerequisites</a>.
1.  Change `PATHS.vips` in `config.yml` to point to `vips/bin`:
    ```yaml
    # config.yml
    PATHS:
      vips: path/to/vips/bin
    ```
1.  Install dpat in editable mode with
    ```
    pip install -e .
    ```
1.  Verify installation
    ```
    python -c "import dpat"
    ```
<!-- 1.  Check if CUDA is available for the installed Pytorch distribution.
    In a Python shell, execute
    ```python
    import torch
    torch.cuda.is_available()
    ```
    If `false`, install Pytorch following its documentation. -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Converting images
To convert all images from directory INPUT_DIR, and output the images as TIFF in OUTPUT_DIR, run
```
dpat convert batch -i INPUT_DIR -o OUTPUT_DIR -e tiff
```
Large images need to be trusted against decompression bomb DOS attack.
Use the `--trust` flag.
To skip images that were already converted to the target extension, use `--skip-existing`.

NOTE: If converting to tiff, the input images are assumed to contain the reference to the scanning program, which must be in {200slow, 300slow, 300fast}.

```
Usage: dpat convert batch [OPTIONS]

Options:
  -i, --input-dir TEXT         Input directory where to find the images to be
                               converted.  [default: .]
  -o, --output-dir TEXT        Output directory where place converted files.
                               [default: ./converted]
  -e, --output-ext [tiff|tif]  Extension to convert to.  [required]
  -w, --num-workers INTEGER    Number of workers that convert the images in
                               parallel.  [default: 4]
  -c, --chunks INTEGER         Number of chunks distributed to every worker.
                               [default: 30]
  --trust                      Trust the source of the images.
  --skip-existing              Skip existing output files.
  --help                       Show this message and exit.
```

### Creating splits
To create train-val-test splits linking paths of images to splits with IMAGE_DIR, output the splits to OUTPUT_DIR, with labels PATH_TO_LABELS_FILE, and dataset name NAME run
```
dpat splits create -i IMAGE_DIR -o OUTPUT_DIR -l PATH_TO_LABELS_FILE -n NAME
```

To filter diagnoses that exactly match diseases, use e.g. `-f medulloblastoma -f "pilocytic astrocytoma"`.
To filter filenames that match certain values, use a glob pattern.
E.g. `-y *slow.tiff` to only include images ending with `slow.tiff`.
To exclude filenames that match certaine values, use a glob pattern with `-x`.
Exclusion is performed on the set specified by inclusion.

```
Usage: dpat splits create [OPTIONS]

Options:
  -i, --input-dir TEXT   Input directory where to find the images.  [required]
  -l, --labels TEXT      Path to labels file.  [required]
  -n, --name TEXT        Name of dataset.  [required]
  -o, --output-dir TEXT  Directory where to put the splits.  [default: splits]
  --overwrite            Overwrite folds in output dir, if available.
  -y, --include TEXT     Glob pattern to include files from `input-dir`
                         [default: *.*]
  -x, --exclude TEXT     Glob pattern to exclue files from `input-dir`,
                         included with `--include`
  -f, --filter TEXT      Filter a diagnosis. For multiple diagnoses, use `-f 1
                         -f 2`.
  --help                 Show this message and exit.
```

### Logging
When using the package as a library, if needed, logging can be turned off with
```python
logging.getLogger('dpat').propagate = False
```

<!-- _For more examples, please refer to the [documentation](https://siemdejong.github.io/shg-strain-stress)._ -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ## Documentation
Documentation is hosted by Github Pages.
Docs are automatically created from docstrings.
Subsequently run
```sh
cd docs
sphinx-apidoc -o src ../src
make html
```
to update the documentation. -->

<!-- ROADMAP -->
<!-- ## Roadmap

- [x] Hyperparameter optimization with Optuna
- [x] Model training with Pytorch
- [ ] Inference
- [ ] Explainable AI
- [ ] Documentation

See the [open issues](https://github.com/siemdejong/shg-strain-stress/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
## Contributing
Contribute using the following steps.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ETYMOLOGY -->
<!-- ## Etymology
Skinstression is a combination of skin, stress and regression.
Second-harmonic generation images and their corresponding stress curves form the basis of the regression task. -->


<!-- LICENSE -->
## License

Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Siem de Jong - [linkedin.com/in/siemdejong](https://www.linkedin.com/in/siemdejong/) - siem.dejong@hotmail.nl

Project Link: [https://github.com/siemdejong/dpat](https://github.com/siemdejong/dpat)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/siemdejong/shg-strain-stress.svg?style=for-the-badge
[contributors-url]: https://github.com/siemdejong/shg-strain-stress/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/siemdejong/shg-strain-stress.svg?style=for-the-badge
[forks-url]: https://github.com/siemdejong/shg-strain-stress/network/members
[stars-shield]: https://img.shields.io/github/stars/siemdejong/shg-strain-stress.svg?style=for-the-badge
[stars-url]: https://github.com/siemdejong/shg-strain-stress/stargazers
[issues-shield]: https://img.shields.io/github/issues/siemdejong/shg-strain-stress.svg?style=for-the-badge
[issues-url]: https://github.com/siemdejong/shg-strain-stress/issues
[license-shield]: https://img.shields.io/github/license/siemdejong/shg-strain-stress.svg?style=for-the-badge
[license-url]: https://github.com/siemdejong/shg-strain-stress/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/siemdejong
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://www.tensorflow.org/
[pyimq]: https://img.shields.io/badge/pyimq-1689a0?style=for-the-badge&logo=pyimq&logoColor=white
[pyimq-url]: https://github.com/sakoho81/pyimagequalityranking
[TensorFlow]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white
[TensorFlow-url]: https://pytorch.org
[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org
[Numpy]: https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org/index.html
[Optuna]: https://img.shields.io/badge/-Optuna-483D8B?style=for-the-badge&logo=optuna&logoColor=white
[Optuna-url]: https://optuna.org/
[scikit-learn]: https://img.shields.io/badge/scikit%20learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/index.html
[Hydra]: https://img.shields.io/badge/hydra-87CEEB?style=for-the-badge&logo=hydra&logoColor=white
[Hydra-url]: https://hydra.cc/docs/intro/
[Jupyter]: https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white
[Jupyter-url]: https://jupyter.org/
[tqdm]: https://img.shields.io/badge/tqdm-FFC107?style=for-the-badge&logo=tqdm&logoColor=white
[tqdm-url]: https://tqdm.github.io/
[matplotlib]: https://img.shields.io/badge/matplotlib-white?style=for-the-badge&logo=matplotlib&logoColor=white
[matplotlib-url]: https://matplotlib.org/
[plotly]: https://img.shields.io/badge/plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white
[plotly-url]: https://plotly.com/python/
[scipy]: https://img.shields.io/badge/scipy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white
[scipy-url]: https://scipy.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
