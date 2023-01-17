<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">March Madness Predictor</h3>

  <p align="center">
    Use machine learning to fill out your bracket
    <br />
    <a href="https://github.com/ajbrumleve/march_madness/issues">Report Bug</a>
    ·
    <a href="https://github.com/ajbrumleve/march_madness">Request Feature</a>
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
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This has been an ongoing project over the years. A few years ago I started working with code I found at https://github.com/adeshpande3/March-Madness-ML. Each year I slightly adapt this code a little more and tweak the features or the model used. I finally decided to clean up the Frankensteins monster the code had become, add some documentation and a basic GUI, and put it out there.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Python][Python]][Python-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

Require packages dill, wxPython.

Use `pip` to install the packages from PyPI:

```bash
pip install dill
pip install wxPython
pip install pandas
pip install scikit-learn
pip install numpy
pip install matplotlib
pip install utils
```


### Installation



1. Download and unzip [this entire repository from GitHub](https://github.com/ajbrumleve/march_madness), either interactively, or by entering the following in your Terminal.
    ```bash
    git clone https://github.com/ajbrumleve/march_madness.git
    ```
2. Navigate into the top directory of the repo on your machine
    ```bash
    cd march_madness
    ```
3. Create a virtualenv and install the package dependencies. If you don't have `pipenv`, you can follow instructions [here](https://pipenv.pypa.io/en/latest/install/) for how to install.
    ```bash
    pipenv install
    ```
4. Run `main.py` to try prebuilt model. Or run `march_madness.py` to train new model. This can take a long time but feel free to change the flags in the build_model method as described in the Usage section.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

As currently configured, running main.py will load a pretrained model and open an interface to get the odds of two teams. Please note that as of now, these predictions will only work with teams who made the tournment in 2022. If you want to train your own model, run march_madness.py. By default, this runs classifier = data_obj.build_model(False,2003,2022,True,True,False). This builds the training data from all data between 2003 and 2022. It uses recursive feature elimination to choose features and applies a grid search to try and optimize the Random Forest model. These are automated but can be turned off with the parameters in the build_model method. When the program finishes, the model will be saved and accessed by the gui.py file. When you are ready to create a Kaggle submission file, there is a button in the interface which will generate the file based on your model.


<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Andrew Brumleve - [@AndrewBrumleve](https://twitter.com/AndrewBrumleve) - ajbrumleve@gmail.com

Project Link: [https://github.com/ajbrumleve/march_madness](https://github.com/ajbrumleve/march_madness)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [march-madness-ml](https://github.com/adeshpande3/March-Madness-ML)
* [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/ajbrumleve/march_madness.svg?style=for-the-badge
[contributors-url]: https://github.com/ajbrumleve/march_madness/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ajbrumleve/march_madness.svg?style=for-the-badge
[forks-url]: https://github.com/ajbrumleve/march_madness/network/members
[stars-shield]: https://img.shields.io/github/stars/ajbrumleve/march_madness.svg?style=for-the-badge
[stars-url]: https://github.com/ajbrumleve/march_madness/stargazers
[issues-shield]: https://img.shields.io/github/issues/ajbrumleve/march_madness.svg?style=for-the-badge
[issues-url]: https://github.com/ajbrumleve/march_madness/issues
[license-shield]: https://img.shields.io/github/license/ajbrumleve/march_madness.svg?style=for-the-badge
[license-url]: https://github.com/ajbrumleve/march_madness/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: (https://www.linkedin.com/in/andrew-brumleve-574239227/)
[product-screenshot]: images/screenshot.png
[Python]:  	https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org/
