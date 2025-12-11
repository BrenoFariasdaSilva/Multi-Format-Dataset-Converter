<div align="center">
  
# [Multi-Format-Dataset-Converter.](Repository-URL) <img src="Icon-Image-URL"  width="3%" height="3%">

</div>

<div align="center">
  
---

Project-Description.
  
---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/Multi-Format-Dataset-Converter)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/BrenoFariasDaSilva/Multi-Format-Dataset-Converter/main)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/Multi-Format-Dataset-Converter)
![GitHub Forks](https://img.shields.io/github/forks/BrenoFariasDaSilva/Multi-Format-Dataset-Converter)
![GitHub Language Count](https://img.shields.io/github/languages/count/BrenoFariasDaSilva/Multi-Format-Dataset-Converter)
![GitHub License](https://img.shields.io/github/license/BrenoFariasdaSilva/Multi-Format-Dataset-Converter)
![GitHub Stars](https://img.shields.io/github/stars/BrenoFariasdaSilva/Multi-Format-Dataset-Converter)
![wakatime](https://wakatime.com/badge/github/BrenoFariasdaSilva/Multi-Format-Dataset-Converter.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/79cb6111c8daa897b9e7c0e941f2f28e277d7d8d.svg "Repobeats analytics image")

</div>

## Table of Contents
- [Multi-Format-Dataset-Converter. ](#multi-format-dataset-converter-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Setup](#setup)
    - [Clone the Repository](#clone-the-repository)
    - [Git](#git)
      - [Linux](#linux)
      - [MacOS](#macos)
      - [Windows](#windows)
  - [Python, Pip and Venv](#python-pip-and-venv)
    - [Linux](#linux-1)
    - [macOS](#macos-1)
    - [Windows](#windows-1)
  - [Make](#make)
    - [Linux](#linux-2)
    - [macOS](#macos-2)
    - [Windows](#windows-2)
  - [Run Programing Language Code:](#run-programing-language-code)
    - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Results - Optional](#results---optional)
  - [How to Cite?](#how-to-cite)
  - [Contributing](#contributing)
  - [Collaborators](#collaborators)
  - [License](#license)
    - [Apache License 2.0](#apache-license-20)

## Introduction

## Setup

This section provides instructions for installing Git, Python, Pip, Make, then to clone the repository (if not done yet) and all required project dependencies. 

### Clone the Repository

To clone this repository with all required submodules, use:

``` bash
git clone --recurse-submodules https://github.com/BrenoFariasdaSilva/Multi-Format-Dataset-Converter.git
```

If you clone without submodules (not recommended):

``` bash
git clone https://github.com/BrenoFariasdaSilva/Multi-Format-Dataset-Converter
```

To initialize submodules manually:

``` bash
git submodule init
git submodule update
```

If you don't have the git command, read the next subsection, which explains how to install it.

### Git

`git` is a distributed version control system that is widely used for tracking changes in source code during software development. In this project, `git` is used to download and manage the analyzed repositories, as well as to clone the project and its submodules. To install `git`, follow the instructions below based on your operating system:

#### Linux

To install `git` on Linux, run:

```bash
sudo apt install git -y
```

#### MacOS

To install `git` on MacOS, you can use Homebrew:

```bash
brew install git
```

#### Windows

On Windows, you can download `git` from the official website [here](https://git-scm.com/downloads) and follow the installation instructions provided there.

## Python, Pip and Venv

You must have Python 3, Pip, and the `venv` module installed.

### Linux

``` bash
sudo apt install python3 python3-pip python3-venv -y
```

### macOS

``` bash
brew install python3
```

### Windows

If you do not have Chocolatey installed, you can install it by running the following command in an **elevated PowerShell (Run as Administrator)**:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; `
[System.Net.ServicePointManager]::SecurityProtocol = `
    [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Once Chocolatey is installed, you can install Python using:

``` bash
choco install python3
```

Or download the installer from the official Python website.

## Make 

<!--  descripton -->

### Linux

``` bash
sudo apt install make -y
```

### macOS

``` bash
brew install make
```

### Windows

Available via Cygwin, MSYS2, or WSL.

## Run Programing Language Code:

```bash
# Command here 
```

### Dependencies

1. Install the project dependencies with the following command:

   ```bash
   make dependencies
   ```

## Usage

In order to run the project, run the following command:

```bash
make
```

This command always verify if the virtual environment and libraries are installed before running the `main.py` python code.

## Results - Optional

Discuss the results obtained in the project.

## How to Cite?

If you use the Multi-Format-Dataset-Converter in your research, please cite it using the following BibTeX entry:

```
@misc{softwareMultiFormatDatasetConverter:2025,
  title        = {Multi-Format-Dataset-Converter: Python Project for Multi-Format Dataset Cleaning and Conversion},
  author       = {Breno Farias da Silva},
  year         = {2025},
  howpublished = {\url{https://github.com/BrenoFariasdaSilva/Multi-Format-Dataset-Converter}},
  note         = {Python-based system for recursive dataset discovery, structural cleaning of text-based formats, and conversion between ARFF, CSV, Parquet, and TXT while preserving input directory layouts. Accessed on December 11, 2025}
}
```

Additionally, a `main.bib` file is available in the root directory of this repository, in which contains the BibTeX entry for this project.

If you find this repository valuable, please don't forget to give it a ⭐ to show your support! Contributions are highly encouraged, whether by creating issues for feedback or submitting pull requests (PRs) to improve the project. For details on how to contribute, please refer to the [Contributing](#contributing) section below.

Thank you for your support and for recognizing the contribution of this tool to your work!

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. If you have suggestions for improving the code, your insights will be highly welcome.
In order to contribute to this project, please follow the guidelines below or read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on how to contribute to this project, as it contains information about the commit standards and the entire pull request process.
Please follow these guidelines to make your contributions smooth and effective:

1. **Set Up Your Environment**: Ensure you've followed the setup instructions in the [Setup](#setup) section to prepare your development environment.

2. **Make Your Changes**:
   - **Create a Branch**: `git checkout -b feature/YourFeatureName`
   - **Implement Your Changes**: Make sure to test your changes thoroughly.
   - **Commit Your Changes**: Use clear commit messages, for example:
     - For new features: `git commit -m "FEAT: Add some AmazingFeature"`
     - For bug fixes: `git commit -m "FIX: Resolve Issue #123"`
     - For documentation: `git commit -m "DOCS: Update README with new instructions"`
     - For refactorings: `git commit -m "REFACTOR: Enhance component for better aspect"`
     - For snapshots: `git commit -m "SNAPSHOT: Temporary commit to save the current state for later reference"`
   - See more about crafting commit messages in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

3. **Submit Your Contribution**:
   - **Push Your Changes**: `git push origin feature/YourFeatureName`
   - **Open a Pull Request (PR)**: Navigate to the repository on GitHub and open a PR with a detailed description of your changes.

4. **Stay Engaged**: Respond to any feedback from the project maintainers and make necessary adjustments to your PR.

5. **Celebrate**: Once your PR is merged, celebrate your contribution to the project!

## Collaborators

We thank the following people who contributed to this project:

<table>
  <tr>
    <td align="center">
      <a href="#" title="defina o titulo do link">
        <img src="https://github.com/BrenoFariasdaSilva/DDoS-Detector/blob/main/.assets/Images/Github.svg" width="100px;" alt="My Profile Picture"/><br>
        <sub>
          <b>Breno Farias da Silva</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

## License

### Apache License 2.0

This project is licensed under the [Apache License 2.0](LICENSE). This license permits use, modification, distribution, and sublicense of the code for both private and commercial purposes, provided that the original copyright notice and a disclaimer of warranty are included in all copies or substantial portions of the software. It also requires a clear attribution back to the original author(s) of the repository. For more details, see the [LICENSE](LICENSE) file in this repository.
