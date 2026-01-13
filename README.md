# Hornet Detector

## Installation

### Install [uv](https://github.com/astral-sh/uv) to manage Python versions and virtual environments.
Use curl to download the script and execute it with sh:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
If your system doesn't have curl, you can use wget:
```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```
### Install Python version and dependencies
You actually don't have to do anything: `uv` will automatically setup the right Python version, 
the virtual environment and install the dependencies as soon as you run something in the project for the first time.

## Run project
To run a script, use the following command:
```bash
uv run <script_name.py>
```
`uv` will automatically install and use the correct Python version, virtual environment and dependencies.


### Development
To run linting and formatting checks, use the following commands:
```bash
uv run ruff check
uv run ruff format
```