# Getting Started
## Installation
### Minimal Installation
```shell
pip install -U poulet-py
```
### Optional Dependencies
Install specific modules:
```shell
pip install -U poulet-py[hardware]          # Only hardware dependencies
pip install -U poulet-py[tools,utils]       # Tools + utilities

pip install -U poulet-py[all]               # Everything
```
### Development Mode (Editable Install)
1. Open a terminal, either within your IDE (e.g., VS Code, PyCharm) or in a separate terminal window.
2. Navigate to the directory where you want to store the repository:
```shell
cd path\to\your\projects
```
3. Clone the repository and navigate into it:
```shell
git clone https://github.com/poulet-lab/poulet_py.git
cd poulet_py
```
4. Create a python environment (We use micromamba):
```shell
micromamba create -n poulet_py python=3.13 pip -y
```
5. Activate the environment:
```shell
micromamba activate poulet_py
```
6. Install `poulet_py` in editable mode from the repository root:
```shell
pip install -U -e .[all]
```
7. Fetch the branch you want to work on or create a new one (replace `\<branch_name>` with the actual branch name):
```shell
git fetch --all
git switch -c origin/\<branch_name>
```
#### Note
You might need to run step 6 again if the branch introduce new dependencies.

Changes made to the local package source are available through the editable installation.<br> Restart your Python session after editing imported code if needed.
#### Tips
* If using an IDE, configure its Python interpreter to use the `poulet_py` environment.<br>
In VS Code, this can be done through the Command Palette (`Ctrl+Shift+P`) by selecting **Python: Select Interpreter** and choosing the `poulet_py` environment.
* Verify the active branch and installation:
  ```shell
  git branch --show-current
  python -m pip show poulet-py
  ```
## Basic Usage
Import the package in Python:
```python
import poulet_py
```
