Getting Started
===============

Installation
------------

1. Open **VS Code** and select **Terminal > New Terminal**.

2. Navigate to the folder where you want to download the repository:

   .. code-block:: powershell

      cd path\to\your\projects

3. Clone the repository and enter its directory:

   .. code-block:: powershell

      git clone https://github.com/poulet-lab/poulet_py.git
      cd poulet_py

4. Fetch branches and switch to the branch you want to work on (replace ``<branch_name>`` with the actual branch name):

   .. code-block:: powershell

      git fetch origin
      git switch --track origin/<branch_name>

   If the branch already exists locally, use ``git switch <branch_name>`` instead.

5. Create a new Micromamba environment:

   .. code-block:: powershell

      micromamba create -n poulet_py python=3.11 pip -y

6. Activate the environment:

   .. code-block:: powershell

      micromamba activate poulet_py

7. Install ``poulet_py`` in editable mode from the repository root:

   .. code-block:: powershell

      pip install -U -e .[all]

8. Open the repository in VS Code (if not already open):

   .. code-block:: powershell

      code .

9. In VS Code, open the Command Palette (``Ctrl+Shift+P``), choose **Python: Select Interpreter**, and select the ``poulet-py`` environment.

10. Check the active branch and installation:

    .. code-block:: powershell

       git branch --show-current
       python -m pip show poulet-py

Basic Usage
-----------

Import the package in Python:

.. code-block:: python

   import poulet_py

Changes made to the local package source are available through the editable installation. Restart your Python session after editing imported code if needed.
