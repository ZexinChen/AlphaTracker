# Installation notes for Sloth labeling tool


We recommend using the Anaconda Prompt instead of the cmd window on a Windows machine.  Sloth is an older tool and requires some special handling to run correctly.
1. Create a sloth environment with the command:

   `conda create -n sloth-test python=2.7`
2. Activate it with the command:

   `conda activate sloth-test`
3. Get the wheel for the PyQt4 for python 2.7 version from this website: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4
    For example, if you are running Windows x32, choose this one: PyQt4‑4.11.4‑cp27‑cp27m‑win32.whl
4. After it downloads, please move it to the AnnotationTool  folder
5. Now you have to install it with the command:

   `pip install PyQt4‑4.11.4‑cp27‑cp27m‑win32.whl`
(or whatever the version you selected)
6. There are a few other things to install: 

   `pip install numpy`

   `pip install pillow`
7.  Next, change to the tool directory and install sloth with these two commands:

   `cd tool`

   `python .\sloth-master\setup.py install`
  
Sloth should now be installed.  To run it type:

   `python .\sloth-master\sloth\bin\sloth`
and you should see a window open.
