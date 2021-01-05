**CMake Build**

* To build Avatar using CMake, clone the Avatar repository into a source directory:

        $ cd ~/src
        $ git clone git@gitlab.sandia.gov:avatar/avatar

* Then, create a separate build directory.

        $ mkdir -p ~/build/avatar
        $ cd ~/build/avatar

* Next, run CMake to configure Avatar.

You have two options for running CMake:

  1. Run CMake directly.  See sample-cmake-configure-script for a
     sample CMake invocation; copy that file and edit the copy,
     following the comments in the file, before running.

  2. Alternately, you may use the CMake curses client:

        $ ccmake ~/src/avatar

* Set CMAKE_BUILD_TYPE to the type of build you want to create.

  The two most common options are DEBUG and RELEASE.

* Use CMAKE_INSTALL_PREFIX to control where Avatar will be installed.

* Optionally, enable building unit tests

If you want to build unit tests, do the following:

  - Configure, build, and install the GNU Check library
    (I have only tested version 0.9.0)
  - When configuring Avatar, set CHECK_INSTALL_DIR to Check's
    installation directory (not the lib directory, but the parent
    of that, that contains the include and lib directories)

If CHECK_INSTALL_DIR is not defined, Avatar's CMake will print a
warning with instructions, but will permit the configuration and build
process to continue.

Avatar cannot distribute Check's source code as part of the Avatar
library, because Check has a GPU LGPL license.

* Build Avatar:

        $ make

* Install Avatar:

        $ make install

* The installation will include a set of binary executables, man
pages, Perl scripts, Python modules, and Python scripts.

* Run Avatar's unit tests:

Copy contents of unittest/data in Avatar's source directory, into the
unittest/data directory in Avatar's build directory.  Then, run
checkall in the unittest/ directory in Avatar's build directory.

* If you wish to use any of the Python modules / scripts, you will need to add their installation path to your PYTHONPATH environment variable (manually substituting the CMAKE_INSTALL_PREFIX you used earlier):

        export PYTHONPATH=<CMAKE_INSTALL_PREFIX>/python/packages:$PYTHONPATH


If you have questions, contact:

Philip Kegelmeyer, wpk@sandia.gov

