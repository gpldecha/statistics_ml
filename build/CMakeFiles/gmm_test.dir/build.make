# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/guillaume/roscode/catkin_ws/src/statistics_ml

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/guillaume/roscode/catkin_ws/src/statistics_ml/build

# Include any dependencies generated for this target.
include CMakeFiles/gmm_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gmm_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gmm_test.dir/flags.make

CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o: CMakeFiles/gmm_test.dir/flags.make
CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o: ../src/test/gmm_test.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/guillaume/roscode/catkin_ws/src/statistics_ml/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o -c /home/guillaume/roscode/catkin_ws/src/statistics_ml/src/test/gmm_test.cpp

CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/guillaume/roscode/catkin_ws/src/statistics_ml/src/test/gmm_test.cpp > CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.i

CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/guillaume/roscode/catkin_ws/src/statistics_ml/src/test/gmm_test.cpp -o CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.s

CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o.requires:
.PHONY : CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o.requires

CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o.provides: CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o.requires
	$(MAKE) -f CMakeFiles/gmm_test.dir/build.make CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o.provides.build
.PHONY : CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o.provides

CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o.provides.build: CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o

# Object files for target gmm_test
gmm_test_OBJECTS = \
"CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o"

# External object files for target gmm_test
gmm_test_EXTERNAL_OBJECTS =

devel/lib/statistics_ml/gmm_test: CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o
devel/lib/statistics_ml/gmm_test: CMakeFiles/gmm_test.dir/build.make
devel/lib/statistics_ml/gmm_test: devel/lib/libstatistics_ml.so
devel/lib/statistics_ml/gmm_test: /opt/ros/indigo/lib/libroscpp.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libboost_signals.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
devel/lib/statistics_ml/gmm_test: /opt/ros/indigo/lib/librosconsole.so
devel/lib/statistics_ml/gmm_test: /opt/ros/indigo/lib/librosconsole_log4cxx.so
devel/lib/statistics_ml/gmm_test: /opt/ros/indigo/lib/librosconsole_backend_interface.so
devel/lib/statistics_ml/gmm_test: /usr/lib/liblog4cxx.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libboost_regex.so
devel/lib/statistics_ml/gmm_test: /opt/ros/indigo/lib/libroscpp_serialization.so
devel/lib/statistics_ml/gmm_test: /opt/ros/indigo/lib/librostime.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
devel/lib/statistics_ml/gmm_test: /opt/ros/indigo/lib/libxmlrpcpp.so
devel/lib/statistics_ml/gmm_test: /opt/ros/indigo/lib/libcpp_common.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libboost_system.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libboost_thread.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libpthread.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libconsole_bridge.so
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
devel/lib/statistics_ml/gmm_test: /usr/local/lib/libarmadillo.so
devel/lib/statistics_ml/gmm_test: CMakeFiles/gmm_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable devel/lib/statistics_ml/gmm_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gmm_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gmm_test.dir/build: devel/lib/statistics_ml/gmm_test
.PHONY : CMakeFiles/gmm_test.dir/build

CMakeFiles/gmm_test.dir/requires: CMakeFiles/gmm_test.dir/src/test/gmm_test.cpp.o.requires
.PHONY : CMakeFiles/gmm_test.dir/requires

CMakeFiles/gmm_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gmm_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gmm_test.dir/clean

CMakeFiles/gmm_test.dir/depend:
	cd /home/guillaume/roscode/catkin_ws/src/statistics_ml/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/guillaume/roscode/catkin_ws/src/statistics_ml /home/guillaume/roscode/catkin_ws/src/statistics_ml /home/guillaume/roscode/catkin_ws/src/statistics_ml/build /home/guillaume/roscode/catkin_ws/src/statistics_ml/build /home/guillaume/roscode/catkin_ws/src/statistics_ml/build/CMakeFiles/gmm_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gmm_test.dir/depend

