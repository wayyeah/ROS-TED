# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nvidia/way/catkin_ws_ted/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nvidia/way/catkin_ws_ted/build

# Include any dependencies generated for this target.
include centerpoint/CMakeFiles/boost.dir/depend.make

# Include the progress variables for this target.
include centerpoint/CMakeFiles/boost.dir/progress.make

# Include the compile flags for this target's objects.
include centerpoint/CMakeFiles/boost.dir/flags.make

centerpoint/CMakeFiles/boost.dir/boost_dummy.c.o: centerpoint/CMakeFiles/boost.dir/flags.make
centerpoint/CMakeFiles/boost.dir/boost_dummy.c.o: centerpoint/boost_dummy.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nvidia/way/catkin_ws_ted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object centerpoint/CMakeFiles/boost.dir/boost_dummy.c.o"
	cd /home/nvidia/way/catkin_ws_ted/build/centerpoint && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/boost.dir/boost_dummy.c.o   -c /home/nvidia/way/catkin_ws_ted/build/centerpoint/boost_dummy.c

centerpoint/CMakeFiles/boost.dir/boost_dummy.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/boost.dir/boost_dummy.c.i"
	cd /home/nvidia/way/catkin_ws_ted/build/centerpoint && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/nvidia/way/catkin_ws_ted/build/centerpoint/boost_dummy.c > CMakeFiles/boost.dir/boost_dummy.c.i

centerpoint/CMakeFiles/boost.dir/boost_dummy.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/boost.dir/boost_dummy.c.s"
	cd /home/nvidia/way/catkin_ws_ted/build/centerpoint && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/nvidia/way/catkin_ws_ted/build/centerpoint/boost_dummy.c -o CMakeFiles/boost.dir/boost_dummy.c.s

# Object files for target boost
boost_OBJECTS = \
"CMakeFiles/boost.dir/boost_dummy.c.o"

# External object files for target boost
boost_EXTERNAL_OBJECTS =

/home/nvidia/way/catkin_ws_ted/devel/lib/libboost.a: centerpoint/CMakeFiles/boost.dir/boost_dummy.c.o
/home/nvidia/way/catkin_ws_ted/devel/lib/libboost.a: centerpoint/CMakeFiles/boost.dir/build.make
/home/nvidia/way/catkin_ws_ted/devel/lib/libboost.a: centerpoint/CMakeFiles/boost.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nvidia/way/catkin_ws_ted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C static library /home/nvidia/way/catkin_ws_ted/devel/lib/libboost.a"
	cd /home/nvidia/way/catkin_ws_ted/build/centerpoint && $(CMAKE_COMMAND) -P CMakeFiles/boost.dir/cmake_clean_target.cmake
	cd /home/nvidia/way/catkin_ws_ted/build/centerpoint && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/boost.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
centerpoint/CMakeFiles/boost.dir/build: /home/nvidia/way/catkin_ws_ted/devel/lib/libboost.a

.PHONY : centerpoint/CMakeFiles/boost.dir/build

centerpoint/CMakeFiles/boost.dir/clean:
	cd /home/nvidia/way/catkin_ws_ted/build/centerpoint && $(CMAKE_COMMAND) -P CMakeFiles/boost.dir/cmake_clean.cmake
.PHONY : centerpoint/CMakeFiles/boost.dir/clean

centerpoint/CMakeFiles/boost.dir/depend:
	cd /home/nvidia/way/catkin_ws_ted/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nvidia/way/catkin_ws_ted/src /home/nvidia/way/catkin_ws_ted/src/centerpoint /home/nvidia/way/catkin_ws_ted/build /home/nvidia/way/catkin_ws_ted/build/centerpoint /home/nvidia/way/catkin_ws_ted/build/centerpoint/CMakeFiles/boost.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : centerpoint/CMakeFiles/boost.dir/depend

