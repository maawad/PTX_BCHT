# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/mawad/.conda/envs/cmake-env/bin/cmake

# The command to remove a file.
RM = /home/mawad/.conda/envs/cmake-env/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mawad/github/PTX_BCHT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mawad/github/PTX_BCHT/build

# Include any dependencies generated for this target.
include CMakeFiles/ptx_cuckoo_hashtable.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ptx_cuckoo_hashtable.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ptx_cuckoo_hashtable.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ptx_cuckoo_hashtable.dir/flags.make

CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o: CMakeFiles/ptx_cuckoo_hashtable.dir/flags.make
CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o: ../ptx_cuckoo_hashtable.cpp
CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o: CMakeFiles/ptx_cuckoo_hashtable.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mawad/github/PTX_BCHT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o -MF CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o.d -o CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o -c /home/mawad/github/PTX_BCHT/ptx_cuckoo_hashtable.cpp

CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mawad/github/PTX_BCHT/ptx_cuckoo_hashtable.cpp > CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.i

CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mawad/github/PTX_BCHT/ptx_cuckoo_hashtable.cpp -o CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.s

# Object files for target ptx_cuckoo_hashtable
ptx_cuckoo_hashtable_OBJECTS = \
"CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o"

# External object files for target ptx_cuckoo_hashtable
ptx_cuckoo_hashtable_EXTERNAL_OBJECTS =

ptx_cuckoo_hashtable: CMakeFiles/ptx_cuckoo_hashtable.dir/ptx_cuckoo_hashtable.cpp.o
ptx_cuckoo_hashtable: CMakeFiles/ptx_cuckoo_hashtable.dir/build.make
ptx_cuckoo_hashtable: /usr/lib/x86_64-linux-gnu/libcuda.so
ptx_cuckoo_hashtable: /usr/local/cuda-11.2/lib64/libcudart_static.a
ptx_cuckoo_hashtable: /usr/lib/x86_64-linux-gnu/librt.so
ptx_cuckoo_hashtable: CMakeFiles/ptx_cuckoo_hashtable.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mawad/github/PTX_BCHT/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ptx_cuckoo_hashtable"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ptx_cuckoo_hashtable.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ptx_cuckoo_hashtable.dir/build: ptx_cuckoo_hashtable
.PHONY : CMakeFiles/ptx_cuckoo_hashtable.dir/build

CMakeFiles/ptx_cuckoo_hashtable.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ptx_cuckoo_hashtable.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ptx_cuckoo_hashtable.dir/clean

CMakeFiles/ptx_cuckoo_hashtable.dir/depend:
	cd /home/mawad/github/PTX_BCHT/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mawad/github/PTX_BCHT /home/mawad/github/PTX_BCHT /home/mawad/github/PTX_BCHT/build /home/mawad/github/PTX_BCHT/build /home/mawad/github/PTX_BCHT/build/CMakeFiles/ptx_cuckoo_hashtable.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ptx_cuckoo_hashtable.dir/depend

