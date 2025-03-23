cmake_minimum_required(VERSION 3.10)
project(vfx_project)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find FFmpeg package
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswresample)

include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIBRARY_DIRS})

# Find FFTW package
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)

include_directories(${FFTW_INCLUDE_DIRS})
link_directories(${FFTW_LIBRARY_DIRS})

# Find SFML package
find_package(SFML 3.0.0 COMPONENTS Audio REQUIRED)
include_directories(${SFML_INCLUDE_DIR}, /usr/local/include)
link_directories(${SFML_LIBRARY_DIR}, /usr/local/lib)
# link static libraries
set(SFML_LIBRARIES sfml-audio)

# Add executable
add_executable(vfx02 vfx02.cc)

# Linking flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/local/lib")

# Link libraries
target_link_libraries(vfx02 ${OpenCV_LIBS} ${FFMPEG_LIBRARIES} ${FFTW_LIBRARIES} ${SFML_LIBRARIES})
