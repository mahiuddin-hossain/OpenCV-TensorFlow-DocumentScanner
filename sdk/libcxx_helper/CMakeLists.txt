cmake_minimum_required(VERSION 3.6)

project(opencv_jni_shared)

# ১৬ KB মেমরি পেজ অ্যালাইনমেন্ট নিশ্চিত করার জন্য লিঙ্কার ফ্ল্যাগ
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")

# dummy target to bring libc++_shared.so into packages
add_library(opencv_jni_shared STATIC dummy.cpp)
