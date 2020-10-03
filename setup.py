import os
from distutils.core import setup, Extension
#from setuptools import find_packages

os.environ["CC"] = "g++"

module = Extension('extractFrames',
                    define_macros = [('NOMAIN', None),
                                     ('USE_PYTHON', None)],
                    include_dirs = ['/usr/include',
                                    '/usr/local/include',
                                    '/usr/local/include/opencv4/'],
                    libraries = ['opencv_gapi',
                                 'opencv_photo',
                                 'opencv_highgui',
                                 'opencv_imgcodecs',
                                 'opencv_stitching',
                                 'opencv_core',
                                 'opencv_videoio',
                                 'opencv_dnn',
                                 'opencv_video',
                                 'opencv_imgproc',
                                 'opencv_ml',
                                 'opencv_features2d',
                                 'opencv_objdetect',
                                 'opencv_flann',
                                 'stdc++fs'],
                    library_dirs = ['/usr/local/lib',
                                    '/usr/lib'],
                    language = "c++",
                    extra_compile_args = ['-std=c++17'],
                    sources = ['extractFrames.cpp'])

setup (name = 'extractFrames',
       version = '1.0',
       description = 'This package is for frames extraction from a video.',
       author = 'Baptiste Roseau',
       python_requires='>3.5',
       ext_modules = [module])