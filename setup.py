import os
import platform
import subprocess
import time

import numpy as np
from setuptools import find_packages, setup

MAJOR = 0
MINOR = 1
PATCH = 0
SUFFIX = ''
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

version_file = 'version.py'


def readme():
    with open('README.md') as f:
        content = f.read()
    return content


def get_git_hash():

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from alphapose.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
short_version = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION))


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if __name__ == '__main__':
    write_version_py()
    setup(
        name='alphatracker',
        version=get_version(),
        description='Code for AlphaTracker',
        long_description=readme(),
        keywords='multi-animal tracking, pose estimation, unsupervised behavioral clustering',
        url='https://github.com/MVIG-SJTU/AlphaTracker',
        packages=find_packages(exclude=('data', 'exp',)),
        package_data={'': ['*.json', '*.txt']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
        license='GPLv3',
        python_requires=">=3",
        setup_requires=['pytest-runner', 'numpy'],
        tests_require=['pytest'],
        install_requires=[
            'numpy', 'scipy', 'opencv-python', 'matplotlib',
            'tqdm', 'googledrivedownloader', 'h5py', 'pandas',
            'nibabel', 'pycocotools', 'tensorboardx', 'munkres',
            'visdom', 'scikit-learn', 'seaborn', 'umap'
        ],
        zip_safe=False)
    
    print('\n Install `torchsample` ...')
    cmd = 'python -m pip install git+https://github.com/ZexinChen/torchsample'
    os.system(cmd)