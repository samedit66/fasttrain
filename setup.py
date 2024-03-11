from setuptools import setup, find_packages


def readme():
    with open('README.md', 'r') as f:
        return f.read()


setup(
    name='fasttrain',
    version='0.0.6',
    author='samedit66',
    author_email='samedit66@yandex.ru',
    description='Framework for building training loops easier and faster',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/samedit66/fasttrain',
    license='Apache 2.0 License',
    packages=find_packages(),
    install_requires=[
        'torch',
        'tqdm',
        'numpy',
        'matplotlib',
        'scipy',
        ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        ],
    keywords='python torch pytorch',
    python_requires='>=3.7'
)