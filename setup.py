from setuptools import (
    setup,
    find_packages,
)


def get_requirements(filenames):
    r_total = []
    for filename in filenames:
        with open(filename) as f:
            r_local = f.read().splitlines()
            r_total.extend(r_local)
    return r_total

setup(
    name='arelight',
    version='0.25.0',
    description='About Mass-media text processing application for your '
                'Relation Extraction task, powered by AREkit.',
    url='https://github.com/nicolay-r/ARElight',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Nicolay Rusnachenko',
    author_email='rusnicolay@gmail.com',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='natural language processing, relation extraction, sentiment analysis',
    packages=find_packages(),
    package_dir={'src': 'src'},
    install_requires=get_requirements(['dependencies.txt']),
    data_files=["logo.png"],
)