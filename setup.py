from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name='src',
    description='A module that combines the power of Reformer/FastFormer, Electra and memory efficient compositional embeddings',
    version='0.0.1',
    author="Keshav Bhandari",
    author_email='keshavbhandari@gmail.com',
    url='https://github.com/keshavbhandari/ElectraReformer',
    keywords=[
        'transformers',
        'artificial intelligence',
        'pretraining'
    ],
    install_requires=[],
    packages=['Modules'],
    long_description=long_description,
    long_description_content_type='text.markdown',
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT'
)
