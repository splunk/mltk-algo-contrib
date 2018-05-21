from setuptools import setup, find_packages


setup(
    name="mltk_algo_contrib",
    version="0.1.0",
    url="https://github.com/splunk/mltk-algo-contrib",
    description="Machine Learning Toolkit Algorithms",
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[],    # For now
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: Apache Software License',
    ],
)
