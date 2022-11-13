from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = list(
        filter(lambda x: "#" not in x, (line.strip() for line in f))
    )

setup(
    name='flowm',
    version='0.0.1',
    description="Supporting code for the flow-matching project.",
    author="Jonas Köhler, Yaoyi Chen, Andreas Krämer, Cecilia Clementi and Frank Noé",
    install_requires=[
        'importlib-metadata; python_version >= "3.8"',
    ] + install_requires,
    license="MIT",
    packages=find_packages(
        # All keyword arguments below are optional:
        where='.',  # '.' by default
        # include=['flowm'],  # ['*'] by default
        exclude=['fetch_data'],  # empty by default
    ),
    # scripts=[],
)

