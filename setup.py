from setuptools import setup, find_packages

setup(
    name="cell_extension_orientation",
    version="0.1.0",
    author="Sindi Nexhipi, Deborah Schmidt",
    description="Package for cell extension orientation analysis",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy==2.1.3",
        "tifffile==2024.9.20",
        "pandas==2.2.3",
        "matplotlib-scalebar==0.8.1",
        "scikit-image==0.24.0",
        "openpyxl==3.1.5",
        "xlsxwriter==3.2.0",
        "xlrd==2.0.1"
    ],
    entry_points={
        'console_scripts': [
            'cell-extension-orientation=cell_extension_orientation.main:run',
        ],
    },
)