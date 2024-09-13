from setuptools import setup, find_packages

VERSION = '0.1.1' 
DESCRIPTION = 'System Regression Code'


# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="system_regression", 
        version=VERSION,
        author="junmingcao",
        author_email="<junmingcao@foxmail.com>",
        description=DESCRIPTION,
        packages=find_packages(),
        # install_requires=get_requirements(), # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ]
)