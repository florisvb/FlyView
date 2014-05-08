from distutils.core import setup

setup(
    name='FlyView',
    version='0.0.1',
    author='Floris van Breugel',
    author_email='florisvb@gmail.com',
    packages = ['flyview'],
    license='BSD',
    description='fly perspective rendering from rectilinear images, based on code by Andrew Straw',
    long_description=open('ReadMe.md').read(),
)



