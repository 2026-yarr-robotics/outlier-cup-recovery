from setuptools import find_packages, setup
from glob import glob

package_name = 'dsr_practice'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ssu',
    maintainer_email='ssu@todo.todo',
    description='Fallen cup recovery: stand_fallen_cup MoveIt node for the Doosan M0609 + RG2.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'stand_fallen_cup = dsr_practice.stand_fallen_cup:main',
            'place_mouth_up_cup = dsr_practice.place_mouth_up_cup:main',
            'outlier_cup_recovery = dsr_practice.outlier_cup_recovery:main',
        ],
    },
)
