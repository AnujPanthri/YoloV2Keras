import setuptools

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt","r") as f:
    reqs = f.readlines()

setuptools.setup(
    name = "YoloV2Keras",
    version ='0.0.2',
    author = "Anuj Panthri",
    author_email="panthrianuj@gmail.com",
    description="Yolo v2 library",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=reqs
)