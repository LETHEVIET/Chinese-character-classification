import gdown
import tarfile
# Download chinese dataset
url = "https://drive.google.com/uc?id=187ID8Bx_tawb7vMiXkteXsyI9Z6kcXg0"
output = "./etl_952_singlechar_size_64.tar.gz"
gdown.download(url, output)

# unpack the file
thetarfile = tarfile.open(output)
thetarfile.extractall()