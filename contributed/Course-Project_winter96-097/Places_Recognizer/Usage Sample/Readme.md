# Predict

## Local Use

### Predict an image given it's address

This script takes an image address, and will predict the image by loading the saved trained model.

#### usage

```bash
python predict.py [ImagesAddress]
```

#### note

if using tensorflow-gpu as keras backend,
tensorflow's logs can be quiet by setting this environment variable:

```CMD
set TF_CPP_MIN_LOG_LEVEL = 3
```

## NonLocal Use

you can run web server in your computer or a remote server, and predict intended images using specified End-points.
a flask server to handle uploading and returning predict result is provided in ```serve.py``` file.

```bash
# this will run on host IP in port 8080
python serve.py
```

#### note 

for uploading images, image is encoded as base64 and places in request headers with key of **file**.

```bash
# example
curl Host:ip -H "file:/9j/4AAQSkZJRgABAQEAYABgAA..."
```