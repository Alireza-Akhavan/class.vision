# Predict

## Predict an image given it's address

This script takes an image address, and will predict the image by loading the saved trained model.

### usage

```bash
python predict.py [ImagesAddress]
```

#### note

if using tensorflow-gpu as keras backend,
tensorflow's logs can be quiet by setting this environment variable:

```CMD
set TF_CPP_MIN_LOG_LEVEL = 3
```