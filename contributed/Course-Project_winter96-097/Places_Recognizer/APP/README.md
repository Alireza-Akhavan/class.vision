# SRU:Places Recognizer App

A Graphical interface for using SRU:Places model to predict places .

written using nativescript-vue framework allowing to create native cross-platform applications using familier javascript
and supports for progressive frameworks like vue.

So this is just a javasctipt project, more specifically, a Vue framework application that is compiled using nativescript to Android. (while this project just focused on Android app, it is easily possible to compile IOS app as well)

for furture information visit [Nativescript-Vue Documantations]('http://nativescript-vue.org/en/docs/getting-started/').

>Using the app is quite simple, just enter the predicter server address by **taping the setting icon** at the top-right corner. then take a picture by taping *take picture* button or choose an image from gallery by taping on *gallery icon*. after image appeared in main page of the app tap **Predict** and wait for the image to upload. result would pop-up after few seconds.

## Downloads

  Download the app from build directory or just click [***HERE***](http://mhsattarian.com/projects/Places%20Recognizer.apk).

## Build from the source

Prerequisite are:

- Android SDK : API version 22 to 26
- Gradle 3.2.1
- JDK8

>As for now that above softwares are all unaccessible for Iranians developers, recomended way to install these packages is avoid installing them seperately using command-line methods (link SDKMAN, Chocolatey, etc.), unless you have the knowledge to set a proxy for this softwares and downloading them using Android Studio SDK manager, again using recommended [FOD (Freedom Of Developers) Proxy](https://github.com/freedomeofdevelopers/fod/).

### Usage

``` bash
# Install dependencies
npm install

# Build for production
npm run build
npm run build:<platform>

# Build, watch for changes and debug the application
npm run debug
npm run debug:<platform>

# Build, watch for changes and run the application
npm run watch
npm run watch:<platform>

# Clean the NativeScript application instance (i.e. rm -rf dist)
npm run clean
```
