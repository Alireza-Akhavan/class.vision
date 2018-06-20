<template>
<FlexboxLayout flexDirection="column">
  <ActivityIndicator :busy="busy"/>
  <Label v-if="imageSrc" :text="status" alignSelf="center"></Label>
  <Label v-if="imageSrc" v-show="endPoint" :text="endPoint" alignSelf="center"></Label>
  <Image :src="imageSrc" style="width:300;height:300" flexShrink="50"/>
  <FlexboxLayout justifyContent="space-around" alignItems="center">
    <Button @tap="takePic()" text="Take Picture" alignSelf="flex-start" width="90%"/>
    <Image class="cfg" src="res://gallery" width="10%" alignSelf="flex-end" @tap="chooseFromGallery()"/>
  </FlexboxLayout>
  <Button class="btn btn-primary btn-purple" text="Predict" @tap="uploadFile" />
</FlexboxLayout>
</template>

<script>
import { Image } from "ui/image";
import * as camera from "nativescript-camera";
var http = require("http");
const imageSourceModule = require("tns-core-modules/image-source");
const fileSystemModule = require("tns-core-modules/file-system");
var enumsModule  = require('ui/enums');
var imagepicker = require("nativescript-imagepicker");
var context = imagepicker.create({ mode: "single" }); // use "multiple" for multiple selection

export default {
  props: {
    endPoint: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      permisioned: false,
      imageSrc: null,
      busy: false,
      status: 'image ready to upload',
    }
  },
  computed: {
  },
  methods: {
    takePic: function() {
      camera.requestPermissions();
      this.permisioned = true;
      camera
        .takePicture({ width: 300, height: 300, keepAspectRatio: true })
        .then(
          imageAsset => {
            this.imageSrc = imageAsset;
          },
          error => console.log(error)
        )
        .catch(err => {
          console.log("Error -> " + err.message);
        });
    },
    uploadFile() {
      if (!this.endPoint) alert('please enter an end-point by taping on config button on the top right corner.');
      let imageAsBase64String = '';
      let imageAsset = this.imageSrc;
      let _this = this;
      imageSourceModule.fromAsset(imageAsset).then(imageFromAsset => {
        imageAsBase64String = imageFromAsset.toBase64String("jpg");
        _this.busy = true;
        _this.status = 'uploading ...';
        _this.axios.post(_this.endPoint, {
          headers: {
            file: imageAsBase64String 
          }
        }).then(response=>{
          _this.busy = false;
          _this.status = 'image ready to upload';
          if (response.status == 200) {
            alert(response.data)
          }
          else {
            alert('Unfortunately somthing went wrong, try again.')
          }
        });
      });
    },
    chooseFromGallery(){
      let _this = this;
      context
      .authorize()
      .then(function() {
          console.log('level 1');
          return context.present();
      })
      .then(function(selection) {
          selection.forEach(function(selected) {
            console.log(typeof(selected));
            console.log('level 2');
            _this.imageSrc = selected;
            console.log('level 3');
          });
          // list.items = selection;
      }).catch(function (e) {
          // process error
      });
    }
  }
};
</script>

<style>
  .cfg {
    margin-bottom: 30px;
  }
</style>
