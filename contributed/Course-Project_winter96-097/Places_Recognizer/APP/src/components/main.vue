<template> 
  <Page class="page">  <!-- Main page -->
    <ActionBar class="action-bar" title="Places Recognition model">
    <ActionItem 
      android.position="position"
      icon="res://settings"
      @tap="promptEndPoint"/>
    </ActionBar>
    <!-- <ScrollView> -->
      <FlexboxLayout flexDirection="column" class="container">
        <Label class="body" textWrap=true text="welcome, please take a picture and make it hard, our model is perfectly trained :)" alignSelf="center"/>
        <predict class="upload" :endPoint="endPoint" flexShrink="50"></predict>
        <Label class="body-bold" textWrap=true text="SRTTU : Computer Vision" alignSelf="center"/>
        <Label class="body-small" textWrap=true text="Mahya Mahdian & Mohammad Hassan Sattarian" alignSelf="center"/>
      </FlexboxLayout>
    <!-- </ScrollView> -->
  </Page>
</template>

<script>
import predict from './predict';

export default {
  data () {
    return {
      predictPage: predict,
      endPoint: ''
    }
  },
  mounted(){
    
  },
  components: {
    predict
  },
  methods: {
    promptEndPoint() { // prompt for getting end-point from user 
      let _this = this;
      prompt({
        title: "Enter End-point address",
        message: "example: http://endpoint.com:8080",
        okButtonText: "set",
        cancelButtonText: "cancel",
        defaultText: "http://",
      })
      .then(result => {
        if (!result.result)
          alert('no end-point recieved!');
        else
          _this.endPoint = result.text;
      });
    }
  }
}
</script>

<style scoped>
  .container {
    margin: 20;
  }

  .body-bold {
    font: bold;
  }
  
  .body-small {
    font-size: small;
  }
</style>
