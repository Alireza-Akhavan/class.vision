import Vue from 'nativescript-vue';
import axios from 'axios'
import VueAxios from 'vue-axios'

Vue.use(VueAxios, axios)

import App from './App';

import './styles.scss';

// Uncommment the following to see NativeScript-Vue output logs
// Vue.config.silent = false;

new Vue({

  render: h => h(App)

}).$start();
