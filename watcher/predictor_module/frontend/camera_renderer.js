// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process because
// `nodeIntegration` is turned off. Use `preload.js` to
// selectively enable features needed in the rendering
// process.
const zerorpc = require("zerorpc");
let client = new zerorpc.Client();
client.connect("tcp://127.0.0.1:4243");
let pic = document.querySelector('#pic');

let div_all = document.querySelector('#all');
create_listener = (target, tag) => {
    let kek = tag;
    let asd = target;
    return function() {
        client.invoke("set_attribute", asd, kek.value, function(error, res, more) {
        });
    }
}
client.invoke("get_attributes", function(error, res, more) {
    for(var key in res){
        var value = res[key];
        one_div = document.createElement('div');
        one_div.append(value[1]+":");
        input = document.createElement('input');
        input.value = value[0];
        input.addEventListener('change', create_listener(key, input));
        one_div.append(input);
        div_all.append(one_div);
    }
});

one_div = document.createElement('div');
one_div.innerHTML = "loshara";
//let z_rotation = document.querySelector('#z_rotation');

//z_rotation.addEventListener('change', rotate_eyes);

func = () => {
    client.invoke("get_frame", function(error, res, more) {
        pic.src = 'data:image/png;base64,' + res;
    });
}
let timerId = setInterval(func, 300)