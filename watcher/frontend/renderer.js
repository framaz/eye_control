// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process because
// `nodeIntegration` is turned off. Use `preload.js` to
// selectively enable features needed in the rendering
// process.
let states = [0, 1, 2, 3, 4];
let cur_state = 0;

const result = document.querySelector('#test');
console.log(require("zerorpc"))
const zerorpc = require("zerorpc");

let client = new zerorpc.Client();
result.textContent = "aaa";
client.connect("tcp://127.0.0.1:4242");
let pic = document.querySelector('#pic');
func = () => {
    let x = x_coord.value
    let y = y_coord.value
    client.invoke("request", x, y, function(error, res, more) {
        pic.src = 'data:image/png;base64,' + res;
    })
}
client.invoke("request", 1, 1, function(error, res, more) { var asd = 1});
let x_coord = document.querySelector('#x');
let y_coord = document.querySelector('#y');

let corner_button = document.querySelector('#corner_change')

corner_change.textContent = cur_state;
button_func = () => {
    cur_state = cur_state + 1;

    corner_change.textContent = cur_state;
    let x = x_coord.value
    let y = y_coord.value
    client.invoke("new_corner", x, y, function(error, res, more) {})
    func()
}

corner_button.addEventListener('click', button_func)
x_coord.addEventListener('change', func);
y_coord.addEventListener('change', func);