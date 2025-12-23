// ============================================================
// JAVASCRIPT AUTOFIX STRESS TEST - Intentional Errors
// ============================================================

// --- COMMON TYPOS ---
fucntion add(x, y) {
    retrun x + y;
}

funtion multiply(a, b) {
    reutrn a * b;
}

// --- WRONG VARIABLE DECLARATIONS ---
var conts = 10;
cosnt MAX = 100;
lte name = "test";

// --- BOOLEAN TYPOS ---
let isActive = ture;
let isEnabled = flase;
let isNull = nul;
let nothing = udnefined;

// --- MISSING SEMICOLONS ---
let x = 5
let y = 10
const z = x + y
console.log(z)

// --- MISSING COMMAS ---
let arr = [1 2 3 4 5];
let obj = { name: "test" age: 25 active: true };

// --- WRONG OPERATORS ---
if (x = 5) {
    console.log("assignment in condition");
}

// --- TRIPLE EQUALS TYPOS ---
if (x == 5) {  // Should be ===
    console.log("loose equality");
}

if (y != 10) {  // Should be !==
    console.log("loose inequality");
}

// --- WRONG ARROW FUNCTION ---
const add = (x, y) -> x + y;

// --- MISSING BRACES ---
if (x > 5)
    console.log("big")
    console.log("very big")

// --- WRONG FUNCTION SYNTAX ---
def greet(name) {
    console.log("Hello " + name);
}

fn process(data) {
    return data * 2;
}

// --- PYTHON-STYLE SYNTAX ---
for item in items:
    console.log(item)

if x == 5:
    console.log("five")

// --- COMMON METHOD TYPOS ---
consoel.log("test");
console.lgo("test");
console.wirte("test");
documnet.getElementById("x");
doucment.querySelector(".x");

// --- ARRAY METHOD TYPOS ---
arr.firlter(x => x > 5);
arr.mpap(x => x * 2);
arr.redcue((a, b) => a + b);
arr.finf(x => x === 5);
arr.incldes(5);

// --- PROMISE TYPOS ---
fetch(url).thn(res => res.json());
fetch(url).ctach(err => console.log(err));

// --- ASYNC/AWAIT TYPOS ---
asnyc function getData() {
    const data = awiat fetch(url);
    return data;
}

// --- OBJECT PROPERTY TYPOS ---
const user = {
    naem: "John",
    adress: "123 Main St",
    emial: "john@test.com"
};

// --- WRONG CLASS SYNTAX ---
class Person {
    construtor(name) {
        this.name = name;
    }
    
    getNamee() {
        return this.naem;
    }
}

// --- WRONG IMPORT SYNTAX ---
improt { Component } from 'react';
form 'lodash' import { map };

// --- WRONG JSON ---
const config = JSON.prase('{"a": 1}');
const str = JSON.stringfy(obj);

// --- WRONG EVENT HANDLERS ---
element.addEventListner('click', handler);
element.removeEventListner('click', handler);

// --- WRONG TIMEOUT ---
setTimout(() => console.log("test"), 1000);
setInteval(() => console.log("test"), 1000);
clearTimout(id);

console.log("End of JavaScript test");
