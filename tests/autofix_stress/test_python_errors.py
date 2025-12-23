# ============================================================
# PYTHON AUTOFIX STRESS TEST - Intentional Errors
# ============================================================

# --- COMMON TYPOS ---
def caluclate(x, y):
    retrun x + y

def proccess_data(items):
    resutl = []
    for itme in items:
        resutl.append(itme * 2)
    retunr resutl

# --- INDENTATION ERRORS ---
def bad_indent():
result = 10
    return result

# --- MISSING COLONS ---
def no_colon()
    pass

if x == 5
    print("five")

for i in range(10)
    print(i)

while True
    break

# --- WRONG OPERATORS ---
if x === 5:
    print("wrong")

if y !== 10:
    print("also wrong")

# --- UNDEFINED VARIABLES ---
print(undefind_var)
result = somefunction(x)

# --- WRONG STRING QUOTES ---
message = "Hello World'
path = 'C:\Users\test"

# --- MISSING PARENTHESES ---
print "Hello"
print "World"

# --- WRONG IMPORTS ---
improt os
form sys import path
import systm

# --- COMMON TYPOS ---
pritn("Hello")
prnit("World")
pirnt("Test")

lenght = len(items)
langth = len(data)

# --- WRONG BOOLEAN ---
is_active = Ture
is_enabled = Flase
value = Noen

# --- MISSING COMMAS ---
data = [1 2 3 4 5]
config = {"a": 1 "b": 2}

# --- SEMICOLONS (not needed in Python) ---
x = 5;
y = 10;

# --- WRONG FUNCTION DEFINITION ---
function test():
    pass

func another():
    pass

# --- BRACES INSTEAD OF INDENTATION ---
if x > 5 {
    print("big")
}

# --- WRONG CLASS SYNTAX ---
class Person {
    def __init__(self):
        pass
}

# --- COMMON LIBRARY TYPOS ---
import numpyy as np
import pandsa as pd
import matplotib.pyplot as plt

# --- EXCEPTION HANDLING TYPOS ---
try:
    x = 1/0
execpt ZeroDivisionError:
    print("error")
finaly:
    print("done")

# --- WRONG COMPREHENSION ---
squares = [x**2 for x in ragne(10)]

# --- WRONG DECORATORS ---
@staticmethd
def my_static():
    pass

@classmethd
def my_class(cls):
    pass

print("End of Python test")
