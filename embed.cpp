#include<iostream>
#include <Python.h>

int main()
{
    Py_Initialize();
    PyObject *name, *load_module, *func, *callfunc, *args;
    name=PyUnicode_FromString((char*)"BreastCancer");
    load_module=PyImport_Import(name);
    func=PyObject_GetAttrString(load_module,(char*)"Call");
    callfunc=PyObject_CallObject(func,NULL);
    return 0;
}
