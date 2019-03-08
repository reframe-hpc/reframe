whatis([[Name : testluamod]])
whatis([[Version : foo]])

help([==[

Description
===========
Helper module for ReFrame unit tests

]==])

setenv("TESTLUAMOD_FOO", "FOO")

family("testluamod")