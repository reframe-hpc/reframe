whatis([[Name : testluamod]])
whatis([[Version : bar]])

help([==[

Description
===========
Helper module for ReFrame unit tests

]==])

setenv("TESTLUAMOD_BAR", "BAR")

family("testluamod")