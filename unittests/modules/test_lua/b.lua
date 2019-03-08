whatis([[Name : test_lua]])
whatis([[Version : b]])

help([==[

Description
===========
Helper module for ReFrame unit tests

]==])

setenv("TEST_LUA_B", "B")

family("testluamod")