whatis([[Name : test_lua]])
whatis([[Version : a]])

help([==[

Description
===========
Helper module for ReFrame unit tests

]==])

setenv("TEST_LUA_A", "A")

family("testluamod")