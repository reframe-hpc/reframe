#!/usr/bin/env ruby

require "rubygems"
require 'gsl'

g1 = GSL::Vector.alloc(1,2,3,4)
a1 = g1.to_na

p a1
